# pi05_droid_jointpos: JAX to PyTorch conversion runlog

Plan-2 Task 4. Converts the openpi `pi05_droid_jointpos` checkpoint from its
native JAX/Flax format to PyTorch so it can be loaded by
`openpi.policies.policy_config.create_trained_policy` with `is_pytorch=True`,
and verifies the converted weights are faithful to the JAX original before
they get used for replay-corpus capture.

## Config-name discovery

The local checkout under `~/Codes/openpi` did not have a `pi05_droid_jointpos`
entry in `src/openpi/training/config.py` (it predates that config). The
conversion instead ran against **cml30's `xuningy` fork**
(`/tmp2/chungyili/openpi`), which defines it:

```python
TrainConfig(
    name="pi05_droid_jointpos",
    model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
    data=SimpleDataConfig(
        assets=AssetsConfig(asset_id="droid"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
            outputs=[_transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)), droid_policy.DroidOutputs()],
        ),
        base_config=DataConfig(prompt_from_task=True),
    ),
),
```

`Pi0Config` defaults printed at conversion time: `action_dim=32,
action_horizon=15, max_token_len=200, paligemma_variant='gemma_2b',
pi05=True`. `DroidOutputs` slices the model's 32-dim padded action to the
real 8-dim droid action (7 joints + gripper), so that is the dimensionality
that actually reaches a policy caller.

Before conversion could run at all, the fork's `transformers_replace` patch
had to be installed (`transformers==4.53.2`) and verified (`True`) — the
stock `transformers` release does not carry the PaliGemma changes the
PyTorch port depends on.

## Why this ran remotely, not locally

Local conversion attempts on this machine (30 GB RAM, an active Isaac Sim
session competing for memory) were killed OOM three times, observed RSS in
the 21-24 GB range for the JAX params + PyTorch conversion buffers held
simultaneously — too close to the 30 GB ceiling once the OS and other
processes are accounted for. The conversion was moved to cml30, which has no
such constraint, and the model work stayed there for the same reason
(conversion, loading, and inference all reserved for the remote GPU box; no
model was loaded on the local machine at any point in this task).

## Conversion command

Run on cml30 (`ssh cml30.csie.ntu.edu.tw`), `cd /tmp2/chungyili/openpi`,
with `PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/tmp2/chungyili/.cache/uv
OPENPI_DATA_HOME=/tmp2/chungyili/.cache/openpi`:

```bash
uv run python examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /tmp2/chungyili/.cache/openpi/openpi-assets-simeval/pi05_droid_jointpos \
    --config_name pi05_droid_jointpos \
    --output_path /tmp2/chungyili/pytorch-ckpt/pi05_droid_jointpos \
    --precision bfloat16
```

Log: `/tmp2/chungyili/convert.log` (cml30). Output: bf16 `model.safetensors`
+ `config.json`, 6.8 GB, at `/tmp2/chungyili/pytorch-ckpt/pi05_droid_jointpos`.

The converted directory does not carry an `assets/` subfolder on its own
(only `model.safetensors` + `config.json`), but `create_trained_policy` reads
normalization stats from `checkpoint_dir / "assets" / <asset_id>` regardless
of backend. Fixed by copying the JAX checkpoint's `assets/` folder
(`droid/norm_stats.json` + `physical-intelligence/droid.lock`) alongside the
converted weights on both cml30 and the local mirror, so the same norm stats
that were used at training time apply to both backends.

## Parity gate

### Method

`openpi.policies.policy.Policy.infer(obs, noise=...)` and both backends'
`sample_actions` (`src/openpi/models/pi0.py` JAX, and
`src/openpi/models_pytorch/pi0_pytorch.py` PyTorch) accept an explicit
`noise` array that overrides the internal `sample_noise` draw and is used as
the flow-matching ODE's starting point `x_T`. With `noise` fixed, denoising
is deterministic given `(obs, noise)`, so the two backends can be compared
**directly** rather than only via mean/std across independent samples — the
stronger of the two gates described in the task brief.

Ran one process per backend on cml30 (GPU 3, the freest of six, picked via
`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
sort -t, -k2 -rn`), script `/tmp2/chungyili/parity_check.py`:

1. Build one fixed synthetic observation with `np.random.default_rng(0)`:
   `exterior_image_1_left` and `wrist_image_left` each `(224, 224, 3)` uint8
   random, `joint_position = rng.normal(0, 0.3, 7)` float32,
   `gripper_position = zeros(1)`, prompt `"put the orange juice carton in
   the grey bin"`.
2. Build 8 fixed noise tensors with `np.random.default_rng(1)`, shape
   `(action_horizon=15, action_dim=32)` each — identical for both backends.
3. JAX pass: `create_trained_policy(get_config("pi05_droid_jointpos"),
   ".../openpi-assets-simeval/pi05_droid_jointpos")`, call
   `policy.infer(obs, noise=noise_i)` for each of the 8 noise tensors, stack
   the resulting `(15, 8)` action chunks, save to `parity_jax.npz`.
4. PyTorch pass (separate process): same config, checkpoint dir
   `/tmp2/chungyili/pytorch-ckpt/pi05_droid_jointpos` (auto-detected as
   PyTorch via the presence of `model.safetensors`), same obs, same 8 noise
   tensors, save to `parity_torch.npz`.
5. Load both `.npz` files in a third process and compare.

JAX params are float32; the converted PyTorch checkpoint is bf16
(`--precision bfloat16`), so some divergence from that alone is expected —
the thresholds below already allow for it.

### Numbers

8 chunks x 15 timesteps x 8 action dims (7 joints + gripper), both backends,
identical noise per chunk (`np.allclose(noise_jax, noise_torch)` = True):

| Metric | Value |
| --- | --- |
| Overall MAE (all chunks/timesteps/dims) | **0.00176 rad** |
| Max abs diff (any single element) | 0.0108 rad |
| Per-chunk MAE, min–max | 0.00107 – 0.00252 rad |
| Per-dim MAE, min–max | 0.00073 – 0.00264 rad |

Per-dim mean over the 8 chunks (JAX vs. PyTorch), first dims shown:

| dim | JAX mean | PyTorch mean | JAX std | PyTorch std |
| --- | --- | --- | --- | --- |
| 0 | -0.2070 | -0.2048 | 0.0594 | 0.0611 |
| 1 | -0.0993 | -0.1004 | 0.0564 | 0.0578 |
| 2 | -0.0128 | -0.0129 | 0.0772 | 0.0787 |
| 3 | 0.0421 | 0.0404 | 0.0990 | 0.1009 |
| 4 | 0.1793 | 0.1779 | 0.1359 | 0.1383 |
| 5 | 0.2096 | 0.2072 | 0.0641 | 0.0643 |
| 6 | -0.0789 | -0.0798 | 0.0916 | 0.0926 |
| 7 (gripper) | 0.0071 | 0.0065 | 0.0046 | 0.0047 |

### Verdict

**PASS.** Gate was direct-comparison MAE < 1e-2 rad; achieved 0.00176 rad
overall, roughly 6x under threshold, with the single largest per-element
outlier (0.0108 rad) still an order of magnitude below the "gross mismatch"
bar (> 0.1 rad mean) that would BLOCK. The converted PyTorch weights are
faithful to the JAX checkpoint within the tolerance expected from a
float32-to-bf16 precision change and are cleared for replay-corpus capture.

## Local capture environment fixes (Task 5)

Two one-time fixes were needed before the local machine's openpi venv
(`~/Codes/openpi/.venv`, openpi @ `215abfb`) could run the converted
checkpoint for activation capture (`analysis/mass_com/capture_pi05.py`).
Both mutate the venv only — no openpi source commits.

1. **`transformers_replace` was not applied locally** (the Task-4 precedent
   above applied it on cml30's fork checkout only). Symptom:
   `ImportError: cannot import name 'check' from 'transformers.models.siglip'`,
   and `PI0Pytorch.__init__` refuses to construct without the patch. The
   installed `transformers==4.53.2` matches the pin, so only the copy step
   was needed:

   ```bash
   cp -r ~/Codes/openpi/src/openpi/models_pytorch/transformers_replace/* \
       ~/Codes/openpi/.venv/lib/python3.11/site-packages/transformers/
   # verify:
   ~/Codes/openpi/.venv/bin/python -c "from transformers.models.siglip import check; \
       print(check.check_whether_transformers_replace_is_installed_correctly())"  # True
   ```

2. **torch was the cu126 build, which has no sm_120 kernels** — the local
   RTX 5090 (compute capability 12.0) failed every GPU op with
   `CUDA error: no kernel image is available for execution on the device`
   (supported archs of cu126 wheels stop at sm_90). Swapped to the cu128
   build of the *same* torch/torchvision versions:

   ```bash
   uv pip install --python ~/Codes/openpi/.venv/bin/python \
       --index-url https://download.pytorch.org/whl/cu128 \
       "torch==2.7.1+cu128" "torchvision==0.22.1+cu128"
   # verify:
   ~/Codes/openpi/.venv/bin/python -c "import torch; \
       x = torch.randn(64,64,device='cuda',dtype=torch.bfloat16); \
       print((x@x).float().abs().sum().item() > 0)"  # True
   ```

   Note: re-running `uv sync` inside `~/Codes/openpi` would revert both fixes
   (reinstalling cu126 torch and a clean transformers); reapply them after any
   sync.

## Checkpoint locations

- JAX (source, unmodified): cml30
  `/tmp2/chungyili/.cache/openpi/openpi-assets-simeval/pi05_droid_jointpos`
- PyTorch (converted, bf16): cml30
  `/tmp2/chungyili/pytorch-ckpt/pi05_droid_jointpos`
- PyTorch (converted, bf16), synced to local machine:
  `~/.cache/openpi/pytorch/pi05_droid_jointpos/` (6.8 GB; includes the copied
  `assets/` folder for norm stats)

Parity artifacts (cml30, kept for reproducibility): `parity_check.py`,
`parity_jax.npz`, `parity_torch.npz`, all under `/tmp2/chungyili/`.
