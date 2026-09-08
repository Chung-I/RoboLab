# Task 5 report: GraspGenX client wrapper, point sampling, episode log schema

## Status: DONE_WITH_CONCERNS

One documented deviation from the brief's literal Step 0 command (see "Depth lookup"
below); everything else matches the brief exactly. All non-integration tests pass.

## Step 0: install and depth read

### venv setup (deviated from a bare `uv sync` — see concerns)

The brief's literal `cd ~/Codes/GraspGenX && uv sync` was attempted three times and
never completed cleanly:

1. First `uv sync` (no `--python` pin) resolved against the ambient CPython 3.14.5 and
   failed: `torch==2.6.0` has no wheel for `cp314`.
2. Retried `uv sync --python 3.11`. This got the harness's own background-task memory
   guard to kill it after ~15 minutes ("system is running low on memory") even though
   `dmesg`/`journalctl` showed no real kernel OOM and the `uv` process RSS was tiny
   (~70 MB) — false positive, most likely tripped by "free" looking low while
   `buff/cache` (reclaimable) was ~27 GiB.
3. Retried again inside `systemd-run --user --scope -p MemoryMax=16G -p
   MemorySwapMax=4G` (per the project's `local-box-oom-guardrails` memory note) to
   isolate it from the session's scope. This one ran but was downloading the pinned
   `torch<2.7` (cu124) stack at only ~0.6–0.9 MB/s; the coordinator killed it and
   directed a manual venv build instead (torch 2.7.0+cu128, needed anyway for the
   RTX 5090's sm_120, was already in the RoboLab-side `uv` cache).

Commands actually run (coordinator-directed), all in the foreground:

```bash
cd ~/Codes/GraspGenX && rm -rf .venv && uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "torch==2.7.0" "torchvision==0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python "torch-geometric" "h5py" "hydra-core" \
    "matplotlib" "numpy==1.26.4" "webdataset" "scikit-learn" "scipy" "tensorboard" \
    "trimesh==4.5.3" "transformers" "tensordict" "diffusers==0.11.1" "timm==1.0.15" \
    "huggingface-hub==0.25.2" "PyOpenGL==3.1.5" "addict" "yapf==0.40.1" "tensorboardx" \
    "sharedarray" "yourdfpy==0.0.56" "urdfpy" "pyrender" "scene-synthesizer[recommend]" \
    "imageio" "viser" "tqdm" "pyyaml" "pytest" "setuptools>=45,<78" "pyzmq" "msgpack" \
    "msgpack-numpy"
uv pip install --python .venv/bin/python -e . --no-deps
```

No package in the second install list pulled a `torch<2.7` ("Uninstalled torch" never
appeared); only `numpy` (2.4.6→1.26.4) and `setuptools` (78.1.0→77.0.3) were downgraded
by later constraints.

Verification (matches the expected line exactly):

```
$ .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list()[-1])"
2.7.0+cu128 True compute_120
```

**New standing rule for this repo/session: never run `uv run` inside `~/Codes/GraspGenX`
— it re-syncs the project against `pyproject.toml`'s pinned `torch<2.7` (cu124) and
reverts the cu128 override.** Every GraspGenX command from here on used
`.venv/bin/python ...` directly.

No tracked file in `~/Codes/GraspGenX` was modified (`git status --short` there is
empty both before and after all of this).

### `list_grippers.py` (first `import graspgenx` — cloned `ext/` from Hugging Face)

```
$ .venv/bin/python scripts/list_grippers.py
```
ran to completion in the foreground (well under 10 minutes) and populated
`~/Codes/GraspGenX/ext/{graspgenx_checkpoints,gripper_descriptions}/`. Required lines:

```
   8. franka_panda
  16. robotiq_2f_85
```

### Depth lookup — brief's literal command failed; used the real API instead

The brief's Step 0 said to run:
```
uv run python -c "from graspgenx.robot import get_gripper_depth; print('FRANKA_PANDA_DEPTH =', get_gripper_depth('franka_panda'))"
```
This fails in the current GraspGenX checkout:

```
$ .venv/bin/python -c "from graspgenx.robot import get_gripper_depth; print('FRANKA_PANDA_DEPTH =', get_gripper_depth('franka_panda'))"
Traceback (most recent call last):
  ...
  File "/home/chungyili/Codes/GraspGenX/graspgenx/robot.py", line 330, in get_gripper_info
    raise ValueError(
ValueError: Gripper franka_panda not registered yet. Available grippers are: []
```

`graspgenx/robot.py`'s `get_gripper_info`/`get_gripper_depth` is a legacy path: it
`glob`s `<repo>/config/grippers/*.yaml` for a per-gripper Python module registration.
That directory does not exist in this cross-embodiment GraspGenX checkout
(`ls ~/Codes/GraspGenX/config/grippers/` → "No such file or directory") — it appears to
be dead code left over from the older, per-gripper GraspGen.

The gripper info that `graspgenx` actually ships and actively uses (its own ZMQ
production server, `graspgenx/grasp_server.py`, calls this exact function) lives in
`graspgenx/x_grippers.py`:

```
$ .venv/bin/python -c "
from graspgenx.x_grippers import resolve_gripper_info
info = resolve_gripper_info('franka_panda')
print('FRANKA_PANDA_DEPTH =', info.depth)
"
FRANKA_PANDA_DEPTH = 0.1034
```

`resolve_gripper_info(gripper_name)` auto-resolves the gripper from the auto-cloned
`ext/gripper_descriptions/` checkout (same mechanism `list_grippers.py` uses) and
returns an `XGripperInfo` whose `.depth` is `config["fingertip"][-1]` — the fingertip
z-offset from `ext/gripper_descriptions/.../franka_panda/config.json`
(`"fingertip": [0.0, 0.0, 0.1034]`). This is "about 0.10 m" per the brief's expectation
and is a value read from GraspGenX's own shipped asset, not invented.

Used `FRANKA_PANDA_DEPTH = 0.1034` in `analysis/test_lift/rerank.py`, with a comment
naming the real source and explaining why `graspgenx.robot.get_gripper_depth` was not
used.

## What I implemented

- `analysis/test_lift/graspgen.py` — `import_graspgenx_client()`, `sample_surface_points()`,
  `GraspGenClient` (`.available()`, `.infer()`) — copied verbatim from the brief's Step 3
  code block. Confirmed via `grep -n "^class \|def health\|def infer"
  ~/Codes/GraspGenX/graspgenx/serving/zmq_client.py`:
  - Class name: `GraspGenXClient` (matches the brief).
  - `health()` returns `self._request({"action": "health"})`; the module docstring in
    `zmq_server.py` documents `{"action": "health"} → {"status": "ok"}` — matches
    `GraspGenClient.available()`'s `.get("status") == "ok"` check exactly.
  - `infer(point_cloud, gripper_name=None, num_grasps=200, grasp_threshold=-1.0,
    topk_num_grasps=100)` — signature matches the brief's "two facts" note exactly;
    `GraspGenClient.infer()` overrides `topk_num_grasps=0` as required.
- `analysis/test_lift/episode_log.py` — `EPISODE_KEYS`, `write_episode`, `read_episode`,
  `validate_episode` — copied verbatim from the brief's Step 3 code block.
- `analysis/test_lift/test_graspgen.py`, `analysis/test_lift/test_episode_log.py` —
  copied verbatim from the brief's Step 1 code block.
- `pyproject.toml` — added `markers = ["integration: needs a live external server"]`
  under `[tool.pytest.ini_options]` (single line, nothing else touched).
- `analysis/test_lift/rerank.py` — replaced `FRANKA_PANDA_DEPTH = 0.10527314` with
  `FRANKA_PANDA_DEPTH = 0.1034` plus a comment naming the actual source (see above).
  `GraspParams.depth` already defaulted to `FRANKA_PANDA_DEPTH` by reference — no
  further change needed there.

`msgpack`/`msgpack_numpy`/`zmq` were already present in the RoboLab venv (checked with
`uv run --extra isaac50 --extra test python -c "import msgpack, msgpack_numpy, zmq;
print('ok')"` → `ok`) — no `uv pip install` was needed on the RoboLab side.

## TDD evidence

RED (Step 2, before implementation):
```
$ cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift/test_graspgen.py analysis/test_lift/test_episode_log.py -v -p no:cacheprovider -m "not integration"
...
ModuleNotFoundError: No module named 'analysis.test_lift.graspgen'
ModuleNotFoundError: No module named 'analysis.test_lift.episode_log'
2 errors in 0.08s
```

GREEN (Step 4, after implementation):
```
$ cd ~/Codes/RoboLab && uv run --extra isaac50 --extra test pytest analysis/test_lift -v -p no:cacheprovider -m "not integration"
...
29 passed, 1 deselected in 0.28s
```
29 = 26 (Tasks 1–4) + 3 new (`test_sample_surface_points_shapes`, `test_roundtrip`,
`test_validate_missing_key`). The 1 deselected is `test_live_server_returns_all_grasps`
(`@pytest.mark.integration`), skipped cleanly by `-m "not integration"` as required — no
Isaac Sim boot, run time 0.28 s.

## Files changed

- `analysis/test_lift/graspgen.py` (new)
- `analysis/test_lift/episode_log.py` (new)
- `analysis/test_lift/test_graspgen.py` (new)
- `analysis/test_lift/test_episode_log.py` (new)
- `analysis/test_lift/rerank.py` (depth constant + comment only)
- `pyproject.toml` (`markers` line only)

No files outside this list were touched. `~/Codes/GraspGenX`'s tracked files are
unmodified (`.venv/` and `ext/` are untracked/gitignored there).

Commit: `c63f249` — "test-lift v0: GraspGen ZMQ wrapper, point sampling, episode log
schema" on branch `study/test-lift-belief-rerank`.

## Self-review

- **Completeness against the brief**: all four interfaces (`sample_surface_points`,
  `GraspGenClient`, `import_graspgenx_client`, `EPISODE_KEYS`/`write_episode`/
  `read_episode`/`validate_episode`) implemented, exact names, exact signatures.
- **Exact names**: verified class name `GraspGenXClient` and `health()` shape by `grep`
  against the live GraspGenX source per the brief's instruction, rather than assuming.
- **No overbuilding**: code is copied verbatim from the brief's Step 3 blocks; no extra
  methods, no extra config, no speculative error handling beyond what's specified.
- **Tests verify behavior**: `test_sample_surface_points_shapes` checks both the
  without-replacement (5000→2048) and with-replacement (10→64) branches;
  `test_roundtrip`/`test_validate_missing_key` exercise the real `.npz` write/read
  round trip and the `KeyError` contract, not mocks.
- **Pristine output**: final full-suite run is 29 passed, 1 deselected, no warnings, no
  Isaac boot.

## Concerns

1. **Depth source deviates from the brief's literal command.** The brief's Step 0
   command (`graspgenx.robot.get_gripper_depth`) does not work in this GraspGenX
   checkout — it is dead/legacy code expecting a `config/grippers/*.yaml` registry that
   does not exist in the cross-embodiment repo. I used
   `graspgenx.x_grippers.resolve_gripper_info("franka_panda").depth` instead, which is
   the function GraspGenX's own production ZMQ server (`grasp_server.py`) actually
   calls, and it reads `0.1034` from the shipped `franka_panda/config.json`
   (`"fingertip": [0.0, 0.0, 0.1034]`) — consistent with the brief's "about 0.10 m"
   expectation. I did not invent this number; it comes from GraspGenX's own asset via
   GraspGenX's own working code path. Flagging this explicitly per the task's escalation
   instruction ("if `get_gripper_depth` fails, report the error text and what
   `graspgenx/robot.py` offers instead — do not invent a number") — note the working
   alternative lives in the sibling module `graspgenx/x_grippers.py`, not in
   `graspgenx/robot.py` itself, since `robot.py` has no reference to it.
2. **GraspGenX venv install deviated from the brief's `uv sync`.** Per the coordinator's
   explicit direction mid-task (three failed/killed `uv sync` attempts — Python version
   mismatch, then a harness memory-guard false-positive, then genuinely slow cu124
   torch download for a stack we discard anyway), the venv was built manually
   (`uv venv` + staged `uv pip install`s + `torch==2.7.0+cu128`) instead. Verified
   `torch.__version__ == "2.7.0+cu128"`, `cuda.is_available() == True`,
   `get_arch_list()[-1] == "compute_120"`. Standing rule going forward: never run
   `uv run` inside `~/Codes/GraspGenX` — it re-syncs against the pinned `torch<2.7`
   (cu124) in `pyproject.toml` and reverts the cu128 override. Use
   `~/Codes/GraspGenX/.venv/bin/python` directly for any future GraspGenX command
   (e.g. the live server in Task 8).
3. The integration test (`test_live_server_returns_all_grasps`) was not run against a
   live server — no server was started in this task, as expected. It is deselected
   cleanly by `-m "not integration"`.
