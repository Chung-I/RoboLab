# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pi0.5 activation capture over the replay corpus (Plan-2 Task 5).

Runs with the openpi venv (~/Codes/openpi/.venv), NOT the RoboLab venv:

    ~/Codes/openpi/.venv/bin/python analysis/mass_com/capture_pi05.py \
        --corpus output/replay_corpus --out output/activations/pi05

For every corpus step it rebuilds the exact observation the live
Pi0DroidJointposClient sent (policies/pi0_family/client.py: resize_with_pad
224x224, joint_position 7, gripper_position 1, prompt from env_cfg.json),
runs one full chunk inference with fixed seed + fixed noise, and captures the
post-block residual stream of all 18 PaliGemma decoder layers (prefix pass)
plus all 18 action-expert (gemma_300m) decoder layers (first denoise step) at
three token positions:

    P0 last_prefix_token   -- last valid language token (prefix stream, D=2048)
    P1 image_tokens_mean   -- mean over the 512 valid image tokens (prefix stream)
    P2 first_suffix_token  -- first action token, expert stream at denoise step
                              t=1.0 (D=1024, zero-padded to 2048)

Token layout of the prefix (pi0_pytorch.embed_prefix + DroidInputs, PI05):
    [0,256)    img_cam1  base_0_rgb        (exterior_image_1_left)
    [256,512)  img_cam2  left_wrist_0_rgb  (wrist_image_left)
    [512,768)  img_pad   right_wrist_0_rgb (all-zero image, mask=False)
    [768,968)  language  "Task: {prompt}, State: {256-bin ints};\\nAction: "
               padded to max_token_len=200; valid length varies per step
               because the discretized state digits vary.

Outputs per condition: <out>/<object>/<condition>/acts.npz with
    acts          (T, 18, 3, 2048) float16
    actions_out   (T, 15, 8) float32   -- full inferred chunk per step
    n_lang_valid  (T,) int32
    lang_text_ranges / lang_state_ranges / lang_tail_ranges  (T, 2) int32
        absolute prefix token indices per step
plus a single <out>/meta.json (layer names, position + token-block
definitions, checkpoint, seed, noise, git SHA, per-condition stats).
"""

import argparse
import json
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from openpi_client import image_tools

# ---------------------------------------------------------------------------
# Constants (verified against openpi @ 215abfb, see module docstring)
# ---------------------------------------------------------------------------

NUM_IMG_TOKENS = 256  # SigLIP 224/14 = 16 -> 16*16 patches per camera
IMG_BLOCKS = {
    "img_cam1": (0, 256),
    "img_cam2": (256, 512),
    "img_pad_right_wrist": (512, 768),
}
LANG_OFFSET = 768
MAX_LANG_TOKENS = 200  # Pi0Config.max_token_len for pi05
PREFIX_LEN = LANG_OFFSET + MAX_LANG_TOKENS  # 968
ACTION_HORIZON = 15
ACTION_DIM_PADDED = 32  # model action dim (DroidOutputs slices to 8)
ACTION_DIM_OUT = 8
NUM_LAYERS = 18  # gemma_2b prefix depth == gemma_300m expert depth
PREFIX_D = 2048  # gemma_2b width
EXPERT_D = 1024  # gemma_300m width
NUM_DENOISE_STEPS = 10  # sample_actions default

SEED = 0
DEFAULT_CHECKPOINT = "~/.cache/openpi/pytorch/pi05_droid_jointpos"
DEFAULT_CONFIG_NAME = "pi05_droid_jointpos_polaris"

CONDITIONS = [
    "MassLight_CoMCenter",
    "MassMedium_CoMCenter",
    "MassHeavy_CoMCenter",
    "MassMedium_CoMUp",
    "MassMedium_CoMDown",
]
OBJECTS = ["orange_juice_carton", "soft_scrub"]


def fixed_noise() -> np.ndarray:
    """The fixed flow-matching starting noise used for every capture step."""
    return np.random.default_rng(0).standard_normal(
        (ACTION_HORIZON, ACTION_DIM_PADDED)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in test_capture_pi05.py)
# ---------------------------------------------------------------------------


def build_step_request(cam1_img, wrist_img, joint_pos, gripper_pos, prompt):
    """Build the exact request dict Pi0DroidJointposClient._pack_request sends.

    Images are HxWx3 uint8 (any resolution; resized with padding to 224x224),
    joint_pos is (7,), gripper_pos is (1,).
    """
    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            np.asarray(cam1_img), 224, 224),
        "observation/wrist_image_left": image_tools.resize_with_pad(
            np.asarray(wrist_img), 224, 224),
        "observation/joint_position": np.asarray(joint_pos),
        "observation/gripper_position": np.asarray(gripper_pos),
        "prompt": prompt,
    }


def _is_digit_piece(piece: str) -> bool:
    stripped = piece.replace("▁", "")  # sentencepiece "▁" word marker
    return stripped.isdigit()


def lang_block_ranges(pieces: list[str]) -> dict[str, tuple[int, int]]:
    """Split the valid language tokens of a pi05 prompt into blocks.

    The tokenized string is "Task: {text}, State: {ints};\\nAction: ".
    Returns ranges RELATIVE to the language block start:
      text  = [0, state_start)   (includes "Task:", the instruction, ", State:")
      state = [state_start, state_end)  (the discretized state integers)
      tail  = [state_end, len)   (";\\nAction: ")
    """
    marker = None
    for i, p in enumerate(pieces):
        if "State" in p:
            marker = i
            break
    if marker is None:
        raise ValueError(f"no 'State' marker piece found in {pieces!r}")
    state_start = None
    for i in range(marker + 1, len(pieces)):
        if _is_digit_piece(pieces[i]):
            state_start = i
            break
    if state_start is None:
        raise ValueError(f"no digit pieces after 'State' marker in {pieces!r}")
    state_end = state_start
    while state_end < len(pieces) and _is_digit_piece(pieces[state_end]):
        state_end += 1
    return {
        "text": (0, state_start),
        "state": (state_start, state_end),
        "tail": (state_end, len(pieces)),
    }


F16_MAX = float(np.finfo(np.float16).max)  # 65504.0


def to_f16_clipped(vec: np.ndarray) -> tuple[np.ndarray, int]:
    """Cast to float16, clipping to the finite f16 range.

    Gemma's residual stream has rare massive activations (observed: one
    element of layer 17 at last_prefix_token exceeds 65504 in bf16); clipping
    keeps the saved acts finite. Returns (f16 array, number of clipped elems).
    """
    vec = np.asarray(vec, dtype=np.float32)
    n_clipped = int((np.abs(vec) > F16_MAX).sum())
    return np.clip(vec, -F16_MAX, F16_MAX).astype(np.float16), n_clipped


def capture_positions(n_lang_valid: int) -> dict:
    """Index definitions for the three capture positions."""
    if not 1 <= n_lang_valid <= MAX_LANG_TOKENS:
        raise ValueError(f"n_lang_valid={n_lang_valid} out of range [1, {MAX_LANG_TOKENS}]")
    return {
        "last_prefix_token": LANG_OFFSET + n_lang_valid - 1,
        "image_token_slice": (0, 2 * NUM_IMG_TOKENS),  # valid cams only
        "first_suffix_token": 0,  # expert stream, action token 0
    }


class LayerTap:
    """Forward-hook recorder for a set of named layers.

    Records every forward call of each registered module as a detached CPU
    tensor (output[0] if the module returns a tuple, like GemmaDecoderLayer).
    """

    def __init__(self):
        self.records: "OrderedDict[str, list[torch.Tensor]]" = OrderedDict()
        self._handles = []

    def register(self, named_modules: dict) -> None:
        for name, module in named_modules.items():
            self.records[name] = []
            self._handles.append(
                module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            self.records[name].append(out.detach().cpu())
        return hook

    def stacked(self, call_index: int = 0) -> np.ndarray:
        """Stack call `call_index` of every layer -> (L, *record_shape) f32."""
        tensors = [recs[call_index].float() for recs in self.records.values()]
        return torch.stack(tensors).numpy()

    def clear(self) -> None:
        for recs in self.records.values():
            recs.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def extract_step_acts(prefix_records, expert_records, n_lang_valid: int) -> tuple[np.ndarray, int]:
    """((L, P, PREFIX_D) float16, n_clipped) from one inference's tap records.

    prefix_records: per-layer lists with >=1 capture of (1, PREFIX_LEN, PREFIX_D).
    expert_records: per-layer lists with NUM_DENOISE_STEPS captures of
                    (1, ACTION_HORIZON, EXPERT_D); capture [0] (t=1.0) is used.
    """
    pos = capture_positions(n_lang_valid)
    img_lo, img_hi = pos["image_token_slice"]
    acts = np.zeros((NUM_LAYERS, 3, PREFIX_D), dtype=np.float16)
    n_clipped = 0
    for li, recs in enumerate(prefix_records.values()):
        rec = recs[0][0].float()  # (PREFIX_LEN, PREFIX_D)
        acts[li, 0], c0 = to_f16_clipped(rec[pos["last_prefix_token"]].numpy())
        acts[li, 1], c1 = to_f16_clipped(rec[img_lo:img_hi].mean(dim=0).numpy())
        n_clipped += c0 + c1
    for li, recs in enumerate(expert_records.values()):
        rec = recs[0][0].float()  # (ACTION_HORIZON, EXPERT_D)
        acts[li, 2, :EXPERT_D], c2 = to_f16_clipped(rec[pos["first_suffix_token"]].numpy())
        n_clipped += c2
    return acts, n_clipped


# ---------------------------------------------------------------------------
# Model-side plumbing
# ---------------------------------------------------------------------------


def load_policy(checkpoint: str, config_name: str, device: str):
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(config_name)
    ckpt = Path(checkpoint).expanduser()
    policy = _policy_config.create_trained_policy(cfg, ckpt, pytorch_device=device)
    assert policy._is_pytorch_model, "expected the converted PyTorch checkpoint"
    return policy


def attach_taps(policy) -> tuple[LayerTap, LayerTap, list[str], list[str]]:
    model = policy._model  # PI0Pytorch
    lm_layers = model.paligemma_with_expert.paligemma.language_model.layers
    ex_layers = model.paligemma_with_expert.gemma_expert.model.layers
    assert len(lm_layers) == NUM_LAYERS and len(ex_layers) == NUM_LAYERS
    lm_names = [f"paligemma.language_model.layers.{i}" for i in range(NUM_LAYERS)]
    ex_names = [f"gemma_expert.model.layers.{i}" for i in range(NUM_LAYERS)]
    tap_lm, tap_ex = LayerTap(), LayerTap()
    tap_lm.register(dict(zip(lm_names, lm_layers)))
    tap_ex.register(dict(zip(ex_names, ex_layers)))
    return tap_lm, tap_ex, lm_names, ex_names


def make_tokenizer_meta_fn(policy):
    """Return fn(request) -> (n_lang_valid, blocks) using the policy's own
    input-transform pipeline (bit-identical to what infer() tokenizes)."""
    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor(
        model_file=str(Path("~/.cache/openpi/big_vision/paligemma_tokenizer.model").expanduser()))

    def fn(request: dict):
        transformed = policy._input_transform({k: v for k, v in request.items()})
        tokens = np.asarray(transformed["tokenized_prompt"])
        mask = np.asarray(transformed["tokenized_prompt_mask"])
        n_valid = int(mask.sum())
        pieces = [sp.id_to_piece(int(t)) for t in tokens[:n_valid]]
        blocks = lang_block_ranges(pieces)
        return n_valid, blocks

    return fn


def capture_step(policy, tap_lm, tap_ex, request, noise, tokenize_meta):
    """One corpus step: seeded chunk inference + activation harvest."""
    n_valid, blocks = tokenize_meta(request)
    tap_lm.clear()
    tap_ex.clear()
    torch.manual_seed(SEED)
    result = policy.infer(request, noise=noise.copy())
    actions = np.asarray(result["actions"], dtype=np.float32)
    assert actions.shape == (ACTION_HORIZON, ACTION_DIM_OUT), actions.shape
    first_lm = next(iter(tap_lm.records.values()))
    first_ex = next(iter(tap_ex.records.values()))
    assert len(first_lm) == 1, f"prefix pass fired {len(first_lm)} times"
    assert len(first_ex) == NUM_DENOISE_STEPS, f"denoise fired {len(first_ex)} times"
    assert first_lm[0].shape == (1, PREFIX_LEN, PREFIX_D), first_lm[0].shape
    assert first_ex[0].shape == (1, ACTION_HORIZON, EXPERT_D), first_ex[0].shape
    acts, n_clipped = extract_step_acts(tap_lm.records, tap_ex.records, n_valid)
    return acts, actions, n_valid, blocks, n_clipped


# ---------------------------------------------------------------------------
# Corpus iteration
# ---------------------------------------------------------------------------


def iter_condition_steps(cond_dir: Path):
    """Yield (t, request) for every step of a corpus condition."""
    import h5py

    env_cfg = json.loads((cond_dir / "env_cfg.json").read_text())
    prompt = env_cfg["instruction"]
    with h5py.File(cond_dir / "replay.hdf5", "r") as f:
        demo = f["data/demo_0"]
        cam1 = demo["obs/image_obs/over_shoulder_left_camera"]
        wrist = demo["obs/image_obs/wrist_cam"]
        joints = demo["obs/proprio_obs/arm_joint_pos"][:]
        grippers = demo["obs/proprio_obs/gripper_pos"][:]
        T = joints.shape[0]
        for t in range(T):
            yield t, T, build_step_request(cam1[t], wrist[t], joints[t], grippers[t], prompt)


def capture_condition(policy, tap_lm, tap_ex, tokenize_meta, cond_dir: Path,
                      out_dir: Path, noise: np.ndarray) -> dict:
    t0 = time.monotonic()
    acts_all = actions_all = None
    n_valid_all = []
    total_clipped = 0
    ranges = {"text": [], "state": [], "tail": []}
    T = None
    for t, T, request in iter_condition_steps(cond_dir):
        acts, actions, n_valid, blocks, n_clipped = capture_step(
            policy, tap_lm, tap_ex, request, noise, tokenize_meta)
        total_clipped += n_clipped
        if acts_all is None:
            acts_all = np.zeros((T, *acts.shape), dtype=np.float16)
            actions_all = np.zeros((T, *actions.shape), dtype=np.float32)
        acts_all[t] = acts
        actions_all[t] = actions
        n_valid_all.append(n_valid)
        for k in ranges:
            lo, hi = blocks[k]
            ranges[k].append((LANG_OFFSET + lo, LANG_OFFSET + hi))
        if t % 50 == 0:
            print(f"  step {t}/{T}", flush=True)
    assert np.isfinite(acts_all).all(), "non-finite activations"
    assert np.isfinite(actions_all).all(), "non-finite actions"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "acts.npz",
        acts=acts_all,
        actions_out=actions_all,
        n_lang_valid=np.asarray(n_valid_all, dtype=np.int32),
        lang_text_ranges=np.asarray(ranges["text"], dtype=np.int32),
        lang_state_ranges=np.asarray(ranges["state"], dtype=np.int32),
        lang_tail_ranges=np.asarray(ranges["tail"], dtype=np.int32),
    )
    wall = time.monotonic() - t0
    n_valid_arr = np.asarray(n_valid_all)
    prompt = json.loads((cond_dir / "env_cfg.json").read_text())["instruction"]
    return {
        "T": int(T),
        "prompt": prompt,
        "wall_time_s": round(wall, 1),
        "n_lang_valid_min": int(n_valid_arr.min()),
        "n_lang_valid_max": int(n_valid_arr.max()),
        "n_f16_clipped_elems": int(total_clipped),
        "lang_blocks_step0": {k: [int(v) for v in ranges[k][0]] for k in ranges},
    }


# ---------------------------------------------------------------------------
# Determinism gate
# ---------------------------------------------------------------------------


def run_determinism_gate(policy, tap_lm, tap_ex, tokenize_meta, corpus: Path,
                         noise: np.ndarray, out_path: Path | None) -> dict:
    """Capture two test steps of carton/MassMedium_CoMCenter twice each and
    require bit-identical actions and activations."""
    cond_dir = corpus / "orange_juice_carton" / "MassMedium_CoMCenter"
    anchor = int(np.load(cond_dir / "ft.npz")["anchor_step"])
    test_steps = [0, anchor]
    steps = {}
    for t, T, request in iter_condition_steps(cond_dir):
        if t in test_steps:
            steps[t] = request
        if len(steps) == len(test_steps):
            break

    verdict = {"steps": test_steps, "bit_identical": True, "max_abs_diff_acts": 0.0,
               "max_abs_diff_actions": 0.0, "result": "PASS"}
    gate_arrays = {}
    for t, request in steps.items():
        a1, act1, *_ = capture_step(policy, tap_lm, tap_ex, request, noise, tokenize_meta)
        a2, act2, *_ = capture_step(policy, tap_lm, tap_ex, request, noise, tokenize_meta)
        gate_arrays[f"acts_{t}"] = a1
        gate_arrays[f"actions_{t}"] = act1
        if not (np.array_equal(a1, a2) and np.array_equal(act1, act2)):
            verdict["bit_identical"] = False
            verdict["max_abs_diff_acts"] = max(
                verdict["max_abs_diff_acts"],
                float(np.abs(a1.astype(np.float64) - a2.astype(np.float64)).max()))
            verdict["max_abs_diff_actions"] = max(
                verdict["max_abs_diff_actions"],
                float(np.abs(act1 - act2).max()))
    if not verdict["bit_identical"]:
        tol_ok = (verdict["max_abs_diff_acts"] < 1e-6
                  and verdict["max_abs_diff_actions"] < 1e-6)
        verdict["result"] = "PASS_WITH_TOLERANCE" if tol_ok else "BLOCKED"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **gate_arrays)
    print(f"[determinism gate] {verdict}", flush=True)
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def git_sha() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def build_meta(args, lm_names, ex_names, per_condition, gate_verdict) -> dict:
    return {
        "checkpoint_path": str(Path(args.checkpoint).expanduser()),
        "config_name": args.config_name,
        "capture_git_sha": git_sha(),
        "seed": SEED,
        "noise": {"rng": "np.random.default_rng(0).standard_normal", "shape": [ACTION_HORIZON, ACTION_DIM_PADDED], "dtype": "float32"},
        "num_denoise_steps": NUM_DENOISE_STEPS,
        "layers": {
            "prefix": lm_names,
            "expert": ex_names,
            "note": "acts[:, l, p, :]: p in {0,1} from prefix layer l (D=2048); p=2 from expert layer l (D=1024, zero-padded to 2048). Post-block residual stream (GemmaDecoderLayer output).",
            "prefix_hidden_dim": PREFIX_D,
            "expert_hidden_dim": EXPERT_D,
        },
        "positions": [
            {"index": 0, "name": "last_prefix_token", "definition": "prefix stream, token LANG_OFFSET + n_lang_valid - 1 (last valid language token of 'Task: ..., State: ...;\\nAction: ')"},
            {"index": 1, "name": "image_tokens_mean", "definition": "prefix stream, mean over tokens [0,512) (the two valid cameras; padded right-wrist cam [512,768) excluded)"},
            {"index": 2, "name": "first_suffix_token", "definition": "expert stream, action token 0, captured at the first denoise step (t=1.0, x_t = fixed noise)"},
        ],
        "token_blocks": {
            "img_cam1": list(IMG_BLOCKS["img_cam1"]),
            "img_cam2": list(IMG_BLOCKS["img_cam2"]),
            "img_pad_right_wrist": list(IMG_BLOCKS["img_pad_right_wrist"]),
            "lang_slots": [LANG_OFFSET, PREFIX_LEN],
            "suffix": [0, ACTION_HORIZON],
            "note": "img/lang ranges index the prefix token sequence (len 968); suffix indexes the expert stream (len 15). text/state/tail sub-blocks of the language region vary per step (state digit count changes); per-step absolute ranges are stored in each acts.npz (lang_text_ranges, lang_state_ranges, lang_tail_ranges); step-0 values per condition are under per_condition.lang_blocks_step0.",
        },
        "f16_clip": {
            "clip_value": F16_MAX,
            "note": "activations are clipped to the finite float16 range before saving; per-condition clipped-element counts are in per_condition.n_f16_clipped_elems (observed: ~1 elem/step, layer 17 at last_prefix_token, a Gemma massive-activation dim slightly above 65504 in bf16)",
        },
        "determinism_gate": gate_verdict,
        "per_condition": per_condition,
        "output_contract": {
            "acts": "(T, 18, 3, 2048) float16",
            "actions_out": "(T, 15, 8) float32",
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="output/replay_corpus")
    ap.add_argument("--out", default="output/activations/pi05")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="subset like orange_juice_carton/MassMedium_CoMCenter")
    ap.add_argument("--test-determinism", action="store_true",
                    help="run only the determinism gate")
    ap.add_argument("--gate-out", default=None,
                    help="npz path for gate activations (cross-process comparison)")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out_root = Path(args.out)
    conditions = args.conditions or [f"{o}/{c}" for o in OBJECTS for c in CONDITIONS]

    print(f"Loading policy {args.config_name} from {args.checkpoint} ...", flush=True)
    policy = load_policy(args.checkpoint, args.config_name, args.device)
    tap_lm, tap_ex, lm_names, ex_names = attach_taps(policy)
    tokenize_meta = make_tokenizer_meta_fn(policy)
    noise = fixed_noise()

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project="mass-com-vla-probing", job_type="capture",
            name="pi05-capture" + ("-gate" if args.test_determinism else ""),
            config={"checkpoint": str(Path(args.checkpoint).expanduser()),
                    "config_name": args.config_name, "seed": SEED,
                    "noise_shape": [ACTION_HORIZON, ACTION_DIM_PADDED],
                    "conditions": conditions, "git_sha": git_sha()})

    gate_out = Path(args.gate_out) if args.gate_out else out_root / "determinism_gate.npz"
    gate_verdict = run_determinism_gate(
        policy, tap_lm, tap_ex, tokenize_meta, corpus, noise, gate_out)
    if wandb_run is not None:
        wandb_run.summary["determinism_gate"] = gate_verdict["result"]
    if gate_verdict["result"] == "BLOCKED":
        print("DETERMINISM GATE FAILED (diff >= 1e-6) -- aborting capture.", flush=True)
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise SystemExit(2)
    if args.test_determinism:
        if wandb_run is not None:
            wandb_run.finish()
        return

    per_condition = {}
    meta_path = out_root / "meta.json"
    if meta_path.exists():
        per_condition = json.loads(meta_path.read_text()).get("per_condition", {})
    for cond in conditions:
        cond_dir = corpus / cond
        print(f"[capture] {cond}", flush=True)
        stats = capture_condition(policy, tap_lm, tap_ex, tokenize_meta,
                                  cond_dir, out_root / cond, noise)
        per_condition[cond] = stats
        print(f"[capture] {cond}: T={stats['T']} wall={stats['wall_time_s']}s", flush=True)
        if wandb_run is not None:
            import wandb
            wandb_run.log({"condition": cond, "T": stats["T"],
                           "wall_time_s": stats["wall_time_s"]})

    meta = build_meta(args, lm_names, ex_names, per_condition, gate_verdict)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {meta_path}", flush=True)
    if wandb_run is not None:
        wandb_run.summary["num_conditions"] = len(per_condition)
        wandb_run.summary["total_steps"] = sum(s["T"] for s in per_condition.values())
        wandb_run.finish()


if __name__ == "__main__":
    main()
