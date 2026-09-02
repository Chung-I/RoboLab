# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CRN activation-patching harness for pi0.5 (Plan-3 Task 5).

Runs with the openpi venv (~/Codes/openpi/.venv/bin/python), from the
worktree root:

    ~/Codes/openpi/.venv/bin/python -u analysis/mass_com/patch_pi05.py \\
        --mode {freeze,gate,sweep,assemble} [...]

Causal question: which token blocks / layers *causally* carry the
between-condition action difference? For a replay-matched pair (clean obs
from cond_a, corrupt obs from cond_b, same object, time-aligned per the
frozen pair family), we run

    clean    = infer(obs_clean,  noise=z)                    (recorded)
    corrupt  = infer(obs_corrupt, noise=z)                   (recorded)
    patched  = infer(obs_clean,  noise=z, site <- corrupt's cached act)

with common random numbers (one fixed z for all three; convert_pi05.md
verified the `noise=` override sets the flow ODE start), BOTH directions per
pair (swap clean/corrupt), and per-pair floors:

    reseed      = infer(obs_clean, noise=z2)   (fresh noise, no patch)
    degradation = patched, but the donor activation comes from an unrelated
                  episode (the OTHER object, same condition name, family-
                  aligned step) at the same site

Model mechanics (openpi src/openpi/models_pytorch/pi0_pytorch.py —
investigated before writing, line numbers in the task report):
`sample_actions` runs the PaliGemma prefix exactly once to build the KV
cache (L394-400); each `paligemma.language_model.layers[i]`
(GemmaDecoderLayer) fires one forward over the (B, 968, 2048) prefix, so a
forward hook that mutates `output[0][row, lo:hi, :]` in place patches the
post-block residual stream of one token block for one batch row; downstream
layers recompute their K/V from the patched stream. (Patching PG layer 17
is causally inert by construction — the prefix pass's last_hidden_state is
discarded; kept in the sweep as a per-row harness null check.) The
denoising loop (L405-419) calls `denoise_step` (L422-462) `num_steps=10`
times; each call re-runs `gemma_expert.model.layers[i]` over the
(B, 15, 1024) suffix, so expert hooks naturally fire on EVERY denoising
iteration; the expert patch writes the corrupt run's cached iteration-k
activation at iteration k, across all 15 action tokens. Token-block ranges
come from the capture meta (img_cam1 [0,256), img_cam2 [256,512)) and, for
text/state, from the policy's own tokenizer via
capture_pi05.make_tokenizer_meta_fn (per-step; patched length =
min(target, donor) from each block's own start).

Batched CRN engine: every forward in the experiment — the record batch
(clean/corrupt/degradation donors + reseed rows) and every patched batch —
runs at ONE fixed padded batch size (--batch-size, default 16). With the
batch shape fixed, kernel selection is identical across all runs and each
row's output depends only on that row's inputs, so actions compared across
batches are exactly comparable; the gate verifies this empirically
(row-position swap bit-identity, repeat bit-identity, and the PG17-inert
patch reproducing the record batch's clean action bit-for-bit across
batches). torch.compile is disabled for the whole harness
(TORCHDYNAMO_DISABLE=1 below): compiled graphs bake forward hooks in at
trace time and were observed to silently skip per-call patch hooks.

Metrics are computed on the post-transform (15, 8) action chunks — the same
space as the parity check — via patch_pairs.patch_metrics (full panel:
signed delta-projection, orthogonal residual, total, per-dim, per-step).

Pairs/sites/thresholds are frozen in analysis/mass_com/pairs_frozen.json
BEFORE the sweep (--mode freeze). The sweep checkpoints one parquet per
pair under --parts-dir and is resumable; drive it in bounded foreground
calls with --max-seconds.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# PI0Pytorch wraps sample_actions in torch.compile (pi0_pytorch.py:112-113).
# Compiled graphs bake forward hooks in at trace time, so per-call Patcher
# hooks are silently SKIPPED once the recompile cache saturates (observed:
# a PG-17 patch hook fired 0x under compile). The harness registers and
# removes hooks around every inference, so it must run fully eager; set
# before any torch import. All runs share the same eager numerics.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import capture_pi05 as cap  # noqa: E402
import patch_pairs as pp  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen experiment grid (written into pairs_frozen.json by --mode freeze)
# ---------------------------------------------------------------------------

PG_LAYERS_FULL = [0, 3, 5, 7, 9, 11, 13, 15, 17]
PG_LAYERS_REDUCED = [0, 5, 9, 11, 15, 17]  # pre-authorized budget fallback
EXPERT_LAYERS = [0, 6, 12, 17]
PG_BLOCKS = ["img_cam1", "img_cam2", "text", "state"]
MAX_PAIRS_PER_CELL = 20
SUBSAMPLE_SEED = 0
RESEED_RNG_SEED = 1  # z2 = default_rng(1).standard_normal((15, 32))
DEFAULT_BATCH = 16

# Pre-registered decision thresholds (see freeze payload for prose)
THRESHOLDS = {
    "min_median_proj": 0.10,
    "reseed_multiple": 3.0,
    "baseline_blocks": ["text"],
    "rule": (
        "A site is reported causal for a (family, direction) iff its median "
        "proj across pairs exceeds min_median_proj, exceeds reseed_multiple x "
        "the median |reseed_proj|, exceeds the median |deg_proj| at the same "
        "site (degradation floor), and exceeds the per-family/direction "
        "baseline = median across pairs of the per-pair MAX proj over "
        "non-hypothesized (text) block sites; and the site's median proj is "
        "positive in BOTH directions. Text-block sites themselves are floor "
        "material, never headline sites."
    ),
}


def make_sites(pg_layers) -> list[dict]:
    sites = [{"kind": "pg", "layer": l, "block": b}
             for l in pg_layers for b in PG_BLOCKS]
    sites += [{"kind": "expert", "layer": l, "block": "suffix"}
              for l in EXPERT_LAYERS]
    return sites


def reseed_noise() -> np.ndarray:
    return np.random.default_rng(RESEED_RNG_SEED).standard_normal(
        (cap.ACTION_HORIZON, cap.ACTION_DIM_PADDED)).astype(np.float32)


# ---------------------------------------------------------------------------
# Obs building (reuses capture_pi05's request construction)
# ---------------------------------------------------------------------------


class ObsProvider:
    """Random-access step requests per corpus condition (lazy h5 handles)."""

    def __init__(self, corpus: Path):
        self.corpus = Path(corpus)
        self._h5 = {}
        self._prompt = {}

    def _cond(self, obj, cond):
        import h5py

        key = (obj, cond)
        if key not in self._h5:
            cond_dir = self.corpus / obj / cond
            self._h5[key] = h5py.File(cond_dir / "replay.hdf5", "r")
            self._prompt[key] = json.loads(
                (cond_dir / "env_cfg.json").read_text())["instruction"]
        return self._h5[key], self._prompt[key]

    def request(self, obj, cond, t) -> dict:
        f, prompt = self._cond(obj, cond)
        demo = f["data/demo_0"]
        return cap.build_step_request(
            demo["obs/image_obs/over_shoulder_left_camera"][t],
            demo["obs/image_obs/wrist_cam"][t],
            demo["obs/proprio_obs/arm_joint_pos"][t],
            demo["obs/proprio_obs/gripper_pos"][t],
            prompt)


# ---------------------------------------------------------------------------
# Batched engine: recording + patch hooks over model.sample_actions
# ---------------------------------------------------------------------------


class _RecordHooks:
    """Caches full-batch residual-stream outputs.

    prefix[layer] -> (B, 968, 2048) CPU tensor (native dtype), 1 firing
    expert[layer] -> list of 10 (B, 15, 1024) CPU tensors (denoise steps)
    """

    def __init__(self, model, pg_layers, expert_layers):
        self.prefix = {}
        self.expert = {l: [] for l in expert_layers}
        self._handles = []
        lm = model.paligemma_with_expert.paligemma.language_model.layers
        ex = model.paligemma_with_expert.gemma_expert.model.layers

        def pg_hook(layer):
            def hook(_m, _i, out):
                assert layer not in self.prefix, f"pg layer {layer} fired twice"
                self.prefix[layer] = _hidden(out).detach().cpu().clone()
            return hook

        def ex_hook(layer):
            def hook(_m, _i, out):
                self.expert[layer].append(_hidden(out).detach().cpu().clone())
            return hook

        for l in pg_layers:
            self._handles.append(lm[l].register_forward_hook(pg_hook(l)))
        for l in expert_layers:
            self._handles.append(ex[l].register_forward_hook(ex_hook(l)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def validate(self, pg_layers, expert_layers):
        assert set(self.prefix) == set(pg_layers), sorted(self.prefix)
        for l in expert_layers:
            n = len(self.expert[l])
            assert n == cap.NUM_DENOISE_STEPS, f"expert layer {l} fired {n}x"


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


class _PatchHooks:
    """Applies per-row site patches during one batched sample_actions call.

    jobs: list of dicts
      pg:     {"kind": "pg", "layer", "row", "lo_t", "n", "donor"}
              donor: (n, 2048) CPU tensor (pre-sliced from the record batch)
      expert: {"kind": "expert", "layer", "row", "donor_iters"}
              donor_iters: list of 10 (15, 1024) CPU tensors
    """

    def __init__(self, model, jobs):
        self._handles = []
        self.fired = {"pg": 0, "expert": 0}
        pg_by_layer, ex_by_layer = {}, {}
        for j in jobs:
            (pg_by_layer if j["kind"] == "pg" else ex_by_layer).setdefault(
                j["layer"], []).append(j)
        self._n_pg = sum(len(v) for v in pg_by_layer.values())
        self._n_ex_layers = len(ex_by_layer)
        lm = model.paligemma_with_expert.paligemma.language_model.layers
        ex = model.paligemma_with_expert.gemma_expert.model.layers

        def pg_hook(layer_jobs):
            def hook(_m, _i, out):
                h = _hidden(out)
                for j in layer_jobs:
                    h[j["row"], j["lo_t"]:j["lo_t"] + j["n"], :] = (
                        j["donor"].to(h.device, h.dtype))
                    self.fired["pg"] += 1
            return hook

        def ex_hook(layer_jobs):
            k = {"i": 0}

            def hook(_m, _i, out):
                h = _hidden(out)
                for j in layer_jobs:
                    h[j["row"], :, :] = j["donor_iters"][k["i"]].to(h.device, h.dtype)
                k["i"] += 1
                self.fired["expert"] += 1
            return hook

        for layer, layer_jobs in pg_by_layer.items():
            self._handles.append(lm[layer].register_forward_hook(pg_hook(layer_jobs)))
        for layer, layer_jobs in ex_by_layer.items():
            self._handles.append(ex[layer].register_forward_hook(ex_hook(layer_jobs)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def validate(self):
        assert self.fired["pg"] == self._n_pg, self.fired
        want = self._n_ex_layers * cap.NUM_DENOISE_STEPS
        assert self.fired["expert"] == want, (self.fired, want)


class Engine:
    """Fixed-batch-size CRN inference engine over model.sample_actions.

    Replicates Policy.infer's plumbing (policy.py:68-106: input transform ->
    batch dim -> Observation.from_dict -> sample_actions -> output transform)
    at batch size B, padding every batch to exactly B rows so that all runs
    share one kernel/shape configuration.
    """

    def __init__(self, policy, corpus: Path, pg_layers, expert_layers,
                 batch_size=DEFAULT_BATCH):
        self.policy = policy
        self.model = policy._model
        self.device = policy._pytorch_device
        self.B = batch_size
        self.obs = ObsProvider(corpus)
        self.tokenize_meta = cap.make_tokenizer_meta_fn(policy)
        self.pg_layers = list(pg_layers)
        self.expert_layers = list(expert_layers)
        self.noise = cap.fixed_noise()
        self.noise2 = reseed_noise()
        self._step_cache = {}
        # per-condition timing info for degradation-donor step mapping
        self.cond_info = {}
        for obj_dir in sorted(p for p in Path(corpus).iterdir() if p.is_dir()):
            for cond_dir in sorted(p for p in obj_dir.iterdir() if p.is_dir()):
                if (cond_dir / "ft.npz").exists():
                    self.cond_info[(obj_dir.name, cond_dir.name)] = (
                        pp._load_cond(cond_dir))
        self.objects = sorted({o for o, _ in self.cond_info})

    # -- per-step transformed inputs (cached) -------------------------------

    def step_inputs(self, obj, cond, t) -> dict:
        """{"inputs": transformed numpy dict, "blocks": absolute (lo, hi)}"""
        key = (obj, cond, t)
        if key not in self._step_cache:
            import jax

            request = self.obs.request(obj, cond, t)
            _n_valid, rel_blocks = self.tokenize_meta(request)
            blocks = {k: (cap.LANG_OFFSET + lo, cap.LANG_OFFSET + hi)
                      for k, (lo, hi) in rel_blocks.items()}
            inputs = self.policy._input_transform(
                jax.tree.map(lambda x: x, request))
            self._step_cache[key] = {"inputs": inputs, "blocks": blocks}
            if len(self._step_cache) > 64:
                self._step_cache.pop(next(iter(self._step_cache)))
        return self._step_cache[key]

    # -- one padded batch ---------------------------------------------------

    def run_batch(self, rows, noises, record=False, jobs=None):
        """rows: list of transformed input dicts (numpy), len <= B.
        noises: per-row (15, 32) float32. Returns (actions list [(15, 8)],
        records or None). Batches are padded to exactly B by repeating
        row 0 (with noise z) so every forward shares one shape."""
        import jax
        import torch

        from openpi.models import model as _model

        n = len(rows)
        assert 1 <= n <= self.B, n
        padded = list(rows) + [rows[0]] * (self.B - n)
        pad_noises = list(noises) + [self.noise] * (self.B - n)
        batched = jax.tree.map(
            lambda *xs: torch.from_numpy(np.stack([np.asarray(x) for x in xs]))
            .to(self.device), *padded)
        noise_t = torch.from_numpy(np.stack(pad_noises)).to(self.device)
        observation = _model.Observation.from_dict(batched)

        rec = _RecordHooks(self.model, self.pg_layers, self.expert_layers) \
            if record else None
        patch = _PatchHooks(self.model, jobs) if jobs else None
        try:
            torch.manual_seed(cap.SEED)
            with torch.no_grad():
                acts = self.model.sample_actions(self.device, observation,
                                                 noise=noise_t)
        finally:
            if rec is not None:
                rec.remove()
            if patch is not None:
                patch.remove()
        if rec is not None:
            rec.validate(self.pg_layers, self.expert_layers)
        if patch is not None:
            patch.validate()
        acts = np.asarray(acts.detach().to("cpu", torch.float32))
        out_actions = []
        for i in range(n):
            out = self.policy._output_transform(
                {"state": np.asarray(padded[i]["state"]), "actions": acts[i]})
            a = np.asarray(out["actions"], dtype=np.float32)
            assert a.shape == (cap.ACTION_HORIZON, cap.ACTION_DIM_OUT), a.shape
            out_actions.append(a)
        records = None
        if rec is not None:
            records = {"prefix": rec.prefix, "expert": rec.expert}
        return out_actions, records

    # -- degradation donor --------------------------------------------------

    def degradation_key(self, obj, cond_corrupt, family, step_rel):
        """Unrelated episode: the OTHER object, same condition name, family-
        aligned step (clipped into the episode)."""
        other = next(o for o in self.objects if o != obj)
        info = self.cond_info[(other, cond_corrupt)]
        base = info["anchor"] if family == "anchor" or info["lift"] is None \
            else info["lift"]
        t = int(np.clip(base + step_rel, 0, info["T"] - 1))
        return other, cond_corrupt, t

    # -- job construction ---------------------------------------------------

    def make_job(self, row, site, donor_row, records, target_blocks,
                 donor_blocks):
        kind, layer, block = site["kind"], site["layer"], site["block"]
        if kind == "pg":
            if block in cap.IMG_BLOCKS:
                lo_t, hi_t = cap.IMG_BLOCKS[block]
                lo_d, n = lo_t, hi_t - lo_t
            else:
                lo_t, hi_t = target_blocks[block]
                lo_d, hi_d = donor_blocks[block]
                n = min(hi_t - lo_t, hi_d - lo_d)
            donor = records["prefix"][layer][donor_row, lo_d:lo_d + n].clone()
            return {"kind": "pg", "layer": layer, "row": row,
                    "lo_t": lo_t, "n": n, "donor": donor}, n
        assert block == "suffix"
        donor_iters = [records["expert"][layer][k][donor_row].clone()
                       for k in range(cap.NUM_DENOISE_STEPS)]
        return {"kind": "expert", "layer": layer, "row": row,
                "donor_iters": donor_iters}, cap.ACTION_HORIZON

    # -- one frozen pair, both directions, all sites ------------------------

    def run_pair(self, pair: dict, sites: list[dict]) -> list[dict]:
        obj, family, step_rel = pair["object"], pair["family"], pair["step_rel"]
        dirs = {
            "a2b": (pair["cond_a"], pair["t_a"], pair["cond_b"], pair["t_b"]),
            "b2a": (pair["cond_b"], pair["t_b"], pair["cond_a"], pair["t_a"]),
        }
        # record batch: clean/corrupt (= the two conditions), the two
        # degradation donors, and the two reseed rows — one padded forward
        step_a = self.step_inputs(obj, pair["cond_a"], pair["t_a"])
        step_b = self.step_inputs(obj, pair["cond_b"], pair["t_b"])
        deg_keys = {d: self.degradation_key(obj, c_cor, family, step_rel)
                    for d, (_, _, c_cor, _) in dirs.items()}
        deg_steps = {d: self.step_inputs(*k) for d, k in deg_keys.items()}
        rows = [step_a["inputs"], step_b["inputs"],
                deg_steps["a2b"]["inputs"], deg_steps["b2a"]["inputs"],
                step_a["inputs"], step_b["inputs"]]
        noises = [self.noise, self.noise, self.noise, self.noise,
                  self.noise2, self.noise2]
        actions, records = self.run_batch(rows, noises, record=True)
        a_by_cond = {pair["cond_a"]: actions[0], pair["cond_b"]: actions[1]}
        reseed_by_cond = {pair["cond_a"]: actions[4], pair["cond_b"]: actions[5]}
        rec_row = {pair["cond_a"]: 0, pair["cond_b"]: 1}
        deg_row = {"a2b": 2, "b2a": 3}
        blocks = {pair["cond_a"]: step_a["blocks"], pair["cond_b"]: step_b["blocks"]}
        deg_blocks = {d: deg_steps[d]["blocks"] for d in dirs}

        # patched batches: rows are always the clean obs with noise z; jobs
        # patch each row at one site from either the corrupt record row
        # (patch) or the unrelated record row (degradation)
        specs = []  # (direction, site, source)
        for direction in dirs:
            for site in sites:
                specs.append((direction, site, "patch"))
                specs.append((direction, site, "deg"))
        results = {}
        for lo in range(0, len(specs), self.B):
            chunk = specs[lo:lo + self.B]
            rows_c, noises_c, jobs = [], [], []
            for i, (direction, site, source) in enumerate(chunk):
                c_cln, _t_cln, c_cor, _ = dirs[direction]
                rows_c.append(self.step_inputs(
                    obj, c_cln, dirs[direction][1])["inputs"])
                noises_c.append(self.noise)
                drow = rec_row[c_cor] if source == "patch" else deg_row[direction]
                dblocks = blocks[c_cor] if source == "patch" else deg_blocks[direction]
                job, n_tok = self.make_job(i, site, drow, records,
                                           blocks[c_cln], dblocks)
                jobs.append(job)
                results[(direction, _site_key(site), source, "n_tok")] = n_tok
            acts_c, _ = self.run_batch(rows_c, noises_c, jobs=jobs)
            for i, (direction, site, source) in enumerate(chunk):
                results[(direction, _site_key(site), source)] = acts_c[i]

        rows_out = []
        for direction, (c_cln, t_cln, c_cor, t_cor) in dirs.items():
            a_clean, a_corrupt = a_by_cond[c_cln], a_by_cond[c_cor]
            m_reseed = pp.patch_metrics(a_clean, a_corrupt, reseed_by_cond[c_cln])
            dk = deg_keys[direction]
            for site in sites:
                sk = _site_key(site)
                m = pp.patch_metrics(a_clean, a_corrupt,
                                     results[(direction, sk, "patch")])
                md = pp.patch_metrics(a_clean, a_corrupt,
                                      results[(direction, sk, "deg")])
                rows_out.append({
                    "pair_id": pair["pair_id"], "object": obj,
                    "family": family, "step_rel": step_rel,
                    "direction": direction, "cond_clean": c_cln,
                    "cond_corrupt": c_cor, "t_clean": t_cln, "t_corrupt": t_cor,
                    "site_kind": site["kind"], "layer": site["layer"],
                    "block": site["block"],
                    "n_tokens_patched": results[(direction, sk, "patch", "n_tok")],
                    **{k: m[k] for k in ("proj", "resid", "total",
                                         "delta_norm", "degenerate",
                                         "per_dim_proj", "per_dim_total",
                                         "per_step_proj", "per_step_total")},
                    "deg_proj": md["proj"], "deg_resid": md["resid"],
                    "deg_total": md["total"],
                    "reseed_proj": m_reseed["proj"],
                    "reseed_resid": m_reseed["resid"],
                    "reseed_total": m_reseed["total"],
                    "deg_object": dk[0], "deg_cond": dk[1], "deg_t": dk[2],
                })
        return rows_out


def _site_key(site):
    return (site["kind"], site["layer"], site["block"])


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_freeze(args):
    """Write pairs_frozen.json BEFORE any sweep result exists."""
    frozen_path = Path(args.pairs_json)
    if frozen_path.exists() and not args.force_refreeze:
        raise SystemExit(f"{frozen_path} already exists; refusing to refreeze "
                         "(pre-registration). Use --force-refreeze to write a "
                         "NEW file and document the change in the report.")
    df = pp.build_pairs(args.corpus)
    sub = pp.subsample_pairs(df, max_per_cell=MAX_PAIRS_PER_CELL,
                             seed=SUBSAMPLE_SEED).reset_index(drop=True)
    sub.insert(0, "pair_id", sub.index)
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": cap.git_sha(),
        "corpus": str(args.corpus),
        "n_pairs_total_before_subsample": int(len(df)),
        "subsample": {"max_per_cell": MAX_PAIRS_PER_CELL,
                      "seed": SUBSAMPLE_SEED},
        "noise": {"z": "np.random.default_rng(0).standard_normal((15,32)) f32",
                  "z2_reseed": "np.random.default_rng(1).standard_normal((15,32)) f32"},
        "sites": {
            "pg_layers_full": PG_LAYERS_FULL,
            "pg_layers_reduced": PG_LAYERS_REDUCED,
            "expert_layers": EXPERT_LAYERS,
            "pg_blocks": PG_BLOCKS,
            "budget_rule": ("run the full pg layer set unless the measured "
                            "projection exceeds 4 h GPU; then drop layers "
                            "{3, 7, 13} (the reduced set) and document — "
                            "pre-authorized by the plan"),
        },
        "directions": ["a2b", "b2a"],
        "floors": {
            "reseed": "same obs, fresh noise z2, no patch",
            "degradation": ("same site patched from an unrelated episode: the "
                            "other object, same condition name, family-aligned "
                            "step (anchor/lift + step_rel, clipped)"),
        },
        "thresholds": THRESHOLDS,
        "pairs": sub.to_dict(orient="records"),
    }
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(payload, indent=1) + "\n")
    counts = sub.groupby(["object", "family"]).size()
    print(f"froze {len(sub)} pairs (of {len(df)}) -> {frozen_path}")
    print(counts.to_string())


def _load_frozen(args):
    payload = json.loads(Path(args.pairs_json).read_text())
    pg_layers = (payload["sites"]["pg_layers_reduced"] if args.site_set == "reduced"
                 else payload["sites"]["pg_layers_full"])
    sites = make_sites(pg_layers)
    return payload, sites, pg_layers


def _make_engine(args, pg_layers):
    policy = cap.load_policy(args.checkpoint, args.config_name, args.device)
    return Engine(policy, Path(args.corpus), pg_layers, EXPERT_LAYERS,
                  batch_size=args.batch_size)


def mode_gate(args):
    """Determinism + batch-consistency gate (all conditions must hold before
    the sweep):
      1. two CRN repeats of the same patched batch -> bit-identical actions
         at a PG site and an expert site;
      2. row-position invariance: swapping which row an obs occupies (with
         different sibling rows) leaves its action bit-identical;
      3. cross-batch null: a PG-17 patch (causally inert by construction)
         reproduces the record batch's clean action bit-for-bit — exactly
         the cross-batch comparison the metrics rely on."""
    payload, sites, pg_layers = _load_frozen(args)
    eng = _make_engine(args, pg_layers)
    pair = dict(payload["pairs"][0])
    obj = pair["object"]
    step_a = eng.step_inputs(obj, pair["cond_a"], pair["t_a"])
    step_b = eng.step_inputs(obj, pair["cond_b"], pair["t_b"])
    verdict = {"pair_id": pair["pair_id"], "batch_size": eng.B, "checks": {},
               "result": "PASS"}

    def check(name, ok, detail=""):
        verdict["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            verdict["result"] = "FAIL"

    # record batch (as the sweep runs it, minus deg rows)
    acts0, records = eng.run_batch(
        [step_a["inputs"], step_b["inputs"]], [eng.noise, eng.noise],
        record=True)
    a_clean, _a_corrupt = acts0

    # 1. CRN repeats of one patched batch (PG + expert sites in one batch)
    gate_sites = [{"kind": "pg", "layer": pg_layers[1], "block": "img_cam1"},
                  {"kind": "expert", "layer": EXPERT_LAYERS[1], "block": "suffix"}]
    jobs = []
    for i, site in enumerate(gate_sites):
        job, _ = eng.make_job(i, site, 1, records, step_a["blocks"], step_b["blocks"])
        jobs.append(job)
    rows = [step_a["inputs"]] * len(gate_sites)
    noises = [eng.noise] * len(gate_sites)
    p1, _ = eng.run_batch(rows, noises, jobs=jobs)
    p2, _ = eng.run_batch(rows, noises, jobs=jobs)
    for i, site in enumerate(gate_sites):
        check(f"crn_repeat_bit_identical_{site['kind']}{site['layer']}",
              np.array_equal(p1[i], p2[i]),
              f"max_abs_move_vs_clean={np.abs(p1[i] - a_clean).max():.2e}")

    # 2. row-position invariance (different position, different siblings)
    s1, _ = eng.run_batch([step_a["inputs"], step_b["inputs"]],
                          [eng.noise, eng.noise])
    s2, _ = eng.run_batch([step_b["inputs"], step_a["inputs"]],
                          [eng.noise, eng.noise])
    check("row_position_invariance_bit_identical",
          np.array_equal(s1[0], s2[1]) and np.array_equal(s1[1], s2[0]))

    # 3. PG17 inert patch reproduces clean across batches
    job17, _ = eng.make_job(0, {"kind": "pg", "layer": 17, "block": "img_cam1"},
                            1, records, step_a["blocks"], step_b["blocks"])
    p17, _ = eng.run_batch([step_a["inputs"]], [eng.noise], jobs=[job17])
    check("pg17_inert_cross_batch_bit_identical",
          np.array_equal(p17[0], a_clean))

    print(json.dumps(verdict, indent=1), flush=True)
    out = Path(args.parts_dir).parent / "patching_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=1) + "\n")
    if verdict["result"] != "PASS":
        raise SystemExit(2)


def mode_sweep(args):
    import pandas as pd

    payload, sites, pg_layers = _load_frozen(args)
    parts = Path(args.parts_dir)
    parts.mkdir(parents=True, exist_ok=True)
    pending = [p for p in payload["pairs"]
               if not (parts / f"pair_{p['pair_id']:04d}.parquet").exists()]
    if args.only_pairs is not None:
        want = {int(x) for x in args.only_pairs.split(",")}
        pending = [p for p in pending if p["pair_id"] in want]
    print(f"pairs pending: {len(pending)} / {len(payload['pairs'])}; "
          f"sites per direction: {len(sites)} (pg layers {pg_layers}); "
          f"batch {args.batch_size}", flush=True)
    if not pending:
        print("SWEEP_COMPLETE", flush=True)
        return
    eng = _make_engine(args, pg_layers)
    t_start = time.monotonic()
    done = 0
    for pair in pending:
        t0 = time.monotonic()
        rows = eng.run_pair(pair, sites)
        pd.DataFrame(rows).to_parquet(
            parts / f"pair_{pair['pair_id']:04d}.parquet", index=False)
        done += 1
        dt = time.monotonic() - t0
        elapsed = time.monotonic() - t_start
        print(f"pair {pair['pair_id']} ({pair['object']}/{pair['family']}"
              f"/s{pair['step_rel']}) rows={len(rows)} {dt:.1f}s "
              f"[{done}/{len(pending)} in {elapsed:.0f}s]", flush=True)
        if args.max_seconds and elapsed > args.max_seconds:
            print(f"time budget reached after {done} pairs", flush=True)
            break
    remaining = len(pending) - done
    if remaining == 0:
        print("SWEEP_COMPLETE", flush=True)
    else:
        per = (time.monotonic() - t_start) / max(done, 1)
        print(f"REMAINING {remaining} pairs, ~{per:.1f}s/pair "
              f"-> ~{remaining * per / 3600:.2f} h", flush=True)


def mode_assemble(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    payload, sites, _ = _load_frozen(args)
    parts = sorted(Path(args.parts_dir).glob("pair_*.parquet"))
    n_frozen = len(payload["pairs"])
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    print(f"assembled {len(df)} rows from {len(parts)}/{n_frozen} pairs")

    # harness null check: every PG17 row must be exactly zero-effect
    pg17 = df[(df.site_kind == "pg") & (df.layer == 17)]
    n_bad = int(((pg17.total != 0) | (pg17.deg_total != 0)).sum())
    print(f"PG17 inertness check: {len(pg17)} rows, {n_bad} with "
          f"total != 0 or deg_total != 0")
    if n_bad:
        print("WARNING: PG17 rows moved actions — harness assumption broken")

    # baseline column: per (pair, direction) MAX proj over non-hypothesized
    # (text) block sites [adj 12/19]
    base_blocks = payload["thresholds"]["baseline_blocks"]
    base = (df[df.block.isin(base_blocks) & ~df.degenerate]
            .groupby(["pair_id", "direction"]).proj.max()
            .rename("baseline_text_max_proj").reset_index())
    df = df.merge(base, on=["pair_id", "direction"], how="left")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out}")

    scored = df[~df.degenerate]
    agg = (scored.groupby(["family", "direction", "site_kind", "layer", "block"])
           .agg(median_proj=("proj", "median"),
                iqr_lo=("proj", lambda s: s.quantile(0.25)),
                iqr_hi=("proj", lambda s: s.quantile(0.75)),
                median_resid=("resid", "median"),
                median_total=("total", "median"),
                median_delta=("delta_norm", "median"),
                median_abs_reseed=("reseed_proj", lambda s: s.abs().median()),
                p95_abs_reseed=("reseed_proj", lambda s: s.abs().quantile(0.95)),
                median_abs_deg=("deg_proj", lambda s: s.abs().median()),
                median_baseline=("baseline_text_max_proj", "median"),
                n=("proj", "size"))
           .reset_index())
    th = payload["thresholds"]
    agg["passes"] = ((agg.median_proj > th["min_median_proj"])
                     & (agg.median_proj > th["reseed_multiple"] * agg.median_abs_reseed)
                     & (agg.median_proj > agg.median_abs_deg)
                     & (agg.median_proj > agg.median_baseline)
                     & ~agg.block.isin(th["baseline_blocks"]))
    # both-direction consistency
    key = ["family", "site_kind", "layer", "block"]
    both = agg.groupby(key).median_proj.min().rename("min_dir_median_proj")
    agg = agg.merge(both.reset_index(), on=key)
    agg["passes"] = agg.passes & (agg.min_dir_median_proj > 0)
    agg_path = out.with_name("patching_sites.parquet")
    agg.to_parquet(agg_path, index=False)

    print("\nTop sites by median proj (per family/direction):")
    top = (agg.sort_values("median_proj", ascending=False)
           .groupby(["family", "direction"]).head(5))
    cols = ["family", "direction", "site_kind", "layer", "block",
            "median_proj", "median_abs_reseed", "median_abs_deg",
            "median_baseline", "min_dir_median_proj", "passes", "n"]
    print(top[cols].to_string(index=False, float_format=lambda v: f"{v: .3f}"))

    # figures: one file per (family, block) — proj vs layer, directions side
    # by side, floors shaded
    fig_dir = out.parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig_paths = []
    for family in sorted(scored.family.unique()):
        fam = agg[agg.family == family]
        for block in PG_BLOCKS + ["suffix"]:
            sub = fam[fam.block == block]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4.2))
            for direction, color in (("a2b", "tab:blue"), ("b2a", "tab:orange")):
                d = sub[sub.direction == direction].sort_values("layer")
                ax.plot(d.layer, d.median_proj, "o-", color=color,
                        label=f"patched ({direction})")
                ax.fill_between(d.layer, d.iqr_lo, d.iqr_hi, color=color, alpha=0.15)
                ax.plot(d.layer, d.median_abs_deg, "s--", color=color, alpha=0.5,
                        label=f"degradation floor ({direction})")
            floor = sub.p95_abs_reseed.max()
            ax.axhspan(-floor, floor, color="gray", alpha=0.2,
                       label="reseed floor (p95 |proj|)")
            base_med = float(sub.median_baseline.median())
            if np.isfinite(base_med):
                ax.axhline(base_med, color="k", ls=":", lw=1,
                           label="text-block baseline (median)")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("layer" + (" (expert)" if block == "suffix" else " (PaliGemma)"))
            ax.set_ylabel("signed delta-projection (median across pairs)")
            ax.set_title(f"patch effect vs layer — {family} family, block {block}")
            ax.legend(fontsize=7)
            fig.tight_layout()
            p = fig_dir / f"patch_proj_vs_layer_{family}_{block}.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            fig_paths.append(p)
    print(f"wrote {len(fig_paths)} figures to {fig_dir}")

    if not args.no_wandb:
        import wandb

        run = wandb.init(project="mass-com-vla-probing", job_type="analysis",
                         name="phase3-patching",
                         config={"n_pairs": int(df.pair_id.nunique()),
                                 "n_rows": int(len(df)),
                                 "sites_per_direction": len(sites),
                                 "thresholds": payload["thresholds"],
                                 "subsample": payload["subsample"],
                                 "batch_size": args.batch_size,
                                 "git_sha": cap.git_sha()})
        run.log({"sites_table": wandb.Table(dataframe=agg.round(4))})
        run.log({p.stem: wandb.Image(str(p)) for p in fig_paths})
        run.summary["n_passing_sites"] = int(agg.passes.sum())
        run.summary["pg17_nonzero_rows"] = n_bad
        run.finish()
        print(f"wandb: {run.url}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["freeze", "gate", "sweep", "assemble"])
    ap.add_argument("--corpus", default="output/replay_corpus")
    ap.add_argument("--pairs-json", default="analysis/mass_com/pairs_frozen.json")
    ap.add_argument("--parts-dir", default="output/probe_results/pi05/patching_parts")
    ap.add_argument("--out", default="output/probe_results/pi05/patching.parquet")
    ap.add_argument("--checkpoint", default=cap.DEFAULT_CHECKPOINT)
    ap.add_argument("--config-name", default=cap.DEFAULT_CONFIG_NAME)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--site-set", default="full", choices=["full", "reduced"])
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--max-seconds", type=float, default=540)
    ap.add_argument("--only-pairs", default=None,
                    help="comma-separated pair_ids (smoke runs)")
    ap.add_argument("--force-refreeze", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    {"freeze": mode_freeze, "gate": mode_gate,
     "sweep": mode_sweep, "assemble": mode_assemble}[args.mode](args)


if __name__ == "__main__":
    main()
