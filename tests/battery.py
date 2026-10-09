"""The golden battery: every read path, intervention path and frontend payload
of an engine, run on a fixed set of positions and flattened into something that
can be stored and compared.

`make_goldens.py` runs it once and writes `golden/<engine>.json.gz` (+ `.npz`);
`test_golden.py` runs it again and compares. Only the legacy public surface is
exercised, so the battery is exactly the contract a refactor has to keep. The
goldens are local working data (gitignored): see make_goldens.py for how to
write them from a known-good commit.

The app bridge's model-specific names were dropped for neutral ones (MaiaApi ->
LensApi, maia_move -> engine_move, maia_* payload keys -> engine_*); the
battery speaks both and compares under the neutral names.

Comparison is golden-subset: every key recorded in the golden must still be
there with the same value, while keys a later version adds are ignored. Values
must match bit for bit when the torch version matches the one that wrote the
golden; otherwise (or with CHESSFORMER_GOLDEN_RTOL set) to a relative
tolerance.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import threading
from pathlib import Path

import chess
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(os.environ.get("CHESSFORMER_GOLDEN_DIR") or Path(__file__).parent / "golden")

# Arrays up to this many elements are stored whole; larger ones as a hash plus
# summary statistics (exact via the hash, approximate via the stats).
SMALL = 1024
LONG_STR = 4096

ELO, OPPO = 1500, 1300

# name -> (fen, moves played from it). Chosen to cover: no history, Black to
# move with more than 8 plies of history, castling both ways, under-promotions
# with and without capture for either colour, and a sharp tactical position.
POSITIONS = {
    "start": (chess.STARTING_FEN, []),
    "ruy_black": (chess.STARTING_FEN,
                  ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"]),
    "castle_both": ("r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 8", []),
    "promo_white": ("r3k3/1P6/8/8/8/8/6K1/8 w q - 0 1", []),
    "promo_black": ("4k3/8/8/8/8/8/p5K1/1R6 b - - 0 1", []),
    "opera": ("4kb1r/p2n1ppp/4q3/4p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 0 16", []),
}
HEAVY = "opera"          # the one position that also gets the full head sweep


def board_for(name: str) -> chess.Board:
    fen, moves = POSITIONS[name]
    b = chess.Board(fen)
    for m in moves:
        b.push_uci(m)
    return b


# ----- normalization / encoding -------------------------------------------------

def _numeric_leaves(x) -> bool:
    if isinstance(x, (list, tuple)):
        return all(_numeric_leaves(v) for v in x)
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def normalize(x):
    """Tensors -> numpy, chess.Move -> uci, numeric nested lists -> arrays,
    tuples -> lists, numpy scalars -> Python scalars."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, chess.Move):
        return x.uci()
    if isinstance(x, dict):
        return {str(k): normalize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        if x and isinstance(x[0], (list, tuple)) and _numeric_leaves(x):
            try:
                return np.asarray(x, dtype=np.float64)
            except ValueError:                       # ragged
                pass
        if x and all(isinstance(v, float) for v in x):
            return np.asarray(x, dtype=np.float64)
        return [normalize(v) for v in x]
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _canonical(a: np.ndarray) -> np.ndarray:
    """float64 arrays whose every value is a float32 are stored as float32 (the
    values came from float32 tensors via .tolist()), which halves the bytes."""
    if a.dtype == np.float64:
        a32 = a.astype(np.float32)
        if np.array_equal(a32.astype(np.float64), a, equal_nan=True):
            return a32
    return a


def _sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _stats(a: np.ndarray) -> dict:
    f = a.astype(np.float64).ravel()
    finite = f[np.isfinite(f)]
    step = max(1, f.size // 32)
    return {"sum": float(finite.sum()) if finite.size else 0.0,
            "abs_sum": float(np.abs(finite).sum()) if finite.size else 0.0,
            "n_nonfinite": int(f.size - finite.size),
            "sample": [float(v) for v in f[::step][:32]]}


def encode(x, path="", arrays=None):
    """Normalized value -> JSON-able tree; small arrays go into `arrays`."""
    if arrays is None:
        arrays = {}
    if isinstance(x, np.ndarray):
        a = _canonical(x)
        if a.size <= SMALL:
            arrays[path] = a
            return {"__arr__": path}, arrays
        return {"__hash__": _sha(a), "shape": list(a.shape), "dtype": str(a.dtype),
                "stats": _stats(a)}, arrays
    if isinstance(x, dict):
        return {k: encode(v, f"{path}/{k}", arrays)[0] for k, v in x.items()}, arrays
    if isinstance(x, list):
        return [encode(v, f"{path}/{i}", arrays)[0] for i, v in enumerate(x)], arrays
    if isinstance(x, str) and len(x) > LONG_STR:
        return {"__strhash__": hashlib.sha256(x.encode()).hexdigest(), "len": len(x)}, arrays
    if isinstance(x, float) and not np.isfinite(x):
        return {"__float__": repr(x)}, arrays
    return x, arrays


# ----- comparison -----------------------------------------------------------------

def _close(a, b, rtol, atol=1e-6) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
                            rtol=rtol, atol=atol, equal_nan=True))


def diff(gold, garr, new, narr, path="", rtol=None, out=None):
    """Mismatches between a golden tree and a freshly encoded one, golden-subset:
    keys only `new` has are fine. `rtol=None` demands bit equality."""
    if out is None:
        out = []
    if isinstance(gold, dict) and "__arr__" in gold:
        if not (isinstance(new, dict) and "__arr__" in new):
            out.append(f"{path}: expected a small array, got {str(new)[:120]}")
            return out
        a, b = garr[gold["__arr__"]], narr[new["__arr__"]]
        if a.shape != b.shape:
            out.append(f"{path}: shape {a.shape} -> {b.shape}")
        elif rtol is None and not (a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)):
            d = np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))) if a.size else 0
            out.append(f"{path}: arrays differ (max abs diff {d:.3e}, dtypes {a.dtype}/{b.dtype})")
        elif rtol is not None and not _close(a, b, rtol):
            out.append(f"{path}: arrays differ beyond rtol={rtol}")
        return out
    if isinstance(gold, dict) and "__hash__" in gold:
        if not (isinstance(new, dict) and "__hash__" in new):
            out.append(f"{path}: expected a large array, got {str(new)[:120]}")
            return out
        if gold["shape"] != new["shape"]:
            out.append(f"{path}: shape {gold['shape']} -> {new['shape']}")
        elif rtol is None and gold["__hash__"] != new["__hash__"]:
            gs, ns = gold["stats"], new["stats"]
            out.append(f"{path}: hash differs (sum {gs['sum']:.9g} -> {ns['sum']:.9g}, "
                       f"abs_sum {gs['abs_sum']:.9g} -> {ns['abs_sum']:.9g})")
        elif rtol is not None and gold["__hash__"] != new["__hash__"]:
            gs, ns = gold["stats"], new["stats"]
            if not (_close(gs["sample"], ns["sample"], rtol) and _close(gs["abs_sum"], ns["abs_sum"], rtol)
                    and gs["n_nonfinite"] == ns["n_nonfinite"]):
                out.append(f"{path}: large array differs beyond rtol={rtol}")
        return out
    if isinstance(gold, dict):
        if not isinstance(new, dict):
            out.append(f"{path}: expected dict, got {type(new).__name__}")
            return out
        for k, v in gold.items():
            if k not in new:
                out.append(f"{path}/{k}: missing")
            else:
                diff(v, garr, new[k], narr, f"{path}/{k}", rtol, out)
        return out
    if isinstance(gold, list):
        if not isinstance(new, list) or len(new) != len(gold):
            out.append(f"{path}: list {len(gold)} -> {len(new) if isinstance(new, list) else type(new).__name__}")
            return out
        for i, (g, n) in enumerate(zip(gold, new)):
            diff(g, garr, n, narr, f"{path}/{i}", rtol, out)
        return out
    if isinstance(gold, float) and isinstance(new, (int, float)) and not isinstance(new, bool):
        if (rtol is None and gold != new) or (rtol is not None and not _close(gold, new, rtol)):
            out.append(f"{path}: {gold!r} -> {new!r}")
        return out
    if gold != new:
        out.append(f"{path}: {str(gold)[:80]!r} -> {str(new)[:80]!r}")
    return out


# ----- the battery ------------------------------------------------------------------

def _keys(d: dict, *names):
    return {k: d[k] for k in names if k in d}


def engine_static(eng) -> dict:
    cfg = eng.cfg
    return {
        "cfg": {k: getattr(cfg, k) for k in ("num_blocks", "num_heads", "dim_vit",
                                             "mlp_ratio", "gab_gen_size", "history")},
        "display_name": eng.spec.display_name,
        "flags": {"has_conditioning": eng.has_conditioning,
                  "has_generated_bias": eng.has_generated_bias, "has_mlh": eng.has_mlh},
        "all_moves": list(eng.all_moves),
        "hook_points": {k: True for k in eng.hook_points},
        "depth_points": eng.depth_points(),
        "gab_templates": eng.gab_templates(),
    }


def engine_position(eng, board: chess.Board, *, heavy: bool) -> dict:
    """Everything an engine says about one position."""
    nb, nh = eng.cfg.num_blocks, eng.cfg.num_heads
    Lm = nb // 2
    out = {}

    res = eng.evaluate(board, ELO, OPPO)
    out["evaluate"] = _keys(res, "policy", "wdl", "mlh", "_logits")
    out["evaluate_self_only"] = _keys(eng.evaluate(board, ELO), "_logits", "wdl")
    out["tokens"] = eng.tokens(board)

    _, cache = eng.run_with_cache(board, ELO, OPPO)
    out["cache"] = cache
    out["logit_lens"] = {k: eng.logit_lens(v, board) for k, v in cache.items()}
    last = list(cache)[-1]
    out["logit_lens_raw"] = eng.logit_lens(cache[last])
    out["logit_lens_unmasked"] = eng.logit_lens(cache["embed_in"], board, legal_only=False)

    for L in (0, Lm, nb - 1):
        for h in (0, nh - 1):
            out[f"attention_L{L}_H{h}"] = eng.attention(board, ELO, OPPO, layer=L, head=h)
    out["gab_coeffs"] = eng.gab_coeffs(board, ELO, OPPO, layer=Lm)
    out["gab_bias_head"] = eng.gab_bias(board, ELO, OPPO, layer=Lm, head=1)
    out["gab_bias_all"] = eng.gab_bias(board, ELO, OPPO, layer=Lm)
    out["qk_head"] = eng.qk_scores(board, ELO, OPPO, layer=Lm, head=1)
    out["qk_all"] = eng.qk_scores(board, ELO, OPPO, layer=Lm)
    out["head_writes"] = eng.head_writes(board, ELO, OPPO, layer=Lm)

    out["ablate_head"] = _keys(eng.ablate_head(board, ELO, Lm, 2, OPPO), "policy", "wdl", "mlh", "_logits")
    out["ablate_head_logits"] = list(eng.ablate_head(board, ELO, Lm, 2, OPPO, return_type="logits"))
    out["run_with_hooks"] = list(eng.run_with_hooks(
        board, ELO, OPPO, fwd_hooks=[(f"attn_{Lm:02d}", lambda a: a * 0.5)], return_type="logits"))

    out["residual_stream"] = eng.residual_stream(board, ELO, OPPO)
    out["compare_residual"] = eng.compare_residual(board, 1100, 1900)

    top = res["policy"][0][0] if res["policy"] else None
    if top is not None:
        out["move_logit_lens"] = eng.move_logit_lens(board, ELO, top, OPPO)
        out["logit_per_depth"] = eng.logit_per_depth(board, ELO, top, OPPO)
        out["policy_per_depth"] = eng.policy_per_depth(board, ELO, top, OPPO)
        out["rank_per_depth"] = eng.rank_per_depth(board, ELO, top, OPPO)
        out["carrier_neurons"] = eng.carrier_neurons(board, ELO, top, OPPO, top_k=4, verify=True)
        out["carrier_neurons_est"] = eng.carrier_neurons(board, ELO, top, OPPO, top_k=4,
                                                         skip_last_layer=False)
        if heavy:
            out["ablate_grid"] = eng.ablate_grid(board, ELO, top, OPPO)

    out["neuron_activation"] = eng.neuron_activation(board, ELO, Lm, 7, OPPO)
    out["neuron_overview"] = eng.neuron_overview(board, ELO, OPPO, k=5)
    out["ablate_neuron"] = _keys(eng.ablate_neuron(board, ELO, Lm, 7, OPPO), "policy", "wdl", "_logits")

    torch.manual_seed(0)
    out["select_t1"] = eng.select_move(board, ELO, OPPO, temperature=1.0)[0]
    torch.manual_seed(1)
    out["select_t07_p09"] = eng.select_move(board, ELO, OPPO, temperature=0.7, top_p=0.9)[0]
    out["select_t0"] = eng.select_move(board, ELO, OPPO, temperature=0)[0]

    forms = {}
    for mv in board.legal_moves:
        info = eng.move_info(board, mv)
        forms[mv.uci()] = {**info,
                           "via_idx": eng.to_move(board, info["idx"]),
                           "via_squares": eng.to_move(board, (info["from"], info["to"])),
                           "via_san": eng.to_move(board, info["san"])}
    out["move_forms"] = forms
    return out


def engine_batch(eng) -> dict:
    """ablate_grid_batch across two positions in one batch."""
    items = []
    for name in ("castle_both", "promo_black"):
        b = board_for(name)
        items.append((b, eng.evaluate(b, ELO, OPPO)["policy"][0][0]))
    return {"ablate_grid_batch": eng.ablate_grid_batch(items, ELO, OPPO, batch_size=2)}


def _bridge(eng):
    from chessformer_lens import bridge
    Api = getattr(bridge, "LensApi", None) or getattr(bridge, "MaiaApi")
    api = Api.__new__(Api)                  # skip __init__'s loader thread
    api.engine, api.ready, api.error = eng, True, None
    api.alias = api.target_name = "golden"
    api.board, api.human, api.human_both, api.san_history = chess.Board(), chess.WHITE, True, []
    api._lock = threading.Lock()
    api._loading = False
    return api


def _neutral(x):
    """Bridge payloads under the neutral names: every maia_* key -> engine_*."""
    if isinstance(x, dict):
        return {(("engine_" + k[5:]) if k.startswith("maia_") else k): _neutral(v)
                for k, v in x.items()}
    if isinstance(x, list):
        return [_neutral(v) for v in x]
    return x


def _rows(d: dict) -> dict:
    """The bridge's comparison rows come out of a set union, so their order
    among equal |Δp| is hash-dependent: key them by uci."""
    if isinstance(d, dict) and "rows" in d:
        d = {**d, "rows": {r["uci"]: r for r in d["rows"]}}
    return d


def frontend(eng, *, heavy: bool) -> dict:
    from chessformer_lens import interp_widget as iw

    api = _bridge(eng)
    nb = eng.cfg.num_blocks
    Lm = nb // 2
    out = {"info": {k: v for k, v in api.info().items() if k != "checkpoint"},
           "gab_templates": api.gab_templates()}
    for name in ("ruy_black", "promo_white"):
        api.board = board_for(name)
        top = eng.evaluate(api.board, ELO)["policy"][0][0]
        p = {
            "policy": api.policy(ELO, save=False),
            "attention": api.attention(ELO, Lm, 1),
            "gab_coeffs": api.gab_coeffs(ELO, Lm),
            "gab_bias": api.gab_bias(ELO, Lm, 1),
            "qk_scores": api.qk_scores(ELO, Lm, 1),
            "residual": api.residual(ELO),
            "compare_residual": api.compare_residual(ELO, 1100),
            "move_lens": api.move_lens(ELO, top),
            "neuron": api.neuron(ELO, Lm, 7),
            "neurons_overview": api.neurons_overview(ELO, 5),
            "ablate_neuron": _rows(api.ablate_neuron(ELO, Lm, 7)),
            "carrier_neurons": api.carrier_neurons(ELO, top, 4),
            "compare_policy": _rows(api.compare_policy(ELO, 1100)),
            "ablate": _rows(api.ablate(ELO, Lm, 2)),
            "attention_payload": iw.attention_payload(eng, api.board, ELO),
            "gab_payload": iw.gab_payload(eng, api.board, ELO),
        }
        if heavy and name == "promo_white":
            p["ablate_grid"] = api.ablate_grid(ELO, top)
        out[name] = p

    api.board, api.human, api.human_both, api.san_history = chess.Board(), chess.BLACK, False, []
    torch.manual_seed(0)
    out["engine_move"] = (getattr(api, "engine_move", None) or api.maia_move)(ELO, 1.0)
    if heavy:
        random.seed(0)
        torch.manual_seed(0)
        out["random_position"] = api.random_position(ELO, 4, 8)
    return _neutral(out)


def run(eng, *, heavy: bool = True) -> dict:
    """The whole battery for one engine, normalized."""
    out = {"static": engine_static(eng)}
    for name in POSITIONS:
        out[f"pos_{name}"] = engine_position(eng, board_for(name), heavy=heavy and name == HEAVY)
    if heavy:
        out.update(engine_batch(eng))
    out["frontend"] = frontend(eng, heavy=heavy)
    return normalize(out)


# ----- engines --------------------------------------------------------------------------

def bt4_weights() -> Path | None:
    import os
    for d in (os.environ.get("CHESSFORMER_WEIGHTS"), REPO / "weights"):
        if not d:
            continue
        for sfx in (".pb.gz", ".onnx"):
            p = Path(d).expanduser() / f"Leela_BT4_large_model{sfx}"
            if p.exists():
                return p
    return None


def make_engine(name: str, activation_dir):
    from chessformer_lens.engine import LeelaEngine, MaiaEngine
    if name == "maia3-5m":
        return MaiaEngine("maia3-5m", device="cpu", activation_dir=activation_dir)
    if name == "bt4":
        return LeelaEngine("bt4", device="cpu", checkpoint_path=bt4_weights(),
                           activation_dir=activation_dir)
    raise ValueError(name)


def save(name: str, result: dict, meta: dict):
    tree, arrays = encode(result)
    GOLDEN_DIR.mkdir(exist_ok=True)
    with gzip.open(GOLDEN_DIR / f"{name}.json.gz", "wt") as fh:
        json.dump({"meta": meta, "tree": tree}, fh)
    np.savez_compressed(GOLDEN_DIR / f"{name}.npz", **{k.lstrip("/"): v for k, v in arrays.items()})


def load(name: str):
    with gzip.open(GOLDEN_DIR / f"{name}.json.gz", "rt") as fh:
        doc = json.load(fh)
    with np.load(GOLDEN_DIR / f"{name}.npz") as z:
        arrays = {"/" + k: z[k] for k in z.files}
    return doc["meta"], doc["tree"], arrays
