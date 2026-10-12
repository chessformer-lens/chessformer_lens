"""
SAE.py — a sparse autoencoder on Maia-3 5M's residual stream.

The plan
    1. collect     residual-stream vectors at one layer, one per square token
    2. train       an Anthropic-style SAE: x ≈ W_dec · ReLU(W_enc (x − b_dec) + b_enc) + b_dec
    3. check       reconstruction, sparsity, and whether the model still plays the
                   same moves with the SAE spliced in
    4. grade       each feature against exact python-chess concepts
    5. look        draw where one feature fires, on the positions it likes most

Run a step at a time:
    python SAE.py collect      # -> sae_data/acts_L04.pt
    python SAE.py train        # -> sae_data/sae_L04.pt
    python SAE.py check
    python SAE.py grade
    python SAE.py look 123     # -> sae_data/feature_0123.png

Facts (checked against this repo):
  - Maia-3 5M: 256-d residual stream, 8 layers ("block_00" … "block_07").
  - `eng.run_with_cache(board, elo)` returns (out, cache); `cache["block_04"]` is
    the residual stream AFTER layer 4, shape (64, 256), on CPU.
  - The 64 rows are squares in the model's frame: side to move at the bottom, Black's
    positions mirrored vertically. `MaiaEngine._canon_square(square, board.turn)`
    maps a python-chess square to its row.
"""
import sys
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chessformer_lens.engine import MaiaEngine
from chessformer_lens import research

PGN = Path.home() / "Desktop/Code/maia_research/lichess_pgns.pgn"
OUT = Path("sae_data")            # gitignore this: activations are big
LAYER = 4                         # a middle layer to start; try others later
SITE = f"block_{LAYER:02d}"       # residual stream after that layer
ELO = 1500                        # Maia's stream depends on the rating: fix one first
N_POSITIONS = 20_000              # 20k x 64 squares = 1.28M vectors
HELD_OUT = 0.1                    # the last 10% of positions are never trained on
D_MODEL = 256
N_FEATURES = 4096                 # 16x expansion
L1_COEFF = 5e-3                   # the knob you will tune most
LR = 1e-3
BATCH = 4096                      # vectors per step
EPOCHS = 5
SEED = 0


def _acts_path():
    return OUT / f"acts_L{LAYER:02d}.pt"


def _sae_path():
    return OUT / f"sae_L{LAYER:02d}.pt"


def _engine():
    return MaiaEngine("maia3-5m", device="cpu")


def _boards(games):
    """Rebuild boards, history included, from saved (root FEN, [uci]) pairs."""
    out = []
    for fen, moves in games:
        b = chess.Board(fen)
        for u in moves:
            b.push_uci(u)
        out.append(b)
    return out


def _split(n_positions):
    """Position indices for training and held out. Split by POSITION, not by
    vector: the 64 vectors of one position are strongly correlated."""
    n_test = max(1, int(n_positions * HELD_OUT))
    return range(0, n_positions - n_test), range(n_positions - n_test, n_positions)


# --------------------------------------------------------------------------------
# 1. collect
# --------------------------------------------------------------------------------
def collect():
    """Residual-stream vectors for N_POSITIONS positions, saved as
    {acts: (N*64, 256) float16, games: [(root FEN, [uci])], site, elo}.
    Row i*64 + c is square c (model frame) of position i."""
    eng = _engine()
    # one position per game, so read enough games: some are too short and skipped
    boards = research.positions_from_pgn(PGN, N_POSITIONS, seed=SEED,
                                         max_games=3 * N_POSITIONS)
    n = len(boards)
    acts = torch.empty(n, 64, D_MODEL, dtype=torch.float16)
    for i, board in enumerate(boards):
        _, cache = eng.run_with_cache(board, ELO)
        acts[i] = cache[SITE].half()
        if (i + 1) % 100 == 0:
            print(f"  {i + 1:,} / {n:,}", flush=True)
    games = [(b.root().fen(), [m.uci() for m in b.move_stack]) for b in boards]
    torch.save({"acts": acts.reshape(n * 64, D_MODEL), "games": games,
                "site": SITE, "elo": ELO}, _acts_path())
    print(f"saved {n:,} positions = {n * 64:,} vectors to {_acts_path()}")


# --------------------------------------------------------------------------------
# 2. the SAE
# --------------------------------------------------------------------------------
class SAE(nn.Module):
    """f = ReLU((x − b_dec) W_enc + b_enc);  x_hat = f W_dec + b_dec.

    x is (batch, d_model) rows, so the weights multiply on the right:
    W_enc (d_model, n_features), W_dec (n_features, d_model)."""

    def __init__(self, d_model=D_MODEL, n_features=N_FEATURES):
        super().__init__()
        w_dec = torch.randn(n_features, d_model)
        w_dec /= w_dec.norm(dim=1, keepdim=True)           # unit-norm feature directions
        self.W_dec = nn.Parameter(w_dec)
        self.W_enc = nn.Parameter(w_dec.t().clone())       # start as the decoder's transpose
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_model))    # set to the data mean in train()

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def normalise_decoder(self):
        """Keep every decoder row at unit norm. Without it the L1 penalty is
        cheated: shrink f, grow W_dec, same reconstruction, smaller loss."""
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)


def loss_fn(x, x_hat, f):
    """(total, mse, l1). mse: squared error summed over the 256 dims, averaged
    over the batch; l1: |f| summed over features, averaged over the batch."""
    mse = (x - x_hat).pow(2).sum(dim=-1).mean()
    l1 = f.abs().sum(dim=-1).mean()
    return mse + L1_COEFF * l1, mse, l1


def fve(x, x_hat):
    """Fraction of variance explained: 1 − residual variance / total variance."""
    return float(1 - (x - x_hat).pow(2).sum() / (x - x.mean(0)).pow(2).sum())


# --------------------------------------------------------------------------------
# 2b. train
# --------------------------------------------------------------------------------
def train():
    data = torch.load(_acts_path())
    acts = data["acts"].float()
    n_pos = len(data["games"])
    train_pos, _ = _split(n_pos)
    x_all = acts[: len(train_pos) * 64]                     # training positions only
    # one scale for the whole dataset: mean vector norm becomes sqrt(d_model), so
    # L1_COEFF means the same thing at every layer and every model
    scale = (D_MODEL ** .5) / x_all.norm(dim=1).mean()
    x_all = x_all * scale

    torch.manual_seed(SEED)
    sae = SAE()
    with torch.no_grad():
        sae.b_dec.copy_(x_all.mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=LR)

    seen_since = torch.zeros(N_FEATURES)                   # vectors since each feature last fired
    step = 0
    for epoch in range(EPOCHS):
        order = torch.randperm(len(x_all))                 # shuffle vectors every epoch
        for i in range(0, len(x_all), BATCH):
            x = x_all[order[i:i + BATCH]]
            x_hat, f = sae(x)
            loss, mse, l1 = loss_fn(x, x_hat, f)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalise_decoder()

            fired = (f > 0).any(dim=0)
            seen_since += len(x)
            seen_since[fired] = 0
            step += 1
            if step % 200 == 0:
                l0 = (f > 0).float().sum(dim=1).mean()
                dead = (seen_since > 1_000_000).float().mean()
                print(f"epoch {epoch}  step {step:5d}  mse {mse:.3f}  l1 {l1:.1f}  "
                      f"L0 {l0:.1f}  fve {fve(x, x_hat.detach()):.3f}  dead {dead:.1%}",
                      flush=True)
    torch.save({"state": sae.state_dict(), "scale": float(scale), "site": SITE,
                "elo": data["elo"], "l1_coeff": L1_COEFF}, _sae_path())
    print(f"saved {_sae_path()}")


def _load_sae():
    saved = torch.load(_sae_path())
    sae = SAE()
    sae.load_state_dict(saved["state"])
    sae.eval()
    return sae, saved["scale"]


def _held_out():
    """(boards, their activations (n*64, 256), scaled) for the held-out positions."""
    data = torch.load(_acts_path())
    _, test_pos = _split(len(data["games"]))
    boards = _boards([data["games"][i] for i in test_pos])
    acts = data["acts"][test_pos.start * 64: test_pos.stop * 64].float()
    return boards, acts


# --------------------------------------------------------------------------------
# 3. check
# --------------------------------------------------------------------------------
@torch.no_grad()
def check(n_splice=200):
    sae, scale = _load_sae()
    boards, acts = _held_out()
    x = acts * scale
    x_hat, f = sae(x)
    l0 = (f > 0).float().sum(dim=1).mean()
    alive = (f > 0).any(dim=0).float().mean()
    print(f"held out: {len(boards):,} positions · fve {fve(x, x_hat):.3f} · "
          f"L0 {l0:.1f} · features that ever fire {alive:.1%}")

    # splice: does the model still play the same moves on the reconstruction?
    eng = _engine()
    mean = acts.mean(0)

    def run(board, replace):
        def hook(act):                                     # act: (1, 64, 256), live device
            return replace(act.reshape(-1, D_MODEL)).reshape(act.shape).to(act.dtype)
        return dict(eng.run_with_hooks(board, ELO, fwd_hooks=[(SITE, hook)])["policy"])

    def through_sae(a):
        return sae(a.float().cpu() * scale)[0] / scale

    def mean_ablate(a):                                    # a "how bad could it be" baseline
        return mean.expand(a.shape[0], -1)

    def kl(p, q):
        return sum(pv * torch.log(torch.tensor(pv / max(q.get(m, 0.0), 1e-12)))
                   for m, pv in p.items() if pv > 0)

    for name, replace in (("SAE", through_sae), ("mean-ablated", mean_ablate)):
        same, kls = 0, []
        for board in boards[:n_splice]:
            clean = dict(eng.evaluate(board, ELO)["policy"])
            spliced = run(board, replace)
            same += max(clean, key=clean.get) == max(spliced, key=spliced.get)
            kls.append(float(kl(clean, spliced)))
        print(f"{name:13s} top move kept {same / n_splice:.1%} · "
              f"mean KL {sum(kls) / len(kls):.3f}")


# --------------------------------------------------------------------------------
# 4. grade features against chess concepts
# --------------------------------------------------------------------------------
PIECES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
          chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}


def concepts(board: chess.Board) -> dict:
    """Per-square boolean labels for one position, in the MODEL'S frame
    (index = MaiaEngine._canon_square(square, board.turn))."""
    us, them = board.turn, not board.turn
    labels = {}

    def mark(name, squares):
        t = torch.zeros(64, dtype=torch.bool)
        for sq in squares:
            t[MaiaEngine._canon_square(sq, us)] = True
        labels[name] = t

    ours = board.occupied_co[us]
    theirs = board.occupied_co[them]
    mark("own_piece", chess.SquareSet(ours))
    mark("their_piece", chess.SquareSet(theirs))
    mark("empty", chess.SquareSet(~board.occupied & chess.BB_ALL))
    for pt, name in PIECES.items():
        mark(f"own_{name}", board.pieces(pt, us))
        mark(f"their_{name}", board.pieces(pt, them))
    mark("attacked_by_them", [sq for sq in chess.SQUARES if board.is_attacked_by(them, sq)])
    mark("defended_by_us", [sq for sq in chess.SQUARES if board.is_attacked_by(us, sq)])
    mark("pinned", [sq for sq in chess.SquareSet(ours) if board.is_pinned(us, sq)])
    mark("hanging", [sq for sq in chess.SquareSet(ours) if board.piece_type_at(sq) != chess.KING
                     and board.is_attacked_by(them, sq) and not board.is_attacked_by(us, sq)])
    king = board.king(them)
    mark("their_king_zone", chess.SquareSet(chess.BB_KING_ATTACKS[king]) if king is not None else [])
    mark("legal_destination", {m.to_square for m in board.legal_moves})
    return labels


@torch.no_grad()
def grade(n_positions=500, min_fires=20):
    """Score every feature against every concept by F1 of (feature > 0) vs the
    concept, over the squares of `n_positions` held-out positions."""
    sae, scale = _load_sae()
    boards, acts = _held_out()
    boards, acts = boards[:n_positions], acts[: n_positions * 64]
    fires = (sae.encode(acts * scale) > 0).float()           # (V, F)
    per_board = [concepts(b) for b in boards]
    names = list(per_board[0])
    labels = torch.stack([torch.cat([c[n] for c in per_board]) for n in names], dim=1).float()

    tp = fires.t() @ labels                                  # (F, C) both true
    n_fire = fires.sum(0)[:, None]                           # (F, 1)
    n_true = labels.sum(0)[None, :]                          # (1, C)
    precision = tp / n_fire.clamp(min=1)
    recall = tp / n_true.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    best_f1, best_c = f1.max(dim=1)
    active = n_fire.squeeze(1) >= min_fires

    print(f"{len(boards)} positions · {int(active.sum())} features fire ≥ {min_fires} times")
    print("\nbest-matching features")
    order = torch.where(active, best_f1, torch.tensor(-1.0)).argsort(descending=True)
    for j in order[:20].tolist():
        c = int(best_c[j])
        print(f"  f{j:<5} {names[c]:18s} F1 {best_f1[j]:.2f}  precision {precision[j, c]:.2f}  "
              f"recall {recall[j, c]:.2f}  fires {int(n_fire[j])}")
    print("\nfire often, match nothing (the interesting unknowns)")
    unknown = active & (best_f1 < .3)
    for j in n_fire.squeeze(1).where(unknown, torch.tensor(0.)).argsort(descending=True)[:10].tolist():
        print(f"  f{j:<5} fires {int(n_fire[j])}  best {names[int(best_c[j])]} F1 {best_f1[j]:.2f}")
    print("\nconcept coverage: the feature that tracks each concept best")
    for c, name in enumerate(names):
        j = int(torch.where(active, f1[:, c], torch.tensor(-1.0)).argmax())
        print(f"  {name:18s} f{j:<5} F1 {f1[j, c]:.2f}")


# --------------------------------------------------------------------------------
# 5. look
# --------------------------------------------------------------------------------
@torch.no_grad()
def look(feature, k=4):
    """Where feature `feature` fires on the k held-out positions it likes most,
    drawn as interp_plot draws every board."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import chessformer_lens.interp_plot as ip

    sae, scale = _load_sae()
    boards, acts = _held_out()
    f = sae.encode(acts * scale)[:, int(feature)].reshape(len(boards), 64)
    top = f.max(dim=1).values.argsort(descending=True)[:k].tolist()
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.3))
    m = float(f[top].max()) or 1e-9
    for ax, i in zip(axes, top):
        ip.draw_board(ax, boards[i], heat=(f[i] / m).tolist(), cmap=ip._divmap)
        ax.set_title(f"{'White' if boards[i].turn else 'Black'} to move · {float(f[i].max()):.2f}",
                     fontsize=9, family=ip.MONO)
    fig.suptitle(f"feature {feature} · {SITE} · its strongest held-out positions", fontsize=10)
    path = OUT / f"feature_{int(feature):04d}.png"
    ip.save(fig, path, dpi=120)
    print(f"saved {path}")


if __name__ == "__main__":
    collect()
    train()
    check()
    grade()
    #look(2912)
