# Third-party notices

The source code in this repository is MIT licensed (see [LICENSE](LICENSE)).
This file records the third-party work it relies on, because MIT on this repo
does not and cannot change the terms of anyone else's code.

## Bundled in this repository

**Chess piece artwork** — `piece_art/*.png`

The cburnett piece set by Colin M.L. Burnett, obtained via python-chess's
`chess.svg.PIECES` and rasterised by `piece_art/regenerate.py`. The artwork is
triple-licensed by its author under GFDL / BSD / GPL; this project uses it
under the **BSD** option, which is compatible with MIT. Attribution to Colin
M.L. Burnett is required and is given here and in `pieces.py`.

Note that `pieces.py` itself contains no artwork — it reads `chess.svg.PIECES`
at import time and wraps each value in an `<svg>` element. Only the rasterised
PNGs under `piece_art/` are redistributed here.

## Installed separately, not contained in this repository

These are fetched by `pip` at install time. No code from either package is
copied into this repository.

| package | license | why it is required |
|---|---|---|
| [python-chess](https://github.com/niklasf/python-chess) | GPL-3.0-or-later | board representation, legal move generation, SAN/UCI parsing |
| [maia3](https://github.com/CSSLab/maia3) | AGPL-3.0 | the Maia-3 model definition, checkpoint loader and tokenizer |

Both are strong-copyleft licenses. What that means in practice:

- **Using this project** — installing it, running it, doing research with it,
  modifying it for your own use — is unaffected. Copyleft obligations attach to
  *distribution*, not to use.
- **Redistributing this repository's source** is covered by MIT. The source
  here is original work; it calls these libraries through their public APIs
  rather than incorporating them.
- **Redistributing a bundle** that ships this project *together with*
  python-chess or maia3 — a Docker image, a conda package, a PyInstaller or
  py2app binary, a vendored copy — makes that bundle a combined work, and the
  bundle as a whole must then satisfy GPL-3.0 (and AGPL-3.0, including its
  network-use clause, if maia3 is included). MIT on this repository does not
  exempt such a bundle.

If you intend to ship a combined artifact, or to build a hosted service on top
of maia3, get the licensing reviewed rather than relying on this summary.

## Model weights

Maia-3 checkpoints are downloaded from
<https://huggingface.co/UofTCSSLab> at runtime and are **not** covered by this
repository's license. Check the terms on the relevant Hugging Face model card
before redistributing weights.
