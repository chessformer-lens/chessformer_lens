"""Built-in model adapters (see core.ModelAdapter). Importing this package
registers them; none imports its model's own package until a model is loaded.

Resolution order matters where aliases could overlap: lc0 first (explicit
aliases and weight-file suffixes), then Maia-3 (whose resolver also accepts
Hugging Face repo ids)."""
from . import lc0, maia3  # noqa: F401
