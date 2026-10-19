"""Built-in model adapters (see core.ModelAdapter). Importing this package
registers them; none imports its model's own package until a model is loaded.

Resolution order matters where aliases could overlap: lc0 first (explicit
aliases and weight-file suffixes), then searchless (a fixed alias table), then
Maia-3 (whose resolver also accepts Hugging Face repo ids and claims anything
that looks like "<n>m")."""
from . import lc0, searchless, maia3  # noqa: F401
