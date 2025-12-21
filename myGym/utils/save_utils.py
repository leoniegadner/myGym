"""Small helpers for save/load path handling."""

SB3_STYLE_ALGOS = {"ppo", "multippo", "mppo", "pets", "mbpo", "planet", "aif", "meta_aif"}


def uses_sb3_style(algo: str) -> bool:
    """Return True if the algo should use the SB3-style save structure."""
    return algo in SB3_STYLE_ALGOS
