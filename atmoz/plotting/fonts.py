"""Font registration.

atmoz ships Inter (``atmoz/assets/fonts``, SIL Open Font License; the license file sits next to the fonts). Styles ask
for ``["Inter", "DejaVu Sans"]``; DejaVu Sans ships with matplotlib, so text always renders. Any ``.ttf`` / ``.otf``
dropped into the folder is registered with matplotlib's font manager the first time a plot is made. This adds the
files to matplotlib's list of available fonts; it does not change any rcParams.
"""

from __future__ import annotations

from pathlib import Path

_done = False


def font_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "fonts"


def register_fonts() -> list[str]:
    """Register bundled fonts once; return the file names registered (empty if none are bundled)."""
    global _done
    if _done:
        return []
    _done = True
    folder = font_dir()
    if not folder.is_dir():
        return []
    from matplotlib import font_manager

    added = []
    for path in sorted([*folder.glob("*.ttf"), *folder.glob("*.otf")]):
        try:
            font_manager.fontManager.addfont(str(path))
            added.append(path.name)
        except Exception:  # a broken font file must never break a plot
            continue
    return added


def available_families(stack) -> list[str]:
    """The families in ``stack`` that matplotlib can actually find, in order, always ending in DejaVu Sans.

    Giving matplotlib only installed families avoids "font family not found" warnings.
    """
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    families = [name for name in stack if name in installed]
    if "DejaVu Sans" not in families:
        families.append("DejaVu Sans")
    return families
