# ==========================================
# File: core/palette.py
# ==========================================

# Validated 8-slot categorical palette (dataviz skill, references/palette.md).
# Fixed order, checked as a set for all-pairs CVD separation — do not reassign
# individual hex values or cycle beyond 8 slots.
CATEGORICAL = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

# Anatomically relevant parts each get a fixed categorical slot. Left/right
# pairs share a color: 3D position already disambiguates them.
PART_COLORS = {
    "skin": CATEGORICAL[0],
    "nose": CATEGORICAL[1],
    "l_eye": CATEGORICAL[2], "r_eye": CATEGORICAL[2],
    "l_brow": CATEGORICAL[3], "r_brow": CATEGORICAL[3],
    "l_ear": CATEGORICAL[4], "r_ear": CATEGORICAL[4],
    "mouth": CATEGORICAL[5],
    "u_lip": CATEGORICAL[6],
    "l_lip": CATEGORICAL[7],
}

# Everything else (background, hair, hat, clothing, eyewear...) recedes into
# a single muted context color instead of competing for a categorical slot.
CONTEXT_COLOR = "#898781"

# Reserved accent for highlighted reference points (landmarks), deliberately
# distinct from the categorical slots so it never impersonates a measured part.
LANDMARK_ACCENT = "#d03b3b"

# Generic accent for previews that aren't tied to one specific part (e.g. the
# full back-projected point cloud, or all 68 landmarks at once) — kept out of
# PART_COLORS so it never gets mistaken for a specific part's identity color.
GENERIC_ACCENT = "#256abf"

INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "gridline": "#e1e0d9",
    "surface": "#fcfcfb",
}
