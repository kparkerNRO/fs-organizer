CREATOR_REMOVES = {
    "CzePeku": ["$5 Rewards", "$10 Rewards", "$1 Rewards"],
    "Limithron": ["Admiral", "Captain"],
    "The Reclusive Cartographer": "_MC",
    "Baileywiki": "",
    "Unknown": "",
    "Caeora": "",
    "DWW": "",
    "MAD Cartographer": "",
    "MikWewa": "$5 Map Rewards",
    "Tom Cartos": ["Tier 2+", "tomcartos"],
    "Borough Bound": "Chief Courier Rewards",
}

FILE_NAME_EXCEPTIONS = {"The Clean": "Clean", "The": ""}
REPLACE_EXCEPTIONS = {
    "ItW": "",
}
CLEAN_EXCEPTIONS = {"5e"}

GROUPING_EXCEPTIONS = (
    "City of",
    "Lair of the",
    "Tower of",
    "The ruins of",
    "The",
    "Bone",
    "Wizard",
    "War",
    "In",
)

# Example known variant sets
# In reality, you might load these from config or define them elsewhere.
season_tokens = {
    "winter",
    "summer",
    "spring",
    "fall",
    "autumn",
    "snow",
    "desert",
    "grass",
    "sandstorm",
    "rain",
    "mountain",
}
time_tokens = {"day", "night", "dawn", "dusk"}
style_tokens = {"clean", "with-assets", "labels"}
other = {
    "VTT",
    "PDF",
    "PDFs",
    "Print",
    "Jpeg",
    "PNGS",
    "PNG",
    "WEBP",
    "Gridded",
    "Gridless",
    "No Grid",
    "Music",
    "Tiles",
    "Transparent",
    "Interior",
    "Interiors",
    "Animated Scenes",
    "Variants",
    "Key & Design Notes",
    "Exterior",
    "Clean",
    "Asset Pack",
}
# Combine them
KNOWN_VARIANT_TOKENS = season_tokens | time_tokens | style_tokens | other
