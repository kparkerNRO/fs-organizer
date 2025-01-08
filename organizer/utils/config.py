# import tomllib

# from common import FileBackupState, ZipBackupState

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

# class Config:
#     def __init__(self, config_file=None, **kwargs) -> None:
#         # organization behavior
#         self.zip_backup_state: ZipBackupState = ZipBackupState.KEEP
#         self.file_backup_state: FileBackupState = FileBackupState.IN_PLACE
#         self.preserve_modules = True

#         # execution behavior
#         self.unzip = True
#         self.organize = True
#         self.should_execute = True

#         # file handling behaviors
#         self.creators_to_exceptions = CREATOR_REMOVES
#         self.keywords_to_remove = REPLACE_EXCEPTIONS
#         self.folder_names_to_rename = FILE_NAME_EXCEPTIONS
#         self.terms_not_to_group = GROUPING_EXCEPTIONS

#         # file structure
#         self.input_dir = None
#         self.output_dir = None
#         self.zip_backup_dir = None
#         self.intermediate_zip_dir = None

#         self.root_dir = None
#         self.levels_to_preserve = 0
#         self.preserve_creators = True

#         if config_file:
#             self._parse_config_file(config_file)

#         self._parse_inputs(kwargs)

#         #TODO check that the minimum needed variables are set

#     def _parse_config_file(self, config_path):
#         with open(config_path, "rb") as f:
#             config = tomllib.load(f)


#         self.zip_backup_state = ZipBackupState[config.get('zip_behavior', self.zip_backup_state.name)]
#         self.file_backup_state = FileBackupState[config.get('backup_behavior', self.file_backup_state.name)]

#         self.preserve_modules = config.get('preserve_modules', self.preserve_modules)
#         self.levels_to_preserve = config.get('levels_to_preserve', self.levels_to_preserve)
#         self.preserve_creators  = config.get('preserve_creators', self.preserve_creators)

#         if 'folders' in config:
#             pass

#         if 'creators' in config:
#             pass

#         if 'folder_renames' in config:
#             pass

#         if 'trim_words' in config:
#             pass

#         if 'skip_grouping' in config:
#             pass


#     def _parse_inputs(self, **kwargs):
#         if "path" in kwargs:
#             self.input_dir = kwargs["path"]

#         if "output" in kwargs:
#             self.output_dir = kwargs["output"]
#             if "copy_data" in kwargs:
#                 self.file_backup_state = FileBackupState.COPY
#             else:
#                 self.file_backup_state = FileBackupState.MOVE

#         if "zip" in kwargs:
#             self.unzip = True
#             if "zip_backup_dir" in kwargs:
#                 self.zip_backup_dir = kwargs["zip_backup_dir"]
#                 self.zip_backup_state = ZipBackupState.MOVE
#             elif "delete_zip" in kwargs:
#                 if "zip_backup_dir" in kwargs:
#                     self.zip_backup_state = ZipBackupState.MOVE
#                 else:
#                     self.zip_backup_state = ZipBackupState.DELETE
#             if "output" in kwargs and "zip_backup_dir" in kwargs:
#                 self.zip_backup_state = ZipBackupState.MOVE

#         if "exec" in kwargs:
#             self.should_execute = True

#         if "skip_organize" in kwargs:
#             self.organize = False

#     def _validate_config(self):
#         pass
