from pathlib import Path
import filecmp


class InsertException(Exception):
    def __init__(self, source_path: Path, target_path: Path):
        self.source_path = source_path
        self.target_path = target_path


class VirtualFile:
    def __init__(self, path: Path):
        self.source_path = path
        self.name = path.name

    def is_dir(self):
        return False

    def is_file(self):
        return True

    def get_subfolders_dict(self, show_files=True):
        return ""

    def get_folders_dict(self, show_files=True):
        return self.name

    def __str__(self) -> str:
        return f"Virtual File: {self.name}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __value: object) -> bool:
        return self.name == __value.name

    def count_files(self):
        return 1


class VirtualFolder:
    def __init__(self, path: Path, name: str = None):
        self.source_path = path

        if name is not None:
            self.name = name
        else:
            self.name = path.name
        self.subfolders = {}

    def is_dir(self):
        return True

    def is_file(self):
        return False

    def add_file(self, path: Path):
        return self.add_virtual_subfolder(VirtualFile(path))

    def add_subfolder(self, path: Path):
        return self.add_virtual_subfolder(VirtualFolder(path))

    def add_virtual_subfolder(self, virtual_path):
        if virtual_path.name not in self.subfolders:
            self.subfolders[virtual_path.name] = virtual_path
        else:
            duplicate_path = self.subfolders[virtual_path.name]
            if isinstance(virtual_path, VirtualFolder):
                # merge subfolders if the name is a duplicate
                for subfolder in virtual_path.subfolders.values():
                        duplicate_path.add_virtual_subfolder(subfolder)

            else:
                # check to see if the files are duplicates. If so
                # we can ignore the insertion
                compare = filecmp.cmp(
                    virtual_path.source_path, duplicate_path.source_path
                )
                if not compare:
                    raise InsertException(
                        virtual_path.source_path, duplicate_path.source_path
                    )
        return self.subfolders[virtual_path.name]

    def get_subfolders_dict(self, show_files=True):
        subfolder_dict = {}
        for subfolder_name, subfolder in self.subfolders.items():
            if (not show_files) and isinstance(subfolder, VirtualFile):
                continue
            subfolder_dict[subfolder.name] = subfolder.get_subfolders_dict(show_files)
        return subfolder_dict

    def get_folders_dict(self, show_files=True):
        subfolder_dict = {}
        for subfolder in self.subfolders.values():
            if (not show_files) and isinstance(subfolder, VirtualFile):
                continue
            subfolder_dict[subfolder.name] = subfolder.get_subfolders_dict(show_files)

        output_dict = {self.name: subfolder_dict}
        return output_dict

    def insert_intermediate_folder(self, name):
        intermediate_folder = VirtualFolder(path=None, name=name)
        intermediate_folder.subfolders = self.subfolders
        self.subfolders = {name: intermediate_folder}

    def count_files(self):
        total = 0
        for folder in self.subfolders.values():
            total += folder.count_files()
        return total

    def __str__(self) -> str:
        # return self.name + " -> " + len(self.subfolders)
        return f"Virtual Folder: {self.name} -> {len(self.subfolders)} files"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if self.name == value.name and len(self.subfolders) == len(value.subfolders):
            subfolder_matches = 0
            for name, subfolder in self.subfolders.items():
                if name not in value.subfolders or subfolder != value.subfolders[name]:
                    return False
                else:
                    subfolder_matches += 1
            return subfolder_matches == len(self.subfolders)

        return False


def build_folder_structure(root_path: Path):
    root_folder = VirtualFolder(root_path)
    for item in root_path.iterdir():
        if item.is_dir():
            current_folder = root_folder.add_subfolder(item)
            build_folder_structure_recursive(current_folder, item)
        elif item.is_file():
            root_folder.add_file(item)
    return root_folder


def build_folder_structure_recursive(folder, path: Path):
    for item in path.iterdir():
        if item.is_dir():
            subfolder = folder.add_subfolder(item)
            build_folder_structure_recursive(subfolder, item)
        elif item.is_file():
            folder.add_file(item)
