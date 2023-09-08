from pathlib import Path
import filecmp

class InsertException(Exception):
    pass


class VirtualFile:
    def __init__(self, path: Path):
        self.source_path = path
        self.name = path.name

    def is_dir(self):
        return False

    def is_file(self):
        return True

    def get_subfolders_dict(self):
        return ""

    def get_folders_dict(self):
        return self.name

    def __str__(self) -> str:
        return f"Virtual File: {self.name}"

    def __repr__(self) -> str:
        return self.__str__()

    def count_files(self):
        return 1


class VirtualFolder:
    def __init__(self, path: Path, name: str = ""):
        self.source_path = path

        if name:
            self.name = name
        else:
            self.name = path.name
        self.subfolders = {}

    def is_dir(self):
        return True

    def is_file(self):
        return False

    def add_subfolder(self, path: Path):
        if path.name not in self.subfolders:
            self.subfolders[path.name] = VirtualFolder(path)
        else:
            raise InsertException
        return self.subfolders[path.name]

    def add_virtual_subfolder(self, virtual_path, overwrite_files = False, ignore_duplicates=True):
        if virtual_path.name not in self.subfolders:
            self.subfolders[virtual_path.name] = virtual_path
        else:
            duplicate_path = self.subfolders[virtual_path.name]
            if isinstance(virtual_path, VirtualFolder):
                for subfolder in virtual_path.subfolders.values():
                    duplicate_path.add_virtual_subfolder(subfolder)
            else:
                compare = filecmp.cmp(virtual_path.source_path, duplicate_path.source_path)
                if not compare:
                    raise InsertException
                else:
                    print("files are identical, ignoring overwrite")
        return self.subfolders[virtual_path.name]

    def add_file(self, path: Path):
        return self.add_virtual_subfolder(VirtualFile(path))
        

    def get_subfolders_dict(self):
        subfolder_dict = {}
        for subfolder_name, subfolder in self.subfolders.items():
            # if isinstance(subfolder, VirtualFolder):
            subfolder_dict[subfolder.name] = subfolder.get_subfolders_dict()
        return subfolder_dict

    def get_folders_dict(self):
        subfolder_dict = {}
        for subfolder in self.subfolders.values():
            subfolder_dict[subfolder.name] = subfolder.get_subfolders_dict()

        output_dict = {self.name: subfolder_dict}
        return output_dict

    def insert_intermediate_folder(self, name):
        # TODO check to see if the intermediate name is actually
        # in the subfiles already
        # if name in self.subfolders:
        #     return
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
