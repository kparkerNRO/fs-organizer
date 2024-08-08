from pathlib import Path
import filecmp
import os
import logging

logger = logging.getLogger(__name__)

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

    def get_subfolders_dict(self, show_files=True, show_file_count=True):
        return ""

    def get_folders_dict(self, show_files=True, show_file_count=True):
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
        self.contents = {}

    def is_dir(self):
        return True

    def is_file(self):
        return False

    def add_file(self, path: Path):
        return self.add_virtual_subfolder(VirtualFile(path))

    def add_subfolder(self, path: Path):
        return self.add_virtual_subfolder(VirtualFolder(path))

    def add_virtual_file(self, file: VirtualFile):
        name = file.name
        if name in self.contents:
            count = 1
            basename, ext = os.path.splitext(name)
            while name in self.contents:
                name = f"{basename}-{count}{ext}"
                count += 1
            file.name = name
        self.contents[name] = file

    def merge_subfolders(self, virtual_path, drop_duplicates=True):
        if virtual_path.name not in self.contents:
            self.contents[virtual_path.name] = virtual_path
            return True
        else:
            duplicate_path = self.contents[virtual_path.name]
            if isinstance(virtual_path, VirtualFolder):
                # merge subfolders if the name is a duplicate
                key_list = list(virtual_path.contents.keys())
                full_transfer = True
                for subfolder_key in key_list:
                    subfolder = virtual_path.contents[subfolder_key]
                    added_file = duplicate_path.merge_subfolders(subfolder)
                    if added_file:
                        virtual_path.contents.pop(subfolder_key)
                    else:
                        if isinstance(virtual_path, VirtualFile) and drop_duplicates:
                            virtual_path.contents.pop(subfolder_key)
                        else:
                            full_transfer = False

                return full_transfer
            elif isinstance(virtual_path, VirtualFile):
                compare = filecmp.cmp(
                    virtual_path.source_path, duplicate_path.source_path
                )
                if compare:
                    # if they are identical, do nothing
                    logger.info(
                        f"Files \n\t{virtual_path.source_path} and \n\t{duplicate_path.source_path} \nare identical. Dropping {duplicate_path.source_path}"
                    )
                else:
                    self.add_virtual_file(virtual_path)

                return not compare

        return False

    def add_virtual_subfolder(self, virtual_path):
        if virtual_path.name not in self.contents:
            self.contents[virtual_path.name] = virtual_path
        else:
            duplicate_path = self.contents[virtual_path.name]
            if isinstance(virtual_path, VirtualFolder):
                # merge subfolders if the name is a duplicate
                for subfolder in virtual_path.contents.values():
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
                else:
                    logger.info(
                        f"Files \n\t{virtual_path.source_path} and \n\t{duplicate_path.source_path} \nare identical. Dropping {duplicate_path.source_path}"
                    )
        return self.contents[virtual_path.name]

    def get_subfolders_dict(self, show_files=True, show_file_count=False):
        subfolder_dict = {}
        file_count = 0
        for subfolder_name, subfolder in self.contents.items():
            if isinstance(subfolder, VirtualFile):
                file_count += 1
                if not show_files:
                    continue
            subfolder_dict[subfolder.name] = subfolder.get_subfolders_dict(
                show_files, show_file_count
            )
        if show_file_count:
            subfolder_dict["files"] = file_count
        return subfolder_dict

    def get_folders_dict(self, show_files=True, show_file_count=False):
        subfolder_dict = {}
        file_count = 0
        for subfolder in self.contents.values():
            if isinstance(subfolder, VirtualFile):
                file_count += 1
                if not show_files:
                    continue
            subfolder_dict[subfolder.name] = subfolder.get_subfolders_dict(
                show_files, show_file_count
            )
        if show_file_count:
            subfolder_dict["files"] = file_count
        output_dict = {self.name: subfolder_dict}

        return output_dict

    def insert_intermediate_folder(self, name):
        intermediate_folder = VirtualFolder(path=None, name=name)
        intermediate_folder.contents = self.contents
        self.contents = {name: intermediate_folder}

    def count_files(self):
        total = 0
        for folder in self.contents.values():
            total += folder.count_files()
        return total

    def __str__(self) -> str:
        # return self.name + " -> " + len(self.contents)
        return f"Virtual Folder: {self.name} -> {len(self.contents)} files"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if self.name == value.name and len(self.contents) == len(value.contents):
            subfolder_matches = 0
            for name, subfolder in self.contents.items():
                if name not in value.contents or subfolder != value.contents[name]:
                    return False
                else:
                    subfolder_matches += 1
            return subfolder_matches == len(self.contents)

        return False


def build_folder_structure(root_path: Path, ignore_hidden_dirs = True):
    root_folder = VirtualFolder(root_path)
    for item in root_path.iterdir():
        if ignore_hidden_dirs and item.name.startswith('.'):
            continue
        
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
