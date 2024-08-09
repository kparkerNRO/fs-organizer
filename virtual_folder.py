from pathlib import Path
import filecmp
import os
import logging
from typing import Optional
import zipfile
import json

from database.tables import Exports, Files, Tags

logger = logging.getLogger(__name__)


class InsertException(Exception):
    def __init__(self, source_path: Path, target_path: Path):
        self.source_path = source_path
        self.target_path = target_path


class VirtualFile:
    def __init__(self, path: Path, zip_path: Optional[Path] = None):
        self.source_path = path
        self.name = path.name
        self.zip_path = zip_path

    def is_dir(self):
        return False

    def is_file(self):
        return True

    def get_subfolders_dict(self, show_files=True, show_file_count=True):
        return self

    def get_folders_dict(self, show_files=True, show_file_count=True):
        return self.name

    def __str__(self) -> str:
        return f"Virtual File: {self.name}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, str):
            return self.name == __value or __value == ""
        elif isinstance(__value, VirtualFolder):
            return self.name == __value.name
        elif isinstance(__value, VirtualFile):
            return self.name == __value.name
        else:
            return False

    def count_files(self):
        return 1


class VirtualFolder:
    def __init__(self, path: Path, name: str = None, zip_path: Optional[Path] = None):
        self.source_path = path
        self.zip_path = zip_path

        if name is not None:
            self.name = name
        else:
            self.name = path.name
        self.contents = {}

    def is_dir(self):
        return True

    def is_file(self):
        return False

    def add_file(self, path: Path, zip_path: Optional[str] = None):
        return self.add_virtual_subfolder(VirtualFile(path, zip_path=zip_path))

    def add_subfolder(self, path: Path, zip_path: str = None):
        return self.add_virtual_subfolder(VirtualFolder(path, zip_path=zip_path))

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
        elif virtual_path.zip_path is not None:
            # this isn't a real file, so there is no comparison
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

    def export_files_to_db(self, session, export_id, tags):
        for name, item in self.contents.items():
            if isinstance(item, VirtualFolder):
                tags.add(name)
                item.export_files_to_db(session, export_id=export_id, tags=tags)
            else:
                session.add(
                    Files(
                        file_name=item.name,
                        export_id=export_id,
                        zip_parent=str(self.zip_path),
                        original_path=str(item.source_path),
                        # tags=tags,
                    )
                )

    def create_database_export(self, session, run_id, export_name = None, export_tags = True):
        
        export = Exports(
            export_stage=export_name or self.name,
            run_id=run_id,
            folder_structure=json.dumps(self.get_folders_dict(show_files=False)),
        )
        session.add(export)
        session.flush()
        tags = set()
        self.export_files_to_db(session, export.id, tags)
        if export_tags:
            for tag in tags:
                session.add(
                    Tags(
                        original_name=tag,
                    )
                )
        session.commit()

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


def build_folder_structure(
    path: Path,
    folder: Optional[VirtualFolder] = None,
    ignore_hidden_dirs=True,
    preserve_modules=True,
    zip_path: Path = None,
):
    if folder is None:
        folder = VirtualFolder(path)

    for item in path.iterdir():
        if ignore_hidden_dirs and item.name.startswith("."):
            continue
        if item.name.startswith("__MACOSX") or ".DS_Store" in item.name:
            continue

        if item.is_dir():
            current_folder = folder.add_subfolder(item, zip_path)
            build_folder_structure(
                folder=current_folder,
                path=item,
                preserve_modules=preserve_modules,
                ignore_hidden_dirs=ignore_hidden_dirs,
                zip_path=zip_path,
            )
        elif hasattr(item, "suffix") and item.suffix == ".zip":
            with zipfile.ZipFile(item, "r") as zipref:
                zip_path = zipfile.Path(zipref)
                zip_children = [p.name for p in zip_path.iterdir()]
                # if it's a foundry module, it's a file
                if preserve_modules and "module.json" in zip_children:
                    folder.add_file(VirtualFile(item, zip_path))

                else:
                    current_folder = folder.add_subfolder(item, zip_path)
                    build_folder_structure(
                        folder=current_folder,
                        path=zip_path,
                        zip_path=item,
                        preserve_modules=preserve_modules,
                        ignore_hidden_dirs=ignore_hidden_dirs,
                    )
        elif item.is_file():
            folder.add_file(item, zip_path)

    return folder
