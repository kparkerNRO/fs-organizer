from virtual_folder import VirtualFile, VirtualFolder
import organizer as organizer
from pathlib import Path

def create_virtual_file(input, key=""):
    if isinstance(input, str):
        vf = VirtualFile(Path(input))
    if isinstance(input, Path):
        vf = VirtualFile(input)
    if isinstance(input, VirtualFile):
        vf = input

    if key:
        vf.name = key

    return vf


def build_virtual_fs_recursive(root_node: VirtualFolder, structure: dict):
    if isinstance(structure, VirtualFile):
        return structure

    for key, data in structure.items():
        folder = VirtualFolder(path=None, name=key)
        root_node.add_virtual_subfolder(folder)
        if isinstance(data, (str, Path, VirtualFile)):
            root_node.contents.pop(key, None)
            virtual_node = create_virtual_file(data, key)
            root_node.add_virtual_subfolder(virtual_node)
        elif isinstance(data, set):
            for entry in data:
                if isinstance(data, (Path, VirtualFile)):
                    root_node.contents.pop(key, None)
                    virtual_node = create_virtual_file(entry)
                    root_node.add_virtual_subfolder(virtual_node)
                else:
                    folder.add_file(Path(entry))
        elif data is None:
            continue
        elif data:
            build_virtual_fs_recursive(folder, data)
        else:
            folder.add_file(Path("./testfile.txt"))


def build_virtual_fs(structure: dict, root_name="root"):
    folder = VirtualFolder(path=None, name=root_name)
    build_virtual_fs_recursive(folder, structure)
    return folder


def build_placeholder_file(name):
    folder = VirtualFolder(path=None, name=name)
    folder.add_file(Path("./testfile.txt"))
    return folder