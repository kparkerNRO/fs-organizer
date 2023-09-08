# first, split out <tc>/<grouping>/<print/vtt>/(etc)
# can't include 'Unsorted' in this, tho

# last pass through - if there are any folders that have just a "base" folder in them
# or a "base" + "clean", remove the base folder
from pathlib import Path

import click
from collections import defaultdict
from common import VIEW_TYPES, delete_empty_dir, print_path_operation, try_move_file


def reorder_file_types(file: Path, working_path: list[str], target_index):
    if file.name in VIEW_TYPES:
        if target_index < len(working_path):
            working_path.insert(target_index, file.name)
            return True
    return False


def reorder_path(working_paths: list, target_index, view_set=VIEW_TYPES):
    for file_type in view_set:
        try:
            index = working_paths.index(file_type)
            if index != target_index:
                working_paths.remove(file_type)
                working_paths.insert(target_index, file_type)
        except ValueError:
            pass


def should_keep_entry(subfile, siblings) -> bool:
    # if only one subfolder exists and is named "base", remove it
    if not subfile == "Base":
        return True

    if len(siblings) == 1:
        return False

    # OR if there are only two directories
    # elif len(siblings) == 2 and "Clean" in siblings:
    #     return False

    return True


def organize_tom_cartos(
    true_path: Path, base_path, working_path, should_execute, reorder_index
):
    printed_rename = False
    # path_reordered = False

    # print(true_path)
    # print(f"\t{path_reordered}")
    # print(working_path)
    if true_path.name == "City of Dorran":
        print()
        pass

    duplicate_files = []
    files = [file.name for file in true_path.iterdir()]
    if "Base" in files:
        print(f"Have a base file. Path is {true_path}\n\tSiblings are: {files}")
    for subfile in true_path.iterdir():
        if subfile.is_dir():

            next_path = working_path.copy()

            path_reordered = reorder_file_types(subfile, next_path, reorder_index)

            if not path_reordered:
                if should_keep_entry(subfile.name, files):
                    next_path.append(subfile.name)

            duplicate_files.extend(
                organize_tom_cartos(
                    subfile, base_path, next_path, should_execute, reorder_index
                )
            )

            delete_empty_dir(subfile, should_execute)
        else:
            reorder_path(working_path, reorder_index+1, {"Base", "Clean", "Base Winter", "Winter Clean"})
            reorder_path(working_path, reorder_index)
            output_path = base_path / Path(*working_path)

            if true_path != output_path:
                if not printed_rename:
                    # print(true_path)
                    print_path_operation(
                        "rename", true_path, output_path, should_execute
                    )

                printed_rename = True

                if result := try_move_file(subfile, output_path, should_execute):
                    duplicate_files.append(result)
    return duplicate_files


def find_duplicates(path: Path):
    filename_to_path = defaultdict(list)
    for subfile in path.iterdir():
        if subfile.is_dir():
            if subfile.name == ".ts":
                continue
            working_names = find_duplicates(subfile)
            for filename, paths in working_names.items():
                if filename in filename_to_path:
                    filename_to_path[filename].extend(paths)
                else:
                    filename_to_path[filename] = paths
        else:
            filename_to_path[subfile.name].append(path)

    return filename_to_path


def process_duplicates(path):
    filename_to_path = find_duplicates(path)
    for filename, paths in filename_to_path.items():
        print_entries = ["Print" in path.parts for path in paths]
        vtt_entries = ["VTT" in path.parts for path in paths]
        last_entry = [path.name for path in paths]

        all_same = last_entry.count(last_entry[0]) == len(last_entry)
        different_types = (True in print_entries and True in vtt_entries) and len(
            paths
        ) == 2
        if len(paths) > 1 and not all_same and not different_types:
            print(f"Have duplicate file {filename}")
            print("Paths are:")
            for path in paths:
                print(f"\t{path}")


@click.command()
@click.argument("path")
@click.option("--exec", is_flag=True, default=False)
def _do_the_work(path, exec):
    path_obj = Path(path)
    parts = list(path_obj.parts)
    start_len = 1
    parts = []
    duplicates = organize_tom_cartos(path_obj, path_obj, parts, exec, start_len)

    if duplicates:
        print()
        print("------ Duplicates -------")
        for file in duplicates:
            print(f"tried to move:\n\t'{file[0]}'")
            print(f"\t\tto \n\t'{file[1]}'")

    process_duplicates(path_obj)


if __name__ == "__main__":
    _do_the_work()
