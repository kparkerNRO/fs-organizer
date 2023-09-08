from pytest import skip
from virtual_folder import VirtualFolder
import organizer_v2 as organizer
import logging
from pathlib import Path
import json
import pytest

log = logging.getLogger("organizer_v2_test")
log.setLevel("INFO")


def build_virtual_fs_recursive(root_node: VirtualFolder, structure: dict):
    for key, data in structure.items():
        folder = VirtualFolder(path=None, name=key)
        root_node.add_virtual_subfolder(folder)
        if data:
            build_virtual_fs_recursive(folder, data)
        else:
            folder.add_file(Path("."))


def build_virtual_fs(structure: dict):
    folder = VirtualFolder(path=None, name="root")
    build_virtual_fs_recursive(folder, structure)
    return folder


# class TestHelpers:
#     pass


class TestOrganizeGroups:
    def generate_test_data(self, expected_results):
        test_data = {key: {} for key, value in expected_results.items()}
        return build_virtual_fs(test_data)

    def run_group_similar_folders(self, expected_results):
        test_fs = self.generate_test_data(expected_results)
        grouped_folders = organizer.group_similar_folders(test_fs)

        assert grouped_folders == expected_results

    def test_group_similar_folders(self):
        expected_results = {
            "test data one": ["test data", "one"],
            "test data two": ["test data", "two"],
        }

        self.run_group_similar_folders(expected_results)

    def test_one_name_overlap(self):
        expected_results = {"test one": ["test one"], "test two": ["test two"]}

        self.run_group_similar_folders(expected_results)

    def test_extract_view_type(self):
        expected_results = {
            "test data one VTT": ["VTT", "test data", "one"],
            "test data two Print": ["Print", "test data", "two"],
        }
        self.run_group_similar_folders(expected_results)

    def test_ignore_common(self):
        expected_results = {
            "City of one": ["City of one"],
            "City of two": ["City of two"],
        }
        self.run_group_similar_folders(expected_results)

    def test_overlapping_groups(self):
        expected_results = {
            "A wild test one": ["A wild", "test one"],
            "A wild test three": ["A wild", "test three"],
            "A wild two": ["A wild", "two"],
            "A wild four": ["A wild", "four"],
        }
        self.run_group_similar_folders(expected_results)

    def test_base_handling(self):
        expected_results = {
            "test data VTT": ["VTT", "test data", "Base"],
            "test data Print": ["Print", "test data", "Base"],
        }
        self.run_group_similar_folders(expected_results)

    def test_organize_groups(self):
        expected_results = {
            "test data VTT": "VTT/test data/Base",
            "test data Print": "Print/test data/Base",
        }

        test_fs = self.generate_test_data(expected_results)
        grouped_folders = organizer.organize_groups(test_fs)

        new_names = {
            old_name: node.name for old_name, node in grouped_folders.subfolders.items()
        }

        assert new_names == expected_results

    def test_organize_groups_multilayer(self):
        test_structure = {
            "A wild test one": {
                "test data 1": {},
                "test data 2": {},
            },
            "A wild test three": {
                "test data 1": {},
                "test data 2": {},
            },
            "A wild two": {
                "test data 1": {},
                "test data 2": {},
            },
            "A wild four": {
                "test data 1": {},
                "test data 2": {},
            },
        }

        test_fs = build_virtual_fs(test_structure)
        grouped_folders = organizer.organize_groups(test_fs)
        output_structure = grouped_folders.get_folders_dict()

        expected_results = {
            "root": {
                "A wild/test one": {
                    "test data/1": {"": ""},
                    "test data/2": {"": ""},
                },
                "A wild/test three": {
                    "test data/1": {"": ""},
                    "test data/2": {"": ""},
                },
                "A wild/two": {
                    "test data/1": {"": ""},
                    "test data/2": {"": ""},
                },
                "A wild/four": {
                    "test data/1": {"": ""},
                    "test data/2": {"": ""},
                },
            }
        }

        assert output_structure == expected_results

