from pytest import skip
from virtual_folder import VirtualFolder
import organizer_v2 as organizer
import logging
from pathlib import Path
import json
import pytest

log = logging.getLogger("organizer_v2_test")
log.setLevel("INFO")

TESTFILE_DICT = {"testfile.txt": ""}


def build_virtual_fs_recursive(root_node: VirtualFolder, structure: dict):
    for key, data in structure.items():
        folder = VirtualFolder(path=None, name=key)
        root_node.add_virtual_subfolder(folder)
        if data:
            build_virtual_fs_recursive(folder, data)
        else:
            folder.add_file(Path("./testfile.txt"))


def build_virtual_fs(structure: dict):
    folder = VirtualFolder(path=None, name="root")
    build_virtual_fs_recursive(folder, structure)
    return folder


def build_placeholder_file(name):
    folder = VirtualFolder(path=None, name=name)
    folder.add_file(Path("./testfile.txt"))
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
        expected_results = {"test one": ["test", "one"], "test two": ["test", "two"]}

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
                    "test data/1": TESTFILE_DICT,
                    "test data/2": TESTFILE_DICT,
                },
                "A wild/test three": {
                    "test data/1": TESTFILE_DICT,
                    "test data/2": TESTFILE_DICT,
                },
                "A wild/two": {
                    "test data/1": TESTFILE_DICT,
                    "test data/2": TESTFILE_DICT,
                },
                "A wild/four": {
                    "test data/1": TESTFILE_DICT,
                    "test data/2": TESTFILE_DICT,
                },
            }
        }

        assert output_structure == expected_results


class TestDictToVirtualFs:
    def test_basic(self):
        test_structure = {
            "one": build_placeholder_file("one"),
            "two": build_placeholder_file("two"),
        }

        expected_results = list(test_structure.values())

        results = organizer.dict_to_virtualfs_nodes(test_structure)
        assert results == expected_results

    def test_empty_folder(self):
        test_structure = {
            "one": VirtualFolder(None, "one"),
        }

        expected_results = list(test_structure.values())
        results = organizer.dict_to_virtualfs_nodes(test_structure)
        assert results == expected_results

    # def test_only_one_subfolder(self):
    #     virtual_fs = VirtualFolder(None, "one")
    #     virtuals_child = build_placeholder_file("two")
    #     virtual_fs.add_virtual_subfolder(virtuals_child)
    #     grandchild = list(virtuals_child.subfolders.values())[0]
    #     output = VirtualFolder(None, "out")
    #     output.add_virtual_subfolder(grandchild)

    #     test_structure = {
    #         "out": virtual_fs,
    #     }

    #     results = organizer.dict_to_virtualfs_nodes(test_structure)

    #     expected_results = [output]

    #     assert results == expected_results

    def test_multiple_subfolders(self):
        virtual_fs = VirtualFolder(None, "one")
        child1 = build_placeholder_file("two")
        child2 = build_placeholder_file("three")
        virtual_fs.add_virtual_subfolder(child1)
        virtual_fs.add_virtual_subfolder(child2)

        output = VirtualFolder(None, "out")
        output.add_virtual_subfolder(child1)
        output.add_virtual_subfolder(child2)

        test_structure = {
            "out": virtual_fs,
        }

        results = organizer.dict_to_virtualfs_nodes(test_structure)

        expected_results = [output]

        assert results == expected_results

    def test_new_heirarchy(self):
        virtual_fs = VirtualFolder(None, "one")
        child1 = build_placeholder_file("two")
        child2 = build_placeholder_file("three")
        virtual_fs.add_virtual_subfolder(child1)
        virtual_fs.add_virtual_subfolder(child2)

        test_structure = {
            "test": {"out": virtual_fs},
        }

        output = VirtualFolder(None, "test")
        output_child = VirtualFolder(None, "out")

        output.add_virtual_subfolder(output_child)
        output_child.add_virtual_subfolder(child1)
        output_child.add_virtual_subfolder(child2)

        results = organizer.dict_to_virtualfs_nodes(test_structure)

        result_tree = results[0].get_folders_dict()
        expected_tree = output.get_folders_dict()

        assert result_tree == expected_tree

    def test_three_deep(self):
        virtual_fs = VirtualFolder(None, "one")
        child1 = build_placeholder_file("two")
        child2 = build_placeholder_file("three")
        virtual_fs.add_virtual_subfolder(child1)
        virtual_fs.add_virtual_subfolder(child2)

        test_structure = {
            "test": {
                "first": {"out": virtual_fs},
            }
        }

        output = VirtualFolder(None, "test")
        output_g1 = VirtualFolder(None, "first")
        output_g2 = VirtualFolder(None, "out")

        output.add_virtual_subfolder(output_g1)
        output_g1.add_virtual_subfolder(output_g2)
        output_g2.add_virtual_subfolder(child1)
        output_g2.add_virtual_subfolder(child2)

        results = organizer.dict_to_virtualfs_nodes(test_structure)
        result_tree = results[0].get_folders_dict()
        expected_tree = output.get_folders_dict()

        assert result_tree == expected_tree

    def test_two_subfolders(self):
        child1 = build_placeholder_file("two")
        child2 = build_placeholder_file("three")

        test_structure = {
            "test": {
                "first": child1,
                "second": child2,
            }
        }

        results = organizer.dict_to_virtualfs_nodes(test_structure)
        result_tree = results[0].get_folders_dict()

        expected_structure = {"test": {"first": TESTFILE_DICT, "second": TESTFILE_DICT}}

        assert result_tree == expected_structure

    def test_two_subfolders_three_deep(self):
        child1 = build_placeholder_file("two")
        child2 = build_placeholder_file("three")

        test_structure = {
            "test": {
                "data": {
                    "first": child1,
                    "second": child2,
                },
                "data2": {
                    "first": child1,
                    "second": child2,
                },
            }
        }

        results = organizer.dict_to_virtualfs_nodes(test_structure)
        result_tree = results[0].get_folders_dict()

        expected_structure = {
            "test": {
                "data": {"first": TESTFILE_DICT, "second": TESTFILE_DICT},
                "data2": {"first": TESTFILE_DICT, "second": TESTFILE_DICT},
            }
        }

        assert result_tree == expected_structure


class TestReorganizeFs:
    def test_basic(self):
        test_structure = {
            "A wild/test one": {
                "test data/1": {},
                "test data/2": {},
            },
            "A wild/test three": {
                "test data/1": {},
                "test data/2": {},
            },
            "A wild/two": {
                "test data/1": {},
                "test data/2": {},
            },
            "A wild/four": {
                "test data/1": {},
                "test data/2": {},
            },
        }

        expected_results = {
            "root": {
                "A wild": {
                    "test one": {
                        "test data": {
                            "1": TESTFILE_DICT,
                            "2": TESTFILE_DICT,
                        }
                    },
                    "test three": {
                        "test data": {
                            "1": TESTFILE_DICT,
                            "2": TESTFILE_DICT,
                        }
                    },
                    "two": {
                        "test data": {
                            "1": TESTFILE_DICT,
                            "2": TESTFILE_DICT,
                        }
                    },
                    "four": {
                        "test data": {
                            "1": TESTFILE_DICT,
                            "2": TESTFILE_DICT,
                        }
                    },
                }
            }
        }

        test_fs = build_virtual_fs(test_structure)
        results = organizer.reorganize_virtualfs(test_fs)
        output_structure = results.get_folders_dict()

        assert output_structure == expected_results


class TestReorganizeTree:
    def test_inversion(self):
        base_tree = {"a": {"b": {"c": {"d": {}}}}}

        frequencies = {"root": 0, "a": 5, "b": 4, "c": 3, "d": 2}
        expected_result = {"root": {"d": {"c": {"b": {"a": TESTFILE_DICT}}}}}

        virtual_fs = build_virtual_fs(base_tree)
        result = organizer.reorganize_tree(virtual_fs, frequencies)
        result_tree = result.get_folders_dict()

        assert result_tree == expected_result

    def test_multiple_folders(self):
        base_tree = {"a": {"b": {"e": {}}, "c": {"e": {}}}}

        frequencies = {"root": 0, "a": 5, "b": 4, "c": 3, "d": 2, "e": 10}
        expected_result = {
            "root": {"b": {"a": {"e": TESTFILE_DICT}}, "c": {"a": {"e": TESTFILE_DICT}}}
        }

        virtual_fs = build_virtual_fs(base_tree)
        result = organizer.reorganize_tree(virtual_fs, frequencies)
        result_tree = result.get_folders_dict()

        assert result_tree == expected_result

    def test_complex_structure(self):
        base_tree = {
            "tom": {
                "PDF": {
                    "sample_folder": {
                        "day": {
                            "base": {"sample_file": {}},
                            "tile": {"sample_file": {}},
                        },
                        "night": {"sample_file": {}},
                    },
                    "sample_file": {},
                },
                "VTT": {
                    "sample_folder": {
                        "day": {
                            "base": {"sample_file": {}},
                            "tile": {"sample_file": {}},
                        },
                        "night": {"sample_file": {}},
                    },
                    "sample_file": {},
                },
                "sample_file": {},
            }
        }

        frequencies = {
            "root": 0,
            "tom": 0,
            "sample_folder": 2,
            "PDF": 3,
            "VTT": 3,
            "night": 4,
            "base": 4,
            "tile": 5,
            "day": 5,
            "sample_file": 10,
        }

        expected_result = {
            "root": {
                "tom": {
                    "PDF": {"sample_file": TESTFILE_DICT},
                    "VTT": {"sample_file": TESTFILE_DICT},
                    "sample_file": TESTFILE_DICT,
                    "sample_folder": {
                        "PDF": {
                            "base": {"day": {"sample_file": TESTFILE_DICT}},
                            "day": {"tile": {"sample_file": TESTFILE_DICT}},
                            "night": {"sample_file": TESTFILE_DICT},
                        },
                        "VTT": {
                            "base": {"day": {"sample_file": TESTFILE_DICT}},
                            "day": {"tile": {"sample_file": TESTFILE_DICT}},
                            "night": {"sample_file": TESTFILE_DICT},
                        },
                    },
                }
            }
        }

        virtual_fs = build_virtual_fs(base_tree)
        result = organizer.reorganize_tree(virtual_fs, frequencies)
        result_tree = result.get_folders_dict()

        print(json.dumps(result_tree, indent=4))
        print(json.dumps(expected_result, indent=4))

        assert result_tree == expected_result
