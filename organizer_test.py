import tempfile
from common import FileBackupState
from virtual_folder import VirtualFile, VirtualFolder
import organizer as organizer
import logging
from pathlib import Path
import json

log = logging.getLogger("organizer_v2_test")
log.setLevel("INFO")

TESTFILE_DICT = {"testfile.txt": ""}


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
            "A wild test one": ["A wild", "test", "one"],
            "A wild test three": ["A wild", "test", "three"],
            "A wild two": ["A wild", "two"],
            "A wild four": ["A wild", "four"],
        }
        self.run_group_similar_folders(expected_results)

    def test_base_handling(self):
        expected_results = {
            "test data VTT": ["VTT", "test data"],
            "test data Print": ["Print", "test data"],
        }
        self.run_group_similar_folders(expected_results)

    def test_organize_groups(self):
        expected_results = {
            "test data VTT": "VTT/test data",
            "test data Print": "Print/test data",
        }

        test_fs = self.generate_test_data(expected_results)
        grouped_folders = organizer.organize_groups(test_fs)

        new_names = {
            old_name: node.name for old_name, node in grouped_folders.contents.items()
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
                "A wild/test/one": {
                    "test data/1": TESTFILE_DICT,
                    "test data/2": TESTFILE_DICT,
                },
                "A wild/test/three": {
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

    def test_nested_three_deep(self):
        expected_results = {
            "test data one": ["test data", "one"],
            "test data two a": ["test data", "two", "a"],
            "test data two three a": ["test data", "two", "three", "a"],
            "test data two three plus": ["test data", "two", "three", "plus"],
            "test data two three four": ["test data", "two", "three", "four"],
        }
        self.run_group_similar_folders(expected_results)


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

    def test_no_splitting(self):
        test_structure = {"test_data": {"one": {"two": "two"}}}

        expected_results = {"root": {"test_data": {"one": {"two": ""}}}}

        test_fs = build_virtual_fs(test_structure)
        results = organizer.reorganize_virtualfs(test_fs)
        output_structure = results.get_folders_dict()

        assert output_structure == expected_results

    def test_mixed_splitting(self):
        test_structure = {"test_data": {"one/three": {"two": "two"}}}

        expected_results = {"root": {"test_data": {"one": {"three": {"two": ""}}}}}

        test_fs = build_virtual_fs(test_structure)
        results = organizer.reorganize_virtualfs(test_fs)
        output_structure = results.get_folders_dict()

        assert output_structure == expected_results

    def test_empty_name(self):
        test_structure = {"test_data": {"one": {"": {"two": "two"}}}}

        expected_results = {"root": {"test_data": {"one": {"two": ""}}}}

        test_fs = build_virtual_fs(test_structure)
        results = organizer.reorganize_virtualfs(test_fs)
        output_structure = results.get_folders_dict()

        assert output_structure == expected_results

    def test_multi_level_empty_name(self):
        test_structure = {"test_data": {"one": {"": {"": {"two": "two"}}}}}

        expected_results = {"root": {"test_data": {"one": {"two": ""}}}}

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


class TestRemoveExtraFolders:
    def test_promote_base_subfolder(self):
        test_structure = {
            "test": {
                "base": {
                    "first": {"sample_file": {}, "second_sample": {}},
                },
            }
        }

        expected_output = {
            "root": {
                "test": {
                    "first": {
                        "sample_file": TESTFILE_DICT,
                        "second_sample": TESTFILE_DICT,
                    }
                }
            }
        }

        virtual_fs = build_virtual_fs(test_structure)
        result = organizer.remove_extra_folders(virtual_fs)
        result_tree = result.get_folders_dict()

        assert result_tree == expected_output

    def test_leave_base_with_files(self):
        test_structure = {
            "test": {"base": {"one", "two"}, "test2": {"sample_file": {}}}
        }

        expected_output = {
            "root": {"test": {"base": {"one": "", "two": ""}, "test2": TESTFILE_DICT}}
        }

        virtual_fs = build_virtual_fs(test_structure)
        result = organizer.remove_extra_folders(virtual_fs)
        result_tree = result.get_folders_dict()

        assert result_tree == expected_output

    def test_promote_base_subfolder_plus_companion(self):
        with tempfile.NamedTemporaryFile() as file, tempfile.NamedTemporaryFile() as file2:
            filePath = Path(file.name)
            basename = filePath.name

            filePath2 = Path(file2.name)
            basename2 = filePath2.name
            test_structure = {
                "test": {
                    "base": {
                        "first": {"sample_file": filePath, "sample_file2": filePath2},
                        "second_sample": filePath,
                    },
                }
            }

            expected_output = {
                "root": {
                    "test": {
                        "first": {
                            "sample_file": "",
                            "sample_file2": "",
                        },
                        "second_sample": "",
                    }
                }
            }

            virtual_fs = build_virtual_fs(test_structure)
            result = organizer.remove_extra_folders(virtual_fs)
            result_tree = result.get_folders_dict()

            assert result_tree == expected_output

    def test_promote_base_subfolder_plus_file(self):
        test_structure = {
            "test": {
                "base": {"sample_file": {}},
            },
        }

        expected_output = {
            "root": {
                "test": {
                    "sample_file": TESTFILE_DICT,
                }
            }
        }

        virtual_fs = build_virtual_fs(test_structure)
        result = organizer.remove_extra_folders(virtual_fs)
        result_tree = result.get_folders_dict()

        assert result_tree == expected_output

    def test_promote_single_subfolder(self):
        test_structure = {
            "test": {
                "first": {"sample_file": {}},
            },
        }

        expected_output = {"root": {"test": {"sample_file": TESTFILE_DICT}}}

        virtual_fs = build_virtual_fs(test_structure)
        result = organizer.remove_extra_folders(virtual_fs)
        result_tree = result.get_folders_dict()

        assert result_tree == expected_output

    def test_remove_empty_folder(self):
        test_structure = {
            "test": {"first": {"sample_file": {}, "second_file": {}}, "second": None},
        }

        expected_output = {
            "root": {
                "test": {
                    "first": {
                        "sample_file": TESTFILE_DICT,
                        "second_file": TESTFILE_DICT,
                    },
                }
            }
        }

        virtual_fs = build_virtual_fs(test_structure)
        result = organizer.remove_extra_folders(virtual_fs)
        result_tree = result.get_folders_dict()

        assert result_tree == expected_output

    def test_nested_conditions(self):
        test_structure = {
            "test": {
                "base": {
                    "first": {"sample_file": {}},
                }
            },
        }

        expected_output = {
            "root": {
                "test": {
                    "first": TESTFILE_DICT,
                },
            }
        }

        virtual_fs = build_virtual_fs(test_structure)
        result = organizer.remove_extra_folders(virtual_fs)
        result_tree = result.get_folders_dict()

        assert result_tree == expected_output

    def test_nested_bases(self):
        test_structure = {
            "base": {
                "base": {"base": {"sample_file": {}}},
                "test2": {},
            },
            "test1": {},
        }

        expected_output = {
            "test": {
                "base": {
                    "base": {"sample_file": TESTFILE_DICT},
                    "test2": TESTFILE_DICT,
                },
                "test1": TESTFILE_DICT,
            }
        }

        virtual_fs = build_virtual_fs(test_structure, "test")
        result = organizer.remove_extra_folders(virtual_fs)
        result_tree = result.get_folders_dict()

        assert result_tree == expected_output


class TestPromoteGrandchildren:
    def test_base(self):
        test_data = {"child": {"grandfile1", "grandfile2"}}

        expected = {"root": {"grandfile1": "", "grandfile2": ""}}

        virtual_fs = build_virtual_fs(test_data)
        organizer.promote_grandchildren(
            virtual_fs, "child", ["grandfile1", "grandfile2"]
        )
        result_tree = virtual_fs.get_folders_dict()

        assert result_tree == expected

    def test_duplicate_file(self):
        with tempfile.NamedTemporaryFile() as file:
            filename = file.name
            filePath = Path(filename)
            basenname = filePath.name
            test_data = {
                "child": {filePath},
            }

            expected = {"root": {basenname: ""}}

            virtual_fs = build_virtual_fs(test_data)
            virtual_fs.add_file(filePath)

            organizer.promote_grandchildren(virtual_fs, "child", [basenname])
            result_tree = virtual_fs.get_folders_dict()

            assert result_tree == expected

    def test_duplicate_filename(self):
        with tempfile.NamedTemporaryFile() as file1, tempfile.NamedTemporaryFile() as file2:
            filename = file1.name
            filePath = Path(filename)
            basenname = filePath.name

            # write some junk data so they're different files
            file1.write(b"test")
            file1.flush()

            dupe_file = VirtualFile(Path(file2.name))
            dupe_file.name = basenname
            test_data = {"child": {basenname: filePath}, basenname: dupe_file}

            expected = {"root": {"child": {basenname: ""}, basenname: ""}}

            virtual_fs = build_virtual_fs(test_data)

            organizer.promote_grandchildren(virtual_fs, "child", [basenname])
            result_tree = virtual_fs.get_folders_dict()

            assert result_tree == expected


class TestMoveFs:
    def setup_method(self):
        td = tempfile.TemporaryDirectory()

        td_path = Path(td.name)
        test_path = td_path / "testdir"
        test_path.mkdir(exist_ok=True, mode=0o777)

        testfile = Path(test_path, "test.txt")

        with open(testfile, "w") as tf:
            tf.write("test")

        input_structure = {"test.txt": testfile}

        virtual_fs = build_virtual_fs(input_structure, "testdir")
        virtual_fs.source_path = td_path

        self.virtual_fs = virtual_fs

        self.td = td
        self.testdir = td.name
        self.file_path = Path("testdir", "test.txt")
        self.testfile = testfile

    def teardown_method(self):
        try:
            self.td.cleanup()
        except:
            pass

    def test_move_file(self):
        # move to td/test2/test/test.txt
        outdir = Path(self.testdir, "test2")
        final_path = Path(self.testdir, "test2", "test.txt")

        organizer.move_fs(
            virtual_fs=self.virtual_fs,
            output_dir=outdir,
            should_execute=True,
            backup_state=FileBackupState.MOVE,
        )
        assert final_path.exists()
        assert not self.testfile.exists()
        assert not self.testfile.parent.exists()

    def test_copy_file(self):
        outdir = Path(self.testdir, "test2")
        final_path = Path(self.testdir, "test2", "test.txt")

        organizer.move_fs(
            virtual_fs=self.virtual_fs,
            output_dir=outdir,
            should_execute=True,
            backup_state=FileBackupState.COPY,
        )
        assert final_path.exists()
        assert self.testfile.exists()

    def test_move_in_place(self):
        outdir = Path(self.testdir, "test2")

        virtual_fs = self.virtual_fs
        virtual_fs.insert_intermediate_folder("intermediate")
        virtual_fs.name = "test2"
        final_path = Path(self.testdir, "intermediate", "test.txt")

        organizer.move_fs(
            virtual_fs=self.virtual_fs,
            output_dir=outdir,
            should_execute=True,
            backup_state=FileBackupState.IN_PLACE,
        )
        assert final_path.exists()
        assert not self.testfile.exists()
        assert not self.testfile.parent.exists()
