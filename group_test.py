import unittest
import pytest
from group import common_token_grouping


@pytest.mark.parametrize(
    "name, input_list, expected",
    [
        ("empty_list", [], None),
        ("single_name", ["example"], {}),
        (
            "multi_word_overlap",
            ["mom's apple pie", "mom's apple tart"],
            {
                "mom's apple pie": ["mom's apple", "pie"],
                "mom's apple tart": ["mom's apple", "tart"],
            },
        ),
        ("no_common_tokens", ["apple", "banana", "cherry"], {}),
        (
            "common_prefix",
            ["apple pie", "apple tart", "apple juice"],
            {
                "apple pie": ["apple", "pie"],
                "apple tart": ["apple", "tart"],
                "apple juice": ["apple", "juice"],
            },
        ),
        (
            "common_suffix",
            ["pie apple", "tart apple", "juice apple"],
            {},
        ),
        (
            "mixed_case",
            ["Apple Pie", "apple tart", "APPLE juice"],
            {
                "Apple Pie": ["Apple", "Pie"],
                "apple tart": ["apple", "tart"],
                "APPLE juice": ["apple", "juice"],
            },
        ),
        (
            "numbers_in_names",
            ["apple 1", "apple 2", "apple 3"],
            {},
        ),
        (
            "partial_overlap",
            ["apple pie", "apple tart", "banana pie"],
            {"apple pie": ["apple", "pie"], "apple tart": ["apple", "tart"]},
        ),
        (
            "overlapping_groups",
            [
                "A wild two",
                "A wild four",
                "A wild test one",
                "A wild test three",
            ],
            {
                "A wild test one": ["A wild", "test one"],
                "A wild test three": ["A wild", "test three"],
                "A wild two": ["A wild", "two"],
                "A wild four": ["A wild", "four"],
            },
        ),
        (
            "overlapping_groups_inverted",
            [
                "A wild test one",
                "A wild test three",
                "A wild two",
                "A wild four",
            ],
            {
                "A wild test one": ["A wild", "test one"],
                "A wild test three": ["A wild", "test three"],
                "A wild two": ["A wild", "two"],
                "A wild four": ["A wild", "four"],
            },
        ),
    ],
)
def test_common_token_grouping(name, input_list, expected):
    result = common_token_grouping(input_list)
    assert result == expected
