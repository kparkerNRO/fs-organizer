import pytest
from .filename_utils import find_shared_word_sequence, find_longest_common_prefix

@pytest.mark.parametrize(
    "path_names, expected",
    [
        (
            ["Gnome City Centre", "Goblin City Centre"],
            "City Centre",
        ),
        (
            ["Foundry Module", "Foundry Module CzepekuScenes CelestialGate", "Foundry Module CzepekuScenes TombOfSand"],
            "Foundry Module",
        ),
        (
            ["Big Red Dragon", "Small Red Dragon", "Ancient Red Dragon"],
            "Red Dragon",
        ),
        (
            ["Forest Temple Ancient", "Mountain Forest Temple", "Desert Forest Temple Ruins"],
            "Forest Temple",
        ),
        (
            ["alpha", "beta"],
            "",
        ),
        (
            [],
            "",
        ),
        (
            ["single string"],
            "",
        ),
    ],
)
def test_find_shared_word_sequence(path_names, expected):
    assert find_shared_word_sequence(path_names) == expected

@pytest.mark.parametrize(
    "names, expected",
    [
        (["applewood", "applecart"], "apple"),
        (["testing", "test", "tester"], "test"),
        (["prefix-A", "prefix-B"], "prefix-"),
        (["nomatch", "completelydifferent"], ""),
        (["same", "same", "same"], "same"),
        ([], ""),
        (["single"], "single"),
        (["item1", "item2", "item3"], "item"),
    ],
)
def test_find_longest_common_prefix(names, expected):
    assert find_longest_common_prefix(names) == expected
