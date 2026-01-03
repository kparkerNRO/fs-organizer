import pytest
from stages.grouping.group_cleanup import GroupEntry, unify_category_spelling


@pytest.mark.parametrize(
    "input_entries,expected_entries",
    [
        # Test case 1: Similar spellings should merge to the more common word
        (
            {
                "cat": [GroupEntry("cat", "cat", ["cat"])],
                "kat": [GroupEntry("kat", "kat", ["kat"])],
            },
            {
                "cat": [GroupEntry("cat", "cat", ["cat"], True, 0.6)],
                "kat": [GroupEntry("kat", "cat", ["cat"], True, 0.6)],
            },
        ),
        # Test case 2: Very different words should not merge
        (
            {
                "dog": [GroupEntry("dog", "dog", ["dog"])],
                "cat": [GroupEntry("cat", "cat", ["cat"])],
            },
            {
                "dog": [GroupEntry("dog", "dog", ["dog"])],
                "cat": [GroupEntry("cat", "cat", ["cat"])],
            },
        ),
        # Test case 3: Similar but both valid words should merge to higher frequency
        (
            {
                "color": [GroupEntry("color", "color", ["color"])],
                "colour": [GroupEntry("colour", "colour", ["colour"])],
            },
            {
                "color": [GroupEntry("color", "color", ["color"], True, 0.6)],
                "colour": [GroupEntry("colour", "color", ["color"], True, 0.6)],
            },
        ),
        # Test case 4: More than 2 similar words should all merge
        (
            {
                "photo": [GroupEntry("photo", "photo", ["photo"])],
                "foto": [GroupEntry("foto", "foto", ["foto"])],
                "photto": [GroupEntry("photto", "photto", ["photto"])],
            },
            {
                "photo": [GroupEntry("photo", "photo", ["photo"], True, 0.6)],
                "foto": [GroupEntry("foto", "photo", ["photo"], True, 0.6)],
                "photto": [GroupEntry("photto", "photo", ["photo"], True, 0.6)],
            },
        ),
    ],
)
def test_correct_spelling(input_entries, expected_entries):
    unify_category_spelling(input_entries)
    for name, entries in input_entries.items():
        entry = entries[0]
        expected_entry = expected_entries[name][0]
        assert entry.grouped_name == expected_entry.grouped_name
        assert entry.confidence == expected_entry.confidence
        assert entry.categories[0] == expected_entry.categories[0]
