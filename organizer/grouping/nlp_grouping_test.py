import pytest
from pathlib import Path
from organizer.pipeline.nlp_grouping import compute_distance_to_shared_parent

@pytest.mark.parametrize("path1, path2, expected", [
    (Path("/a/b/c"), Path("/a/b/c"), 0),
    (Path("/a/b/c"), Path("/a/b/d"), 2),
    (Path("/a/b/c"), Path("/a/x/y"), 4),
    (Path("/a/b/c"), Path("/x/y/z"), 6),
    (Path("/"), Path("/"), 0),
    (Path(""), Path("/a/b/c"), 3),
    (Path("/a/b/c"), Path(""), 3),
    (Path(""), Path(""), 0),
])
def test_compute_distance_to_shared_parent(path1, path2, expected):
    assert compute_distance_to_shared_parent(path1, path2) == expected
