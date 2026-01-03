from fastapi.testclient import TestClient
from organizer_api import app

client = TestClient(app)

def test_find_shared_word_sequence_api():
    response = client.post(
        "/api/find_shared_word_sequence",
        json={"names": ["Gnome City Centre", "Goblin City Centre"]},
    )
    assert response.status_code == 200
    assert response.json() == {"shared_string": "City Centre"}

    response = client.post(
        "/api/find_shared_word_sequence",
        json={
            "names": [
                "Foundry Module",
                "Foundry Module CzepekuScenes CelestialGate",
                "Foundry Module CzepekuScenes TombOfSand",
            ]
        },
    )
    assert response.status_code == 200
    assert response.json() == {"shared_string": "Foundry Module"}

    response = client.post(
        "/api/find_shared_word_sequence",
        json={"names": ["alpha", "beta"]},
    )
    assert response.status_code == 200
    assert response.json() == {"shared_string": ""}

    response = client.post(
        "/api/find_shared_word_sequence",
        json={"names": []},
    )
    assert response.status_code == 200
    assert response.json() == {"shared_string": ""}

def test_find_longest_common_prefix_api():
    response = client.post(
        "/api/find_longest_common_prefix",
        json={"names": ["applewood", "applecart"]},
    )
    assert response.status_code == 200
    assert response.json() == {"shared_string": "apple"}

    response = client.post(
        "/api/find_longest_common_prefix",
        json={"names": ["testing", "test", "tester"]},
    )
    assert response.status_code == 200
    assert response.json() == {"shared_string": "test"}

    response = client.post(
        "/api/find_longest_common_prefix",
        json={"names": ["prefix-A", "prefix-B"]},
    )
    assert response.status_code == 200
    assert response.json() == {"shared_string": "prefix-"}

    response = client.post(
        "/api/find_longest_common_prefix",
        json={"names": ["nomatch", "completelydifferent"]},
    )
    assert response.status_code == 200
    assert response.json() == {"shared_string": ""}

    response = client.post(
        "/api/find_longest_common_prefix",
        json={"names": []},
    )
    assert response.status_code == 200
    assert response.json() == {"shared_string": ""}

    response = client.post(
        "/api/find_longest_common_prefix",
        json={"names": ["single"]},
    )
    assert response.status_code == 200
    assert response.json() == {"shared_string": "single"}
