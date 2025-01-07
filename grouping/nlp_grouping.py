from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pathlib import Path
from database import (
    get_session,
    Folder,
    FolderCategory,
    GroupRecord,
    setup_group,
)
from grouping.helpers import ClassificationType

import numpy as np
from sklearn.cluster import AgglomerativeClustering


@dataclass
class CustomDistanceFolder:
    folder_id: int
    category_id: int
    name: str
    text_vec: np.ndarray
    depth: int
    path: Path


def compute_distance_to_shared_parent(A_path: Path, B_path: Path) -> int:
    """
    Compute the distance between two paths by counting how far each is from the shared parent.
    """
    i = 0
    while (
        i < len(A_path.parts)
        and i < len(B_path.parts)
        and A_path.parts[i] == B_path.parts[i]
    ):
        i += 1
    # i is now how many common segments they share
    # The distance "steps up" might be:
    # (len(A_path) - i) + (len(B_path) - i)
    return (len(A_path.parts) - i) + (len(B_path.parts) - i)


def compute_custom_distance_matrix(folders: list[CustomDistanceFolder]) -> np.ndarray:
    """
    folders: list of dicts with keys:
      - 'id'
      - 'text_vec': numpy array or embedding for textual representation
      - 'depth': integer
    Returns an (n x n) distance matrix combining text + structural proximity.
    """
    n = len(folders)
    D = np.zeros((n, n), dtype=float)

    alpha = 0.5  # weight for text distance vs structural distance

    for i in range(n):
        for j in range(i + 1, n):
            text_vec_i = folders[i].text_vec  # e.g., embedding
            text_vec_j = folders[j].text_vec
            # text distance = 1 - cos sim
            text_dist = 1.0 - np.dot(text_vec_i, text_vec_j) / (
                np.linalg.norm(text_vec_i) * np.linalg.norm(text_vec_j) + 1e-8
            )

            # structural distance = difference in depth
            struct_dist = compute_distance_to_shared_parent(
                folders[i].path, folders[j].path
            )

            # combine
            dist = alpha * text_dist + (1 - alpha) * (struct_dist / (1 + struct_dist))
            D[i, j] = dist
            D[j, i] = dist

    return D


def cluster_with_custom_metric(db_path):
    setup_group(db_path)
    session = get_session(db_path)
    uncertain_categories = (
        session.query(FolderCategory, Folder)
        .join(Folder, FolderCategory.folder_id == Folder.id)
        .filter(FolderCategory.classification != ClassificationType.VARIANT)
        .all()
    )

    corpus = [category[0].name for category in uncertain_categories]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit_transform(corpus)

    folders = [
        CustomDistanceFolder(
            folder_id=uc[1].id,
            category_id=uc[0].id,
            name=uc[0].name,
            text_vec=vectorizer.transform([uc[0].name]).toarray()[0],
            depth=uc[1].depth,
            path=Path(uc[1].folder_path),
        )
        for uc in uncertain_categories
    ]
    # Step 1: build distance matrix
    dist_matrix = compute_custom_distance_matrix(folders)

    # Step 2: run Agglomerative (or DBSCAN) with precomputed distance
    clusterer = AgglomerativeClustering(
        n_clusters=None,  # or some fixed number
        distance_threshold=.65,  # experiment with a lower threshold
        metric="precomputed",
        linkage="average",
    )
    labels = clusterer.fit_predict(dist_matrix)

    # Step 3: assign cluster labels
    for i, f in enumerate(folders):
        group_record = GroupRecord(
            folder_id=f.folder_id,
            category_id=f.category_id,
            group_name=labels[i],
            cannonical_name=f.name,
            path = str(f.path),
        )
        session.add(group_record)
    session.commit()
    session.close()



def group_uncertain(db_path: Path):
    """Group uncertain categories using KMeans clustering."""
    setup_group(db_path)
    session = get_session(db_path)
    uncertain_categories = (
        session.query(FolderCategory, Folder)
        .join(Folder, FolderCategory.folder_id == Folder.id)
        .filter(FolderCategory.classification != ClassificationType.VARIANT)
        .all()
    )
    n_clusters = 20

    corpus = [category[0].name for category in uncertain_categories]

    # Vectorize the category names
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    # vectorizer.transform([folder_name]).toarray()[0]

    # Cluster the categories
    km = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = km.fit_predict(X)

    for i, uf in enumerate(uncertain_categories):
        cluster_id = clusters[i]
        group_record = GroupRecord(
            folder_id=uf[1].id,
            category_id=uf[0].id,
            group_name=f"cluster_{cluster_id}",
            cannonical_name=uf[0].name,
        )
        session.add(group_record)
        # uf.classification = f"cluster_{cluster_id}"

    session.commit()
    session.close()
