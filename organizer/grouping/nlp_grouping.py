from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from data_models.database import (
    Folder,
    GroupCategoryEntry,
)

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


TEXT_DISTANCE_RATIO = 0.7
DISTANCE_THRESHOLD = 0.6


@dataclass
class ClusterItem:
    folder_id: int
    partial_category_id: int
    name: str
    original_name: str
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


def compute_custom_distance_matrix(folders: list[ClusterItem], text_distance_ratio=TEXT_DISTANCE_RATIO) -> np.ndarray:
    """
    folders: list of dicts with keys:
      - 'id'
      - 'text_vec': numpy array or embedding for textual representation
      - 'depth': integer
    Returns an (n x n) distance matrix combining text + structural proximity.
    """
    n = len(folders)
    D = np.zeros((n, n), dtype=float)

    alpha = text_distance_ratio

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


def prepare_records(
    previous_round_groups: list[tuple[GroupCategoryEntry, Folder]],
):
    corpus = [category[0].processed_name for category in previous_round_groups]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit_transform(corpus)

    items = [
        ClusterItem(
            folder_id=pair[1].id,
            partial_category_id=pair[0].id,
            name=pair[0].processed_name,
            original_name=pair[0].pre_processed_name,
            text_vec=vectorizer.transform([pair[0].processed_name]).toarray()[0],
            depth=pair[1].depth or 1,
            path=Path(pair[1].folder_path) if pair[1] else Path(),
        )
        for pair in previous_round_groups
    ]

    return items


def cluster_with_custom_metric(cluster_items, iteration_id, text_distance_ratio=0) -> list[GroupCategoryEntry]:
    """
    Cluster the calculated categories into groups using a custom distance metric.
    """

    # Step 1: build distance matrix
    dist_matrix = compute_custom_distance_matrix(cluster_items, text_distance_ratio)

    # Step 2: run Agglomerative (or DBSCAN) with precomputed distance
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        metric="precomputed",
        linkage="average",
    )
    labels = clusterer.fit_predict(dist_matrix)

    # Calculate confidence scores
    cluster_confidence = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_indices = [id for id, label in enumerate(labels) if label == label]

        if len(cluster_indices) > 1:
            # Calculate average distance to other items in cluster
            distances = [dist_matrix[i, j] for j in cluster_indices if j != i]
            avg_distance = sum(distances) / len(distances)
            confidence = max(0.0, 1.0 - avg_distance)
        else:
            # Single item clusters get lower confidence
            confidence = 0.5

        cluster_confidence[label].append((i, confidence))

    # Create GroupCategoryEntry objects
    entries = []
    for i, item in enumerate(cluster_items):
        label = int(labels[i])
        confidence = next(conf for idx, conf in cluster_confidence[label] if idx == i)

        entry = GroupCategoryEntry(
            folder_id=item.folder_id,
            partial_category_id=item.partial_category_id,
            group_id=None,  # Will be set during refinement
            pre_processed_name=item.name,
            # name=item.name,
            path=str(item.path),
            confidence=0.8,
            # confidence=confidence,
            processed=False,
            derived_names=[item.name],
            iteration_id=iteration_id,
        )
        entry.cluster_id = label  # Temporary attribute for refinement
        entries.append(entry)

    return entries


# def group_uncertain(db_path: Path):
#     """Group uncertain categories using KMeans clustering."""
#     session = get_session(db_path)
#     uncertain_categories = (
#         session.query(PartialNameCategory, Folder)
#         .join(Folder, PartialNameCategory.folder_id == Folder.id)
#         .filter(PartialNameCategory.classification != ClassificationType.VARIANT)
#         .all()
#     )
#     n_clusters = 20

#     corpus = [category[0].name for category in uncertain_categories]

#     # Vectorize the category names
#     vectorizer = TfidfVectorizer(stop_words="english")
#     X = vectorizer.fit_transform(corpus)
#     # vectorizer.transform([folder_name]).toarray()[0]

#     # Cluster the categories
#     km = KMeans(n_clusters=n_clusters, random_state=42)
#     clusters = km.fit_predict(X)

#     for i, uf in enumerate(uncertain_categories):
#         cluster_id = clusters[i]
#         group_record = GroupCategoryEntry(
#             folder_id=uf[1].id,
#             category_id=uf[0].id,
#             group_name=f"cluster_{cluster_id}",
#             original_name=uf[0].name,
#         )
#         session.add(group_record)
#         # uf.classification = f"cluster_{cluster_id}"

#     session.commit()
#     session.close()
