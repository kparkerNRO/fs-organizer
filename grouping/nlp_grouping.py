from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pathlib import Path
from database import (
    get_session,
    Folder,
    FolderCategory,
    GroupRecord,
    Category,
    setup_group,
)
from grouping.helpers import ClassificationType


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
