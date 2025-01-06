from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pathlib import Path
from database import (
    setup_group,
    get_session,
    setup_category_summarization,
    Folder,
    FolderCategory,
    Group,
    ProcessedName,
    Category,
)
from grouping.helpers import ClassificationType


def group_uncertain(db_path: Path):
    """Group uncertain categories using KMeans clustering."""
    session = get_session(db_path)
    uncertain_categories = (
        session.query(Category)
        .filter(Category.classification != ClassificationType.VARIANT)
        .all()
    )
    n_clusters = 20

    corpus = [category.category_name for category in uncertain_categories]

    # Vectorize the category names
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)

    # Cluster the categories
    km = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = km.fit_predict(X)

    for i, uf in enumerate(uncertain_categories):
        cluster_id = clusters[i]
        uf.classification_counts = uf.classification_counts | {
            f"cluster_{cluster_id}": 1
        }
        uf.classification = f"cluster_{cluster_id}"

    session.commit()
    session.close()
