
import re
import sqlite3
from pathlib import Path
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from utils.filename_utils import get_max_common_string, strip_part_from_base
from nltk.corpus import stopwords
from nltk.metrics import edit_distance

from utils.config import KNOWN_VARIANT_TOKENS

nltk.download("punkt_tab")
nltk.download("stopwords")


STOPWORDS = set(stopwords.words("english"))



# def parse_name(name: str) -> tuple[list[str], list[str]]:
#     """
#     Parse out known names and suspected components
#     """
#     categories = []
#     variants = []

#     clean_row = clean_filename(name)
#     components = clean_row.split("-")
#     cleaned_components = [clean_filename(component) for component in components]

#     for component in cleaned_components:
#         category_current = component
#         while category_current:
#             category, variant = split_view_type(category_current, KNOWN_VARIANT_TOKENS)
#             if category and category not in categories:
#                 if category in KNOWN_VARIANT_TOKENS:
#                     variants.append(category)
#                 else:
#                     categories.append(category)
#             if variant:
#                 variants.append(variant)

#             if category_current == category:
#                 break

#             category_current = category

#     return categories, variants


# def heuristic_categorize(session) -> None:
#     """
#     Break the filenames into tokens, and identify known variants and suspected categories.
#     Clean up the tokens, and assign the folder name to the first category or variant

#     Fills out the "partial name category"
#     """

#     # Get all folders
#     folders = session.query(Folder).all()

#     for folder in folders:
#         name = folder.folder_name

#         if name.endswith(".zip"):
#             categories, variants = parse_name(name[:-4])
#         else:
#             categories, variants = parse_name(name)

#         variants = list(set(variants))
#         cleaned_name = categories[0] if categories else variants[0] if variants else ""

#         # Update the folder
#         folder.cleaned_name = cleaned_name
#         folder.variants = variants
#         folder.categories = categories

#         if len(categories) == 0 and len(variants) > 0:
#             folder.classification = ClassificationType.VARIANT
#         elif len(categories) == 1 and len(variants) == 0:
#             folder.classification = ClassificationType.CATEGORY
#         else:
#             folder.classification = ClassificationType.UNKNOWN

#         for category in categories:
#             category_lookup = PartialNameCategory(
#                 folder_id=folder.id,
#                 original_name=folder.folder_name,
#                 name=category,
#                 classification=folder.classification,
#             )
#             session.add(category_lookup)

#         for variant in variants:
#             category_lookup = PartialNameCategory(
#                 folder_id=folder.id,
#                 original_name=folder.folder_name,
#                 name=variant,
#                 classification=ClassificationType.VARIANT,
#             )
#             session.add(category_lookup)
#     session.commit()

#     non_variant_folders = (
#         session.query(Folder)
#         .filter(Folder.classification != ClassificationType.VARIANT)
#         .all()
#     )
#     # for each of these folders, check if
#     # (a) there is only one category and (b) that all sub-folders are classified as variant
#     for folder in non_variant_folders:
#         if len(folder.categories) == 1:
#             children = (
#                 session.query(Folder)
#                 .filter(Folder.parent_path.startswith(folder.folder_path))
#                 .all()
#             )
#             if len(children) == 0 or all(
#                 child.classification == ClassificationType.VARIANT for child in children
#             ):
#                 folder.classification = ClassificationType.SUBJECT
#                 folder.subject = folder.categories[0]
#                 # also update the FolderCategory reference for folder.categories[0]
#                 category_lookup = (
#                     session.query(PartialNameCategory)
#                     .filter_by(folder_id=folder.id, name=folder.categories[0])
#                     .one()
#                 )
#                 category_lookup.classification = ClassificationType.SUBJECT
#     session.commit()

def classify_folders(db_path: Path):
    """
    Classify each folder in the 'folders' table using:
      1) Known variant detection
      2) If not variant, check frequency + children
    Updates the 'classification' column in 'folders' table.
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    rows = cur.execute(
        "SELECT id, folder_name, folder_path, parent_path, depth FROM folders ORDER BY depth DESC"
    ).fetchall()

    # Precompute the global frequency of each folder_name at or above the same depth.
    freq_map = {}
    freq_rows = cur.execute("SELECT folder_name, count FROM counts").fetchall()
    for fname, cval in freq_rows:
        freq_map[fname] = cval

    def children_are_all_variant(folder_path: str) -> bool:
        """
        Return True if all direct children of folder_path are classified as variant
        (or if it has no children).
        """
        # Find children
        child_rows = cur.execute(
            """
            SELECT classification FROM folders WHERE parent_path = ?
        """,
            (folder_path,),
        ).fetchall()

        # If no children, we'll return False for "all variant" (or maybe True if you want).
        if not child_rows:
            return False

        for (child_class,) in child_rows:
            if child_class != "variant":
                return False
        return True

    # Classification pass
    for row in rows:
        folder_id, folder_name, folder_path, parent_path, depth = row

        # 1) Known Variant Detection
        folder_name_stripped = re.sub(r"[^\w\s]", "", folder_name)
        tokens = folder_name_stripped.lower().split()
        # If *all* tokens are in known_variant_tokens => classify as variant
        if all(t in KNOWN_VARIANT_TOKENS for t in tokens):
            classification = "variant"

        else:
            # Not a known variant => check frequency
            freq_val = freq_map.get(folder_name, 1)  # default=1 if not found
            if freq_val > 1:
                # suspect collection
                classification = "collection"
            elif freq_val == 1:
                # possibly a subject => check if children are variant
                if children_are_all_variant(folder_path):
                    classification = "subject"
                else:
                    classification = "uncertain"
            else:
                classification = "uncertain"  # fallback

        # Update the DB
        cur.execute(
            """
            UPDATE folders
            SET classification = ?
            WHERE id = ?
        """,
            (classification, folder_id),
        )

    conn.commit()
    conn.close()
