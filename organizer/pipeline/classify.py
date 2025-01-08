import difflib
import os
import json
import re
import sqlite3
from pathlib import Path
from rapidfuzz import fuzz as rapidfuzz_fuzz
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


########### Initial pass ############
def classify_folders(db_path: Path):
    """
    Classify each folder in the 'folders' table using:
      1) Known variant detection
      2) If not variant, check frequency + children
    Updates the 'classification' column in 'folders' table.
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # We might want to do a multi-pass approach, or we can sort folders by descending depth
    # if we want "lowest-level first".
    # For example:
    rows = cur.execute(
        "SELECT id, folder_name, folder_path, parent_path, depth FROM folders ORDER BY depth DESC"
    ).fetchall()

    # Precompute the global frequency of each folder_name at or above the same depth.
    # But we can also rely on the 'counts' table if that suits your logic. We'll keep it simple here:
    # We'll store them in a dictionary: {folder_name: freq_count}
    freq_map = {}
    freq_rows = cur.execute("SELECT folder_name, count FROM counts").fetchall()
    for fname, cval in freq_rows:
        freq_map[fname] = cval

    # We'll define a helper to see if children are variant:
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
        tokens = (
            folder_name_stripped.lower().split()
        )  # simplistic tokenizing on whitespace
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


