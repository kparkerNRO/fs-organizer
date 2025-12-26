import argparse
import random
import sqlite3
from pathlib import Path
from typing import List, Tuple


def ensure_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS classification (
            file_path TEXT PRIMARY KEY,
            leaf TEXT NOT NULL,
            true_classification TEXT,
            predicted_classification TEXT,
            grouping TEXT
        )
        """
    )


def load_folders(cursor: sqlite3.Cursor) -> None:
    rows = cursor.execute(
        """
        SELECT
            cleaned_path,
            folder_path,
            cleaned_name,
            folder_name,
            classification
        FROM folders
        """
    ).fetchall()
    prepared_rows = []
    for cleaned_path, folder_path, cleaned_name, folder_name, classification in rows:
        file_path = cleaned_path or folder_path
        if cleaned_path:
            leaf = Path(cleaned_path).name
        else:
            leaf = cleaned_name or folder_name
        prepared_rows.append((file_path, leaf, classification))

    cursor.executemany(
        """
        INSERT INTO classification (
            file_path,
            leaf,
            true_classification,
            predicted_classification,
            grouping
        )
        VALUES (?, ?, ?, NULL, NULL)
        ON CONFLICT(file_path) DO UPDATE SET
            leaf = excluded.leaf,
            true_classification = excluded.true_classification,
            predicted_classification = excluded.predicted_classification,
            grouping = excluded.grouping
        """,
        prepared_rows,
    )


def split_groupings(
    rows: List[Tuple[str]],
    seed: int,
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    paths = [r[0] for r in rows]
    rng.shuffle(paths)

    total = len(paths)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    train_paths = paths[:train_count]
    val_paths = paths[train_count : train_count + val_count]
    test_paths = paths[train_count + val_count : train_count + val_count + test_count]
    return train_paths, test_paths, val_paths


def apply_grouping(cursor: sqlite3.Cursor, grouping: str, paths: List[str]) -> None:
    cursor.executemany(
        "UPDATE classification SET grouping = ? WHERE file_path = ?",
        [(grouping, p) for p in paths],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create/refresh the classification table from folders and assign splits."
        )
    )
    parser.add_argument("db_path", help="Path to the organizer sqlite database")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.test_ratio + args.val_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    db_path = Path(args.db_path)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        ensure_table(cursor)
        load_folders(cursor)

        rows = cursor.execute("SELECT file_path FROM classification").fetchall()
        train_paths, test_paths, val_paths = split_groupings(
            rows,
            seed=args.seed,
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
        )

        apply_grouping(cursor, "train", train_paths)
        apply_grouping(cursor, "test", test_paths)
        apply_grouping(cursor, "validation", val_paths)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
