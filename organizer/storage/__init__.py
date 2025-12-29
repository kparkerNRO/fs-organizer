"""Storage module for filesystem index and intermediary work databases.

This module implements the two-database architecture:
1. Filesystem index (data/index/index.db) - Immutable snapshots of filesystem
2. Intermediary work (data/work/work.db) - Pipeline processing keyed by snapshot_id + run_id

Configuration data is managed separately via YAML files in organizer/config/
(see organizer/utils/config.py).
"""

from storage.manager import StorageManager

__all__ = ["StorageManager"]
