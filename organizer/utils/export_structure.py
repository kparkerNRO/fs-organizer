import json
from pathlib import Path
from sqlalchemy import select as sql_select
from storage.manager import StorageManager
from storage.index_models import Snapshot, Node


def export_snapshot_structure(
    output_path: Path,
    storage_path: Path | None = None,
    include_files: bool = False,
) -> dict:
    """
    Export the most recent snapshot's directory structure to a JSON file.

    Args:
        output_path: Path where the JSON file will be written
        storage_path: Storage directory containing index.db (None for default)
        include_files: Whether to include files in the structure (default: directories only)

    Returns:
        Dictionary containing the exported structure metadata

    Raises:
        ValueError: If no snapshots are found in the database
    """
    storage = StorageManager(storage_path)

    with storage.get_index_session(read_only=True) as session:
        # Get most recent snapshot
        snapshot = (
            session.execute(sql_select(Snapshot).order_by(Snapshot.created_at.desc()))
            .scalars()
            .first()
        )

        if not snapshot:
            raise ValueError("No snapshots found in database")

        # Query nodes, optionally filtering files
        query = sql_select(Node).where(Node.snapshot_id == snapshot.snapshot_id)
        if not include_files:
            query = query.where(Node.kind == "dir")

        nodes = list(session.execute(query.order_by(Node.rel_path)).scalars().all())

        # Build tree structure
        tree = _build_tree_structure(snapshot, nodes, include_files)

        # Write to JSON file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(tree, f, indent=2)

        return {
            "snapshot_id": snapshot.snapshot_id,
            "created_at": snapshot.created_at,
            "root_path": snapshot.root_abs_path,
            "total_nodes": len(nodes),
            "output_path": str(output_path),
        }


def _build_tree_structure(
    snapshot: Snapshot,
    nodes: list[Node],
    include_files: bool,
) -> dict:
    """
    Build hierarchical tree structure from flat list of nodes.

    Args:
        snapshot: The snapshot metadata
        nodes: Flat list of Node objects
        include_files: Whether files are included in the nodes

    Returns:
        Dictionary with tree structure
    """
    # Create children map
    children_map = {}
    root_nodes = []

    for node in nodes:
        if node.parent_node_id is None:
            root_nodes.append(node)
        else:
            if node.parent_node_id not in children_map:
                children_map[node.parent_node_id] = []
            children_map[node.parent_node_id].append(node)

    def node_to_dict(node: Node) -> dict:
        """Convert a Node to a dictionary representation."""
        result = {
            "name": node.name,
            "kind": node.kind,
            "rel_path": node.rel_path,
            "depth": node.depth,
        }

        if include_files and node.kind == "file":
            result.update(
                {
                    "ext": node.ext,
                    "size": node.size,
                    "file_source": node.file_source,
                }
            )

        # Add children recursively
        if node.node_id in children_map:
            result["children"] = [
                node_to_dict(child)
                for child in sorted(children_map[node.node_id], key=lambda n: n.name)
            ]
            result["num_children"] = len(children_map[node.node_id])

        return result

    # Build complete tree structure
    tree = {
        "snapshot_id": snapshot.snapshot_id,
        "created_at": snapshot.created_at,
        "root_path": snapshot.root_path,
        "root_abs_path": snapshot.root_abs_path,
        "include_files": include_files,
        "total_nodes": len(nodes),
        "structure": [
            node_to_dict(root) for root in sorted(root_nodes, key=lambda n: n.name)
        ],
    }

    return tree
