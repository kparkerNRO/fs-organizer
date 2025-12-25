# Implementation Plan: Filesystem Index & Intermediary Pipeline Stages

## Overview

Implement the two-database storage architecture: separate filesystem indexing (index.db) from pipeline processing work (work.db). Configuration remains in YAML files, loaded in-memory via config.py.

**Current State**: Single `run_data.db` created per gather run containing both raw filesystem data and processing results.

**Target State**: Two-database architecture with in-memory config:
1. **Configuration data** (`organizer/config/*.yaml`) - ✅ Already implemented - loaded in-memory via config.py
2. **Filesystem index** (`data/index/index.db`) - 🔨 To implement - immutable snapshots of filesystem
3. **Intermediary work** (`data/work/work.db`) - 🔨 To implement - pipeline processing keyed by snapshot_id + run_id

**Key Benefits**:
- Separate concerns: raw data vs. derived results
- Reproducibility via snapshot versioning
- Multiple experimental runs on same snapshot
- Efficient reset/rerun of pipeline stages

---

## Phase 1: Storage Infrastructure

### 1.1 Create Storage Module Structure

**New directory**: `/Users/mantrasong/dev/fs-organizer/organizer/storage/`

**Files to create**:
```
organizer/storage/
├── __init__.py
├── manager.py           # Core StorageManager class
├── index_models.py      # SQLAlchemy models for index.db
└── work_models.py       # SQLAlchemy models for work.db
```

### 1.2 Define Database Schemas with SQLAlchemy

#### index.db Models (index_models.py)

**File**: `/Users/mantrasong/dev/fs-organizer/organizer/storage/index_models.py`

```python
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, CheckConstraint, Index
from sqlalchemy.orm import declarative_base, relationship

IndexBase = declarative_base()

class Snapshot(IndexBase):
    __tablename__ = 'snapshot'

    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(String, nullable=False)
    root_path = Column(String, nullable=False)
    root_abs_path = Column(String, nullable=False)
    preprocess_version = Column(String)
    preprocess_hash = Column(String)
    reference_hash = Column(String)
    notes = Column(String)

    nodes = relationship("Node", back_populates="snapshot", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_snapshot_root', 'root_abs_path'),
    )

class Node(IndexBase):
    __tablename__ = 'node'

    node_id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, ForeignKey('snapshot.snapshot_id'), nullable=False)
    parent_node_id = Column(Integer, ForeignKey('node.node_id'), nullable=True)
    kind = Column(String, CheckConstraint("kind IN ('file', 'dir')"), nullable=False)
    name = Column(String, nullable=False)
    rel_path = Column(String, nullable=False)
    abs_path = Column(String, nullable=False)
    ext = Column(String)
    size = Column(Integer)
    mtime = Column(Float)
    ctime = Column(Float)
    inode = Column(Integer)
    depth = Column(Integer, nullable=False)
    file_source = Column(String, nullable=False, default='filesystem')  # 'filesystem' | 'zip_file' | 'zip_content'
    num_folder_children = Column(Integer, default=0)
    num_file_children = Column(Integer, default=0)

    snapshot = relationship("Snapshot", back_populates="nodes")
    parent = relationship("Node", remote_side=[node_id], backref="children")
    features = relationship("NodeFeatures", back_populates="node", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_node_snapshot', 'snapshot_id'),
        Index('idx_node_parent', 'snapshot_id', 'parent_node_id'),
        Index('idx_node_kind', 'snapshot_id', 'kind'),
        Index('idx_node_path', 'snapshot_id', 'rel_path', 'file_source', unique=True),
    )

class NodeFeatures(IndexBase):
    __tablename__ = 'node_features'

    node_id = Column(Integer, ForeignKey('node.node_id'), primary_key=True)
    normalized_name = Column(String)
    tokens_json = Column(String)  # JSON array
    hints_json = Column(String)   # JSON object

    node = relationship("Node", back_populates="features")

class Meta(IndexBase):
    __tablename__ = 'meta'

    key = Column(String, primary_key=True)
    value = Column(String)

# Schema version constant (increment on breaking changes)
INDEX_SCHEMA_VERSION = "1.0.0"
```

**Schema versioning**:
- Store schema version in Meta table with key `'schema_version'`
- On initialization, check version matches `INDEX_SCHEMA_VERSION`
- If mismatch, raise error requiring re-ingestion
- Increment version on any breaking schema changes

**Path namespace and uniqueness**:
- **Problem**: ZIP files can create path collisions:
  - Real file: `a.zip/b.txt` (if directory `a.zip` exists)
  - ZIP content: `a.zip/b.txt` (file `b.txt` inside `a.zip`)
- **Solution**: Unique constraint on `(snapshot_id, rel_path, file_source)`
  - Same `rel_path` allowed if `file_source` differs
  - `file_source` values:
    - `'filesystem'` - Real files/directories on disk
    - `'zip_file'` - The ZIP archive file itself
    - `'zip_content'` - Files/directories inside ZIP archives
- **Examples**:
  - Real directory `a.zip` containing `b.txt`:
    - `rel_path='a.zip'`, `file_source='filesystem'`, `kind='dir'`
    - `rel_path='a.zip/b.txt'`, `file_source='filesystem'`, `kind='file'`
  - ZIP file `a.zip` containing `b.txt`:
    - `rel_path='a.zip'`, `file_source='zip_file'`, `kind='file'`
    - `rel_path='a.zip/b.txt'`, `file_source='zip_content'`, `kind='file'`
  - Nested ZIP (`data.zip` contains `nested.zip` contains `file.txt`):
    - `rel_path='data.zip'`, `file_source='zip_file'`, `kind='file'`
    - `rel_path='data.zip/nested.zip'`, `file_source='zip_content'`, `kind='file'`
    - `rel_path='data.zip/nested.zip/file.txt'`, `file_source='zip_content'`, `kind='file'`
- **Query pattern**: Always filter by both `rel_path` AND `file_source` when looking up specific nodes

**ZIP parent modeling**:
- **Problem**: ZIP content nodes have `parent_node_id` pointing to the ZIP file node itself (which has `kind='file'`), breaking traversal logic that assumes all parents are directories.
- **Solution**: Treat ZIP files as **virtual directories** for hierarchy purposes:
  - ZIP file node: `rel_path='a.zip'`, `file_source='zip_file'`, `kind='file'`, `parent_node_id=<actual_parent_dir>`
  - ZIP content nodes: `rel_path='a.zip/b.txt'`, `file_source='zip_content'`, `kind='file'`, `parent_node_id=<zip_file_node_id>`
  - **Traversal rule**: When walking hierarchy, treat nodes with `file_source='zip_file'` as container nodes (similar to directories) for their children
  - **Query pattern**: To get children of a ZIP: `filter(Node.parent_node_id == zip_file_node.node_id, Node.file_source == 'zip_content')`
- **Alternative considered and rejected**: Creating synthetic directory nodes for ZIPs adds complexity and duplicate entries

**Schema mapping from current run_data.db**:
- `folders` table → `Node` (kind='dir')
- `files` table → `Node` (kind='file')
- `cleaned_name` → `NodeFeatures.normalized_name`
- Classification/categories → **MOVE to work.db**

#### work.db Models (work_models.py)

**File**: `/Users/mantrasong/dev/fs-organizer/organizer/storage/work_models.py`

```python
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship

WorkBase = declarative_base()

class Run(WorkBase):
    __tablename__ = 'run'

    run_id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, nullable=False)  # FK to index.db (cross-database)
    started_at = Column(String, nullable=False)
    finished_at = Column(String)
    status = Column(String, default='running')  # 'running' | 'completed' | 'failed' | 'cancelled'
    pipeline_version = Column(String)
    config_hash = Column(String)
    model_id = Column(String)
    notes = Column(String)

    stages = relationship("StageState", back_populates="run", cascade="all, delete-orphan")
    group_iterations = relationship("GroupIteration", back_populates="run", cascade="all, delete-orphan")

class StageState(WorkBase):
    __tablename__ = 'stage_state'

    stage_name = Column(String, primary_key=True)
    snapshot_id = Column(Integer, primary_key=True)  # Note: Redundant with run.snapshot_id, but kept in PK for query performance
    run_id = Column(Integer, ForeignKey('run.run_id'), primary_key=True)
    completed_at = Column(String)
    input_fingerprint = Column(String)
    output_fingerprint = Column(String)

    run = relationship("Run", back_populates="stages")

    # IMPORTANT: snapshot_id must be validated against run.snapshot_id in mark_stage_complete()

class GroupIteration(WorkBase):
    __tablename__ = 'stg_group_iteration'

    iteration_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('run.run_id'), nullable=False)
    snapshot_id = Column(Integer, nullable=False)  # Note: Redundant with run.snapshot_id, but useful for queries
    timestamp = Column(String)
    description = Column(String)
    parameters_json = Column(String)

    run = relationship("Run", back_populates="group_iterations")
    entries = relationship("GroupEntry", back_populates="iteration", cascade="all, delete-orphan")
    categories = relationship("GroupCategory", back_populates="iteration", cascade="all, delete-orphan")

    # IMPORTANT: snapshot_id must be validated against run.snapshot_id in create_group_iteration()

class GroupEntry(WorkBase):
    __tablename__ = 'stg_group_entry'

    entry_id = Column(Integer, primary_key=True, autoincrement=True)
    iteration_id = Column(Integer, ForeignKey('stg_group_iteration.iteration_id'), nullable=False)
    node_id = Column(Integer, nullable=False)  # FK to index.db node
    cluster_id = Column(Integer)
    pre_processed_name = Column(String)
    processed_name = Column(String)
    derived_names_json = Column(String)
    confidence = Column(Float, default=1.0)
    processed = Column(Boolean, default=False)

    iteration = relationship("GroupIteration", back_populates="entries")

    __table_args__ = (
        Index('idx_group_entry_iter', 'iteration_id'),
        Index('idx_group_entry_node', 'node_id'),
    )

class GroupCategory(WorkBase):
    __tablename__ = 'stg_group_category'

    group_id = Column(Integer, primary_key=True, autoincrement=True)
    iteration_id = Column(Integer, ForeignKey('stg_group_iteration.iteration_id'), nullable=False)
    name = Column(String, nullable=False)
    count = Column(Integer)
    group_confidence = Column(Float)
    needs_review = Column(Boolean, default=False)
    reviewed = Column(Boolean, default=False)

    iteration = relationship("GroupIteration", back_populates="categories")

    __table_args__ = (
        Index('idx_group_category_iter', 'iteration_id'),
    )

class Classification(WorkBase):
    __tablename__ = 'stg_classification'

    classification_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('run.run_id'), nullable=False)
    node_id = Column(Integer, nullable=False)
    classification = Column(String)  # 'variant' | 'collection' | 'subject' | 'uncertain'
    confidence = Column(Float)
    method = Column(String)  # 'structural' | 'llm' | 'manual'

    __table_args__ = (
        Index('idx_classification_run', 'run_id'),
        Index('idx_classification_node', 'node_id'),
        Index('idx_classification_run_node', 'run_id', 'node_id', unique=True),
    )

class FolderStructure(WorkBase):
    __tablename__ = 'out_folder_structure'

    structure_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('run.run_id'), nullable=False)
    structure_type = Column(String, nullable=False)  # 'original' | 'organized' | 'grouped'
    structure_json = Column(String, nullable=False)
    created_at = Column(String)

class FileMapping(WorkBase):
    __tablename__ = 'out_file_mapping'

    mapping_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('run.run_id'), nullable=False)
    node_id = Column(Integer, nullable=False)
    original_path = Column(String, nullable=False)
    new_path = Column(String)
    groups_json = Column(String)

    __table_args__ = (
        Index('idx_file_mapping_run', 'run_id'),
        Index('idx_file_mapping_node', 'node_id'),
        Index('idx_file_mapping_run_node', 'run_id', 'node_id', unique=True),
    )

class Meta(WorkBase):
    __tablename__ = 'meta'

    key = Column(String, primary_key=True)
    value = Column(String)

# Schema version constant (increment on breaking changes)
WORK_SCHEMA_VERSION = "1.0.0"
```

**Schema versioning**:
- Store schema version in Meta table with key `'schema_version'`
- On initialization, check version matches `WORK_SCHEMA_VERSION`
- If mismatch, raise error requiring database deletion/recreation
- Increment version on any breaking schema changes

### 1.3 Implement StorageManager

**File**: `/Users/mantrasong/dev/fs-organizer/organizer/storage/manager.py`

**Class**: `StorageManager`

**Key methods**:
```python
# Database initialization
__init__(index_path: Path = DATA_DIR/"index"/"index.db",
         work_path: Path = DATA_DIR/"work"/"work.db")
_ensure_databases()
_init_index_schema()
_init_work_schema()

# Filesystem indexing
ingest_filesystem(root_path: Path, preprocess_version: str,
                  notes: Optional[str]) -> int  # returns snapshot_id

# Run management
start_run(snapshot_id: int, pipeline_version: str,
          notes: Optional[str]) -> int  # returns run_id
mark_stage_complete(stage_name: str, snapshot_id: int, run_id: int,
                    input_fingerprint: str, output_fingerprint: str)
                    # CRITICAL: validates snapshot_id == run.snapshot_id
create_group_iteration(run_id: int, snapshot_id: int, ...) -> int
                    # CRITICAL: validates snapshot_id == run.snapshot_id

# Queries
get_latest_snapshot(root_path: Optional[Path]) -> Optional[int]
get_run_by_id(run_id: int) -> Optional[Dict[str, Any]]
get_nodes_by_snapshot(snapshot_id: int, kind: Optional[str]) -> List[Node]

# Referential integrity (cross-database validation)
_validate_snapshot_exists(snapshot_id: int) -> bool
_validate_node_exists(node_id: int, snapshot_id: int) -> bool
_check_snapshot_has_runs(snapshot_id: int) -> bool
_validate_snapshot_id_matches_run(snapshot_id: int, run_id: int) -> bool
```

**Implementation notes**:
- Use SQLAlchemy ORM with declarative base (consistent with existing codebase)
- Store databases in `organizer/data/` (git-ignored)
- Schema created automatically via `Base.metadata.create_all(engine)`
- Enable WAL mode: `PRAGMA journal_mode=WAL` (using SQLAlchemy 2.x syntax)
- Enable foreign keys: `PRAGMA foreign_keys=ON` (using SQLAlchemy 2.x syntax)
- **Schema version checking**: On init, verify DB schema version matches code version

**Example initialization**:
```python
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from storage.index_models import IndexBase
from storage.work_models import WorkBase

# CRITICAL: Enable foreign keys per connection (not just once)
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

# Create engines
index_engine = create_engine(f"sqlite:///{INDEX_DB}")
work_engine = create_engine(f"sqlite:///{WORK_DB}")

# Create all tables
IndexBase.metadata.create_all(index_engine)
WorkBase.metadata.create_all(work_engine)
```

### 1.4 Schema Version Checking

**Problem**: index.db is append-only and long-lived. If schema changes, existing databases become incompatible.

**Solution**: Version checking on initialization (NO automatic migration, require re-generation).

**Implementation in StorageManager**:

```python
def _init_index_schema(self):
    """Initialize index.db schema and verify version."""
    engine = create_engine(f"sqlite:///{self.index_path}")

    # Note: PRAGMAs are set via event listener (see Phase 1.3)
    # to ensure they apply to ALL connections, not just the first one

    # Create tables if needed
    IndexBase.metadata.create_all(engine)

    # Check/set schema version
    self._verify_index_schema_version(engine)

def _verify_index_schema_version(self, engine):
    """Verify index.db schema version matches code version."""
    from sqlalchemy.orm import sessionmaker
    from storage.index_models import Meta, INDEX_SCHEMA_VERSION

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Get stored version
        meta = session.query(Meta).filter_by(key='schema_version').first()

        if meta is None:
            # New database, set version
            meta = Meta(key='schema_version', value=INDEX_SCHEMA_VERSION)
            session.add(meta)
            session.commit()
        else:
            # Existing database, check version
            if meta.value != INDEX_SCHEMA_VERSION:
                raise RuntimeError(
                    f"index.db schema version mismatch: "
                    f"database is v{meta.value}, code expects v{INDEX_SCHEMA_VERSION}. "
                    f"Delete {self.index_path} and re-run gather to regenerate snapshots."
                )
    finally:
        session.close()

def _init_work_schema(self):
    """Initialize work.db schema and verify version."""
    engine = create_engine(f"sqlite:///{self.work_path}")

    # Note: PRAGMAs are set via event listener (see Phase 1.3)
    # to ensure they apply to ALL connections, not just the first one

    # Create tables if needed
    WorkBase.metadata.create_all(engine)

    # Check/set schema version
    self._verify_work_schema_version(engine)

def _verify_work_schema_version(self, engine):
    """Verify work.db schema version matches code version."""
    from sqlalchemy.orm import sessionmaker
    from storage.work_models import Meta, WORK_SCHEMA_VERSION

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Get stored version
        meta = session.query(Meta).filter_by(key='schema_version').first()

        if meta is None:
            # New database, set version
            meta = Meta(key='schema_version', value=WORK_SCHEMA_VERSION)
            session.add(meta)
            session.commit()
        else:
            # Existing database, check version
            if meta.value != WORK_SCHEMA_VERSION:
                raise RuntimeError(
                    f"work.db schema version mismatch: "
                    f"database is v{meta.value}, code expects v{WORK_SCHEMA_VERSION}. "
                    f"Delete {self.work_path} to recreate (WARNING: loses all run data)."
                )
    finally:
        session.close()
```

**Version upgrade policy**:
- **Breaking changes**: Increment version (e.g., 1.0.0 → 2.0.0), require re-generation
- **Non-breaking additions**: Can keep version if backward compatible
- **Error messages**: Tell user exactly which file to delete and consequences

**When to increment versions**:
- Adding/removing columns: **YES** (breaking change)
- Changing column types: **YES** (breaking change)
- Adding new tables: **NO** (if existing code still works)
- Changing indexes: **NO** (metadata only)

### 1.5 Reference Hash Computation

**Where**: Add to `/Users/mantrasong/dev/fs-organizer/organizer/utils/config.py`

```python
def compute_reference_hash() -> str:
    """Compute hash of all YAML config files."""
    config_files = sorted(CONFIG_DIR.glob("*.yaml"))
    hasher = hashlib.sha256()
    for f in config_files:
        hasher.update(f.read_bytes())
    return hasher.hexdigest()
```

**Usage**: Store in `snapshot.reference_hash` to track config version per snapshot.

---

## Phase 1.6: Cross-Database Referential Integrity & Snapshot Immutability

### 1.6.1 Cross-Database Foreign Key Problem

**Issue**: SQLite cannot enforce foreign key constraints across database files. `work.db` tables store `snapshot_id` and `node_id` that reference `index.db`, but these references are not enforceable at the database level.

**Risks**:
- Orphaned work.db rows if snapshots are deleted
- Invalid node_id references if nodes are modified/deleted
- Data inconsistency between databases
- No cascade delete behavior

**Solution: Application-Level Referential Integrity**

#### A. Validation on Insert/Update

All StorageManager methods that write cross-database references MUST validate first:

```python
def start_run(self, snapshot_id: int, pipeline_version: str, ...) -> int:
    """Create a new run. Validates snapshot exists before creating."""
    # CRITICAL: Validate snapshot exists in index.db
    if not self._validate_snapshot_exists(snapshot_id):
        raise ValueError(f"Snapshot {snapshot_id} does not exist")

    # Now safe to create run in work.db
    session = self.get_work_session()
    run = Run(snapshot_id=snapshot_id, ...)
    session.add(run)
    session.commit()
    return run.run_id

def insert_group_entry(self, iteration_id: int, node_id: int, ...) -> int:
    """Insert group entry. Validates node exists before creating."""
    # Get snapshot_id from iteration
    iteration = self._get_iteration(iteration_id)

    # CRITICAL: Validate node exists in index.db
    if not self._validate_node_exists(node_id, iteration.snapshot_id):
        raise ValueError(f"Node {node_id} does not exist in snapshot {iteration.snapshot_id}")

    # Now safe to create entry in work.db
    entry = GroupEntry(iteration_id=iteration_id, node_id=node_id, ...)
    self.get_work_session().add(entry)
    self.get_work_session().commit()
    return entry.entry_id
```

#### B. Validation Helper Methods

```python
def _validate_snapshot_exists(self, snapshot_id: int) -> bool:
    """Check if snapshot exists in index.db."""
    session = self.get_index_session()
    return session.query(Snapshot).filter_by(snapshot_id=snapshot_id).first() is not None

def _validate_node_exists(self, node_id: int, snapshot_id: int) -> bool:
    """Check if node exists in index.db for given snapshot."""
    session = self.get_index_session()
    return session.query(Node).filter_by(
        node_id=node_id,
        snapshot_id=snapshot_id
    ).first() is not None

def _check_snapshot_has_runs(self, snapshot_id: int) -> bool:
    """Check if any runs reference this snapshot in work.db."""
    session = self.get_work_session()
    return session.query(Run).filter_by(snapshot_id=snapshot_id).first() is not None

def _validate_snapshot_id_matches_run(self, snapshot_id: int, run_id: int) -> bool:
    """Validate that provided snapshot_id matches the run's snapshot_id.

    CRITICAL: Work tables (StageState, GroupIteration) store snapshot_id redundantly.
    This method prevents inconsistencies where snapshot_id diverges from run.snapshot_id.
    """
    session = self.get_work_session()
    run = session.query(Run).filter_by(run_id=run_id).first()
    if not run:
        return False
    return run.snapshot_id == snapshot_id
```

#### C. Deletion Policy

**RULE**: Snapshots referenced by runs CANNOT be deleted.

```python
def delete_snapshot(self, snapshot_id: int):
    """Delete snapshot. Fails if runs exist that reference it."""
    # CRITICAL: Check for referencing runs
    if self._check_snapshot_has_runs(snapshot_id):
        raise ValueError(
            f"Cannot delete snapshot {snapshot_id}: "
            f"runs still reference it. Delete runs first."
        )

    # Safe to delete
    session = self.get_index_session()
    snapshot = session.query(Snapshot).filter_by(snapshot_id=snapshot_id).first()
    if snapshot:
        session.delete(snapshot)  # Cascades to nodes, node_features via SQLAlchemy
        session.commit()

def delete_run(self, run_id: int):
    """Delete run and all associated work data."""
    session = self.get_work_session()
    run = session.query(Run).filter_by(run_id=run_id).first()
    if run:
        # SQLAlchemy cascade deletes stages, group_iterations, etc.
        session.delete(run)
        session.commit()
```

#### D. Validation Tests

**Critical testing note: In-memory database setup**

When using `:memory:` databases for testing, SQLAlchemy creates a **new in-memory database for each connection**. This causes:
- Schema/version checks to fail (each connection sees empty database)
- Session-based reads to return no data (different DB than where data was written)

**Solution**: Use one of these approaches:
1. **StaticPool + check_same_thread=False** (recommended for tests):
   ```python
   from sqlalchemy.pool import StaticPool
   engine = create_engine(
       "sqlite:///:memory:",
       connect_args={"check_same_thread": False},
       poolclass=StaticPool
   )
   ```
2. **Use temporary files instead**: `sqlite:////tmp/test_index.db`

All tests in Phases 5.1, Day 5, Day 10, Day 11 must use this pattern.

**Required unit tests** (`storage/manager_test.py`):

```python
def test_start_run_invalid_snapshot():
    """Cannot create run for non-existent snapshot."""
    mgr = StorageManager(":memory:", ":memory:")
    with pytest.raises(ValueError, match="Snapshot 999 does not exist"):
        mgr.start_run(snapshot_id=999, pipeline_version="test")

def test_insert_group_entry_invalid_node():
    """Cannot create group entry for non-existent node."""
    mgr = StorageManager(":memory:", ":memory:")
    snapshot_id = mgr.ingest_filesystem(Path("/tmp"))
    run_id = mgr.start_run(snapshot_id, "test")
    iteration_id = mgr.create_group_iteration(run_id, snapshot_id)

    with pytest.raises(ValueError, match="Node 999 does not exist"):
        mgr.insert_group_entry(iteration_id, node_id=999, ...)

def test_delete_snapshot_with_runs_fails():
    """Cannot delete snapshot if runs reference it."""
    mgr = StorageManager(":memory:", ":memory:")
    snapshot_id = mgr.ingest_filesystem(Path("/tmp"))
    run_id = mgr.start_run(snapshot_id, "test")

    with pytest.raises(ValueError, match="runs still reference it"):
        mgr.delete_snapshot(snapshot_id)

def test_delete_run_allows_snapshot_deletion():
    """After deleting all runs, snapshot can be deleted."""
    mgr = StorageManager(":memory:", ":memory:")
    snapshot_id = mgr.ingest_filesystem(Path("/tmp"))
    run_id = mgr.start_run(snapshot_id, "test")

    mgr.delete_run(run_id)  # Now no runs reference snapshot
    mgr.delete_snapshot(snapshot_id)  # Should succeed

def test_snapshot_id_consistency_in_stage_state():
    """StageState snapshot_id must match run.snapshot_id."""
    mgr = StorageManager(":memory:", ":memory:")
    snapshot_id = mgr.ingest_filesystem(Path("/tmp"))
    run_id = mgr.start_run(snapshot_id, "test")

    # Attempt to mark stage complete with wrong snapshot_id
    wrong_snapshot_id = snapshot_id + 1
    with pytest.raises(ValueError, match="snapshot_id .* does not match run"):
        mgr.mark_stage_complete("gather", wrong_snapshot_id, run_id, "in_fp", "out_fp")

def test_snapshot_id_consistency_in_group_iteration():
    """GroupIteration snapshot_id must match run.snapshot_id."""
    mgr = StorageManager(":memory:", ":memory:")
    snapshot_id = mgr.ingest_filesystem(Path("/tmp"))
    run_id = mgr.start_run(snapshot_id, "test")

    # Attempt to create iteration with wrong snapshot_id
    wrong_snapshot_id = snapshot_id + 1
    with pytest.raises(ValueError, match="snapshot_id .* does not match run"):
        mgr.create_group_iteration(run_id, wrong_snapshot_id, description="test")
```

### 1.6.2 Snapshot Immutability Contract

**Issue**: The plan states snapshots are "immutable" but the ingestion flow includes post-processing (`compute_node_features`), which could create ambiguity about when snapshots are finalized.

**Principle**: Snapshots are **write-once, read-many**. Once `ingest_filesystem()` returns, the snapshot MUST NOT be modified.

#### A. Atomic Snapshot Creation

All snapshot data (nodes + features) MUST be written in a single transaction during `ingest_filesystem()`:

```python
def ingest_filesystem(
    self,
    root_path: Path,
    preprocess_version: str = "1.0.0",
    notes: Optional[str] = None
) -> int:
    """Create immutable snapshot of filesystem. ALL data written atomically."""
    session = self.get_index_session()

    try:
        # 1. Create snapshot record
        snapshot = Snapshot(
            created_at=datetime.now().isoformat(),
            root_path=str(root_path),
            root_abs_path=str(root_path.resolve()),
            preprocess_version=preprocess_version,
            reference_hash=compute_reference_hash(),
            notes=notes
        )
        session.add(snapshot)
        session.flush()  # Get snapshot_id

        # 2. Walk filesystem and insert nodes
        self._walk_and_insert_nodes(session, snapshot.snapshot_id, root_path)

        # 3. CRITICAL: Compute node features BEFORE commit
        #    This ensures snapshot is complete when ingest_filesystem() returns
        self._compute_node_features_internal(session, snapshot.snapshot_id)

        # 4. Commit transaction - snapshot is now immutable
        session.commit()

        return snapshot.snapshot_id

    except Exception as e:
        session.rollback()
        raise RuntimeError(f"Failed to create snapshot: {e}")
```

**Key points**:
- All node insertions happen in one transaction
- `compute_node_features` is called BEFORE commit (not after)
- If any step fails, entire snapshot creation is rolled back
- Once `ingest_filesystem()` returns, snapshot is complete and frozen

#### B. Prevent Post-Creation Modifications

Add safeguards to prevent accidental snapshot modifications:

```python
def get_index_session(self, read_only: bool = False):
    """Get SQLAlchemy session for index.db.

    Args:
        read_only: If True, returns session that raises error on flush/commit.
                   Use for snapshot queries to prevent accidental mutations.

    IMPORTANT: Snapshots are immutable. When querying snapshots, prefer
    read_only=True to prevent accidental modifications via session.
    """
    Session = sessionmaker(bind=self.index_engine)
    session = Session()

    if read_only:
        # Prevent writes by raising on flush
        @event.listens_for(session, "before_flush")
        def prevent_flush(session, flush_context, instances):
            raise RuntimeError(
                "Cannot modify index.db with read-only session. "
                "Snapshots are immutable after creation."
            )

    return session

# Example: prevent modification methods
def update_node(self, node_id: int, **kwargs):
    """NOT ALLOWED: Nodes cannot be modified after snapshot creation."""
    raise NotImplementedError(
        "Nodes are immutable. Create a new snapshot to capture changes."
    )

def compute_node_features(self, snapshot_id: int):
    """NOT ALLOWED: Features must be computed during snapshot creation."""
    raise NotImplementedError(
        "Node features are computed during ingest_filesystem(). "
        "This method should not be called externally."
    )
```

#### C. Optional: Snapshot Versioning

If snapshots need to be "updated" (e.g., re-ingesting with new preprocessing logic):

```python
def create_snapshot_from_snapshot(
    self,
    source_snapshot_id: int,
    new_preprocess_version: str,
    notes: Optional[str] = None
) -> int:
    """Create new snapshot by reprocessing an existing snapshot.

    Use case: New preprocessing algorithm, want to recompute features
    without re-scanning filesystem.
    """
    source_snapshot = self.get_snapshot_by_id(source_snapshot_id)
    if not source_snapshot:
        raise ValueError(f"Source snapshot {source_snapshot_id} not found")

    # Re-ingest from original path with new version
    return self.ingest_filesystem(
        Path(source_snapshot.root_abs_path),
        preprocess_version=new_preprocess_version,
        notes=f"Re-ingested from snapshot {source_snapshot_id}. {notes or ''}"
    )
```

This creates a NEW snapshot rather than modifying the existing one.

#### D. Documentation in Code

Add docstrings that enforce the contract:

```python
class Snapshot(IndexBase):
    """Immutable snapshot of filesystem state.

    IMPORTANT: Snapshots are write-once, read-many. Once created via
    ingest_filesystem(), they MUST NOT be modified. All nodes and node_features
    are computed atomically during creation.

    To capture a new state of the filesystem, create a new snapshot instead
    of modifying an existing one.
    """
    __tablename__ = 'snapshot'
    # ... fields
```

#### E. Immutability Tests

**Required unit tests**:

```python
def test_snapshot_immutability_contract():
    """Snapshot data cannot be modified after creation."""
    mgr = StorageManager(":memory:", ":memory:")
    snapshot_id = mgr.ingest_filesystem(Path("/tmp"))

    # All features should be computed
    session = mgr.get_index_session()
    nodes = session.query(Node).filter_by(snapshot_id=snapshot_id).all()
    for node in nodes:
        assert node.features is not None  # Features must exist
        assert node.features.normalized_name is not None

def test_ingest_filesystem_is_atomic():
    """If ingestion fails, no partial snapshot is created."""
    mgr = StorageManager(":memory:", ":memory:")

    # Mock a failure during feature computation
    with patch.object(mgr, '_compute_node_features_internal', side_effect=RuntimeError("Test error")):
        with pytest.raises(RuntimeError, match="Failed to create snapshot"):
            mgr.ingest_filesystem(Path("/tmp"))

    # Verify no snapshot was created
    session = mgr.get_index_session()
    assert session.query(Snapshot).count() == 0
    assert session.query(Node).count() == 0

def test_node_modification_not_allowed():
    """Attempting to modify nodes raises NotImplementedError."""
    mgr = StorageManager(":memory:", ":memory:")
    snapshot_id = mgr.ingest_filesystem(Path("/tmp"))
    nodes = mgr.get_nodes_by_snapshot(snapshot_id)

    with pytest.raises(NotImplementedError, match="immutable"):
        mgr.update_node(nodes[0].node_id, name="new_name")
```

### 1.6.3 Summary of Guarantees

After implementing these changes, the storage layer provides:

1. **Referential Integrity**: All cross-database references validated before insert
2. **Safe Deletion**: Cannot delete snapshots referenced by runs
3. **Cascade Cleanup**: Deleting runs removes all associated work data
4. **Snapshot Immutability**: Snapshots are complete and frozen after `ingest_filesystem()` returns
5. **Atomic Creation**: Snapshot creation is all-or-nothing (no partial snapshots)
6. **Explicit Contract**: Code and tests document and enforce immutability

---

## Phase 2: Filesystem Index Implementation

### 2.1 Refactor gather.py to Create Snapshots

**File**: `/Users/mantrasong/dev/fs-organizer/organizer/pipeline/gather.py`

**Current function**: `gather_folder_structure_and_store(base_path, db_path)`
- Uses `os.walk()` to scan filesystem
- Inserts `dbFolder` and `dbFile` records
- Handles nested ZIP files via `process_zip()`
- Computes `cleaned_name` via `clean_filename()`

**New implementation approach**:

**Option A**: Dual-path during migration
```python
def gather_folder_structure_and_store(
    base_path: Path,
    db_path: Path,
    use_new_storage: bool = False
):
    if use_new_storage:
        return _gather_to_index_db(base_path)
    else:
        # Keep existing implementation
        ...

def _gather_to_index_db(base_path: Path) -> Tuple[int, int]:
    """New implementation using StorageManager.

    CRITICAL: Snapshot creation is ATOMIC. All nodes and features are
    computed inside ingest_filesystem() before it returns.
    """
    storage = StorageManager()

    # Create immutable snapshot (includes nodes + features)
    # This is atomic - either succeeds completely or rolls back
    snapshot_id = storage.ingest_filesystem(
        root_path=base_path,
        preprocess_version=get_git_hash(),
        notes=None
    )

    # Snapshot is now complete and immutable

    # Create initial run (validates snapshot exists via referential integrity check)
    run_id = storage.start_run(
        snapshot_id=snapshot_id,
        pipeline_version=get_git_hash(),
        config_hash=compute_reference_hash()
    )

    return snapshot_id, run_id
```

**Key changes**:
- `ingest_filesystem()` handles ALL snapshot creation logic atomically
- Extract `process_zip()` logic to work with `node` table instead of `Folder`/`File`
- `clean_filename()` and feature computation happen INSIDE `ingest_filesystem()` transaction
- Store `file_source` ('filesystem'|'zip_file'|'zip_content') in `node.file_source`
- Once `ingest_filesystem()` returns, snapshot is complete and immutable
- `start_run()` validates snapshot exists (referential integrity check)

### 2.2 Use SQLAlchemy Models

Models defined in `storage/index_models.py` (see section 1.2 above).

**Key relationships**:
- `Snapshot.nodes` → one-to-many with `Node`
- `Node.parent` → self-referential for directory tree
- `Node.features` → one-to-one with `NodeFeatures`

---

## Phase 3: Intermediary Work Database

### 3.1 Use Work Database Models

Models defined in `storage/work_models.py` (see section 1.2 above).

**Key relationships**:
- `Run.stages` → one-to-many with `StageState`
- `Run.group_iterations` → one-to-many with `GroupIteration`
- `GroupIteration.entries` → one-to-many with `GroupEntry`
- `GroupIteration.categories` → one-to-many with `GroupCategory`

### 3.2 Refactor group.py to Use Work Database

**File**: `/Users/mantrasong/dev/fs-organizer/organizer/grouping/group.py`

**Current flow**:
```python
group_folders(db_path: Path):
    setup_folder_categories(db_path)  # resets tables
    process_folders_to_groups(session, group_id)
    pre_process_groups(session)
    refine_groups(session, ...)
```

**New flow**:
```python
group_folders_new(snapshot_id: int, run_id: int):
    storage = StorageManager()

    # Get nodes from index.db
    nodes = storage.get_nodes_by_snapshot(snapshot_id, kind='dir')

    # Create grouping iteration
    iteration_id = storage.create_group_iteration(
        run_id=run_id,
        description="NLP clustering",
        parameters={'distance_threshold': 0.55, ...}
    )

    # Process each node (same logic as current process_folders_to_groups)
    for node in nodes:
        entry = GroupEntry(
            iteration_id=iteration_id,
            node_id=node.node_id,
            pre_processed_name=node.name,
            processed_name=node.features.normalized_name,
            ...
        )
        storage.insert_group_entry(entry)

    # Pre-process groups (same logic)
    _pre_process_groups(storage, iteration_id)

    # Refine groups (same logic)
    _refine_groups(storage, iteration_id)

    # Mark stage complete
    storage.mark_stage_complete("group", snapshot_id, run_id, ...)
```

**Key adapter pattern**:
- Create `storage/grouping_adapter.py` with functions that:
  - Query nodes from index.db
  - Fetch node_features for preprocessing
  - Write results to work.db stage tables
- Preserve existing NLP clustering logic (`nlp_grouping.py`, `group_cleanup.py`)

### 3.3 Refactor categorize.py to Use Work Database

**File**: `/Users/mantrasong/dev/fs-organizer/organizer/pipeline/categorize.py`

**Current**: Reads `File` table, writes `new_path`, creates `FolderStructure`

**New**:
```python
calculate_folder_structure_new(snapshot_id: int, run_id: int):
    storage = StorageManager()
    nodes = storage.get_nodes_by_snapshot(snapshot_id, kind='file')
    iteration_id = storage.get_latest_iteration(run_id)

    for node in nodes:
        # Get categories by walking up node hierarchy
        categories = storage.get_categories_for_node(node.node_id, iteration_id)
        new_path = "/".join([cat.processed_name for cat in categories])

        storage.insert_file_mapping(
            run_id=run_id,
            node_id=node.node_id,
            original_path=node.rel_path,
            new_path=new_path,
            groups=[cat.name for cat in categories]
        )

    # Build final structure
    folder_structure = storage.build_structure_from_mappings(run_id)
    storage.insert_folder_structure(run_id, 'organized', folder_structure)
```

---

## Phase 4: CLI Integration

### 4.1 Add New Storage Flag to Commands

**File**: `/Users/mantrasong/dev/fs-organizer/organizer/organizer.py`

**Updated commands**:

```python
@app.command()
def gather(
    base_path: Path,
    output_dir: Path,
    new_storage: bool = typer.Option(
        False,
        "--new-storage",
        help="Use new snapshot-based index.db architecture"
    )
):
    """Scan filesystem and create database."""
    if new_storage:
        storage = StorageManager()
        snapshot_id = storage.ingest_filesystem(base_path)
        run_id = storage.start_run(snapshot_id, get_git_hash())
        typer.echo(f"✓ Created snapshot {snapshot_id}, run {run_id}")
        typer.echo(f"  Root: {base_path}")
        typer.echo(f"  Index: {storage.index_path}")
    else:
        # Legacy implementation (keep for backward compatibility)
        base_output = output_dir
        base_output.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = base_output / timestamp_str
        run_dir.mkdir()
        db_path = run_dir / "run_data.db"
        setup_gather(db_path)
        gather_folder_structure_and_store(base_path, db_path)
        # ... (rest of legacy logic)

@app.command()
def group(
    db_path: Optional[str] = typer.Argument(None),
    run_id: Optional[int] = typer.Option(None, "--run-id", help="Run ID for new storage"),
    new_storage: bool = typer.Option(False, "--new-storage")
):
    """Process grouping on gathered data."""
    if new_storage:
        if not run_id:
            raise typer.BadParameter("--run-id required with --new-storage")
        storage = StorageManager()
        run_info = storage.get_run_by_id(run_id)
        if not run_info:
            raise typer.BadParameter(f"Run {run_id} not found")
        group_folders_new(run_info['snapshot_id'], run_id)
        typer.echo(f"✓ Grouping complete for run {run_id}")
    else:
        # Legacy
        group_folders(Path(db_path))
        calculate_folder_structure(db_path)
        typer.echo("Grouping complete.")

@app.command()
def folders(
    db_path: Optional[str] = typer.Argument(None),
    run_id: Optional[int] = typer.Option(None, "--run-id"),
    new_storage: bool = typer.Option(False, "--new-storage"),
    structure_type: StructureType = typer.Option(StructureType.organized)
):
    """Generate final folder structure."""
    if new_storage:
        if not run_id:
            raise typer.BadParameter("--run-id required")
        storage = StorageManager()
        run_info = storage.get_run_by_id(run_id)
        calculate_folder_structure_new(run_info['snapshot_id'], run_id)
    else:
        # Legacy
        recalculate_cleaned_paths_for_structure(db_path, structure_type)
        folder_tree = get_folder_heirarchy(db_path, structure_type)
        # ... (rest of legacy logic)
```

### 4.2 Add Snapshot Management Commands

```python
@app.command()
def list_snapshots():
    """List all filesystem snapshots."""
    storage = StorageManager()
    snapshots = storage.get_all_snapshots()
    for snap in snapshots:
        typer.echo(f"[{snap.snapshot_id}] {snap.created_at}: {snap.root_path}")

@app.command()
def list_runs(snapshot_id: Optional[int] = None):
    """List processing runs."""
    storage = StorageManager()
    runs = storage.get_runs(snapshot_id)
    for run in runs:
        typer.echo(f"[{run.run_id}] Snapshot {run.snapshot_id}: {run.status}")

@app.command()
def reset_run(run_id: int):
    """Delete all work for a run (keeps snapshot)."""
    storage = StorageManager()
    storage.delete_run(run_id)
    typer.echo(f"✓ Deleted run {run_id}")
```

---

## Phase 5: Testing

### 5.1 Testing Strategy

#### Unit Tests

**File**: `/Users/mantrasong/dev/fs-organizer/organizer/storage/manager_test.py`

```python
def test_create_snapshot(tmp_path):
    """Test snapshot creation."""
    mgr = StorageManager(":memory:", ":memory:")
    snapshot_id = mgr.ingest_filesystem(tmp_path)
    assert snapshot_id == 1

def test_start_run(tmp_path):
    """Test run creation."""
    mgr = StorageManager(":memory:", ":memory:")
    snapshot_id = mgr.ingest_filesystem(tmp_path)
    run_id = mgr.start_run(snapshot_id, "test-v1")
    assert run_id == 1

    run = mgr.get_run_by_id(run_id)
    assert run['snapshot_id'] == snapshot_id
    assert run['status'] == 'running'

def test_stage_completion():
    """Test stage state tracking."""
    # ... test mark_stage_complete()
```

#### Integration Tests

**File**: `/Users/mantrasong/dev/fs-organizer/organizer/storage/integration_test.py`

```python
def test_full_pipeline_new_storage(tmp_path):
    """Test complete pipeline with new storage."""
    # Setup test data
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    (test_dir / "folder1").mkdir()
    (test_dir / "folder1" / "file.txt").touch()

    # Run gather
    storage = StorageManager(":memory:", ":memory:")
    snapshot_id = storage.ingest_filesystem(test_dir)
    run_id = storage.start_run(snapshot_id, "test")

    # Run group
    group_folders_new(snapshot_id, run_id)

    # Run folders
    calculate_folder_structure_new(snapshot_id, run_id)

    # Verify output
    structure = storage.get_folder_structure(run_id, 'organized')
    assert structure is not None
```

### 5.2 Performance Benchmarking

**File**: `/Users/mantrasong/dev/fs-organizer/organizer/storage/benchmark.py`

```python
def benchmark_gather(test_dir: Path, iterations: int = 3):
    """Compare legacy vs. new storage gather performance."""
    times_legacy = []
    times_new = []

    for _ in range(iterations):
        # Time legacy
        start = time.time()
        gather_folder_structure_and_store(test_dir, ":memory:")
        times_legacy.append(time.time() - start)

        # Time new
        storage = StorageManager(":memory:", ":memory:")
        start = time.time()
        storage.ingest_filesystem(test_dir)
        times_new.append(time.time() - start)

    print(f"Legacy: {np.mean(times_legacy):.2f}s ± {np.std(times_legacy):.2f}s")
    print(f"New:    {np.mean(times_new):.2f}s ± {np.std(times_new):.2f}s")
```

**Acceptance criteria**: New storage should not be >20% slower than legacy.

---

## Implementation Sequence

### Week 1: Foundation
- [ ] Create `organizer/storage/` module structure
- [ ] Create `storage/index_models.py` with SQLAlchemy models (Snapshot, Node, NodeFeatures)
- [ ] Create `storage/work_models.py` with SQLAlchemy models (Run, StageState, GroupIteration, etc.)
- [ ] Implement `StorageManager.__init__()`, `_ensure_databases()` using `Base.metadata.create_all()`
- [ ] Implement `compute_reference_hash()` in `utils/config.py`
- [ ] Write unit tests for database creation

### Week 2: Index Database
- [ ] Implement `StorageManager.ingest_filesystem()`
  - Reuse `gather.py` filesystem walking logic
  - Convert `dbFolder`/`dbFile` → `Node` ORM inserts
- [ ] Implement node_features computation (cleaned names, tokens)
- [ ] Test snapshot creation with real directories

### Week 3: Work Database
- [ ] Implement `start_run()`, `mark_stage_complete()`
- [ ] Create `storage/grouping_adapter.py`
  - Query nodes from index.db
  - Write group results to work.db
- [ ] Refactor `group.py` to support new storage
- [ ] Test grouping with new storage

### Week 4: Pipeline Integration
- [ ] Refactor `categorize.py` for new storage
- [ ] Update `organizer.py` CLI:
  - Add `--new-storage` flag to `gather`, `group`, `folders`
  - Add `list-snapshots`, `list-runs`, `reset-run` commands
- [ ] Integration testing with full pipeline

### Week 5: Refinement
- [ ] Performance benchmarking and optimization
- [ ] Add fingerprinting logic for stage inputs/outputs
- [ ] Documentation updates (README.md, CLAUDE.md)
- [ ] Update frontend to read from new storage (or export compatibility layer)

---

## Critical Files to Modify

| File | Purpose | Changes |
|------|---------|---------|
| `/Users/mantrasong/dev/fs-organizer/organizer/storage/manager.py` | Core storage API | **NEW** - Primary database interface |
| `/Users/mantrasong/dev/fs-organizer/organizer/storage/index_models.py` | Index DB models | **NEW** - SQLAlchemy models for snapshots/nodes |
| `/Users/mantrasong/dev/fs-organizer/organizer/storage/work_models.py` | Work DB models | **NEW** - SQLAlchemy models for runs/stages |
| `/Users/mantrasong/dev/fs-organizer/organizer/pipeline/gather.py` | Filesystem scanning | Refactor to support new storage |
| `/Users/mantrasong/dev/fs-organizer/organizer/grouping/group.py` | NLP clustering | Add new storage adapter |
| `/Users/mantrasong/dev/fs-organizer/organizer/pipeline/categorize.py` | Folder structure generation | Refactor for work.db queries |
| `/Users/mantrasong/dev/fs-organizer/organizer/organizer.py` | CLI entry point | Add flags and new commands |
| `/Users/mantrasong/dev/fs-organizer/organizer/utils/config.py` | Config management | Add `compute_reference_hash()` |

---

## Risk Mitigation

### Critical Risk (NOW MITIGATED): PRAGMA foreign_keys Per-Connection
- **Original Risk**: Setting `PRAGMA foreign_keys=ON` only once during engine init means new connections silently disable FK enforcement, allowing orphaned NodeFeatures and broken parent links.
- **Mitigation (Phase 1.3)**:
  - SQLAlchemy event listener on `Engine.connect` sets PRAGMAs for every new connection
  - `@event.listens_for(Engine, "connect")` ensures consistent FK enforcement
  - No deprecated `engine.execute()` usage (SQLAlchemy 2.x compatible)
  - Day 4 checklist explicitly includes event listener setup

### Critical Risk (NOW MITIGATED): In-Memory Test Database Isolation
- **Original Risk**: Using `StorageManager(":memory:", ":memory:")` creates separate DBs per connection, causing schema/version checks and session reads to intermittently fail.
- **Mitigation (Phase 1.6.1D, Days 5/10/11)**:
  - All tests use `StaticPool + check_same_thread=False` for :memory: databases
  - Alternative: Use temp files instead of :memory:
  - Explicit note added to test plan in both documents
  - Test helper factory provides correct engine configuration

### High Risk (NOW MITIGATED): snapshot_id Consistency Across Work Tables
- **Original Risk**: `Run.snapshot_id`, `StageState.snapshot_id`, and `GroupIteration.snapshot_id` can diverge with no validation, causing inconsistent pipeline state.
- **Mitigation (Phase 1.2/3.1, Days 12-13)**:
  - Added `_validate_snapshot_id_matches_run(snapshot_id, run_id)` validation method
  - `mark_stage_complete()` validates snapshot_id matches run.snapshot_id before insert
  - `create_group_iteration()` validates snapshot_id matches run.snapshot_id before insert
  - Work model classes documented with validation requirements
  - Comprehensive tests for snapshot_id consistency

### High Risk (NOW MITIGATED): file_source Nullability Breaking Uniqueness
- **Original Risk**: If `file_source` is nullable, the composite unique constraint `(snapshot_id, rel_path, file_source)` won't prevent collisions (NULL != NULL in SQL).
- **Mitigation (Phase 1.2, Day 2)**:
  - Changed `file_source` to `nullable=False, default='filesystem'`
  - Ingestion code explicitly sets file_source for all nodes
  - Composite unique constraint now reliably prevents path collisions
  - Tests verify collision prevention works

### High Risk (NOW MITIGATED): ZIP Parent Modeling Ambiguity
- **Original Risk**: ZIP content nodes have `parent_node_id` pointing to a ZIP file node (`kind='file'`), breaking traversal logic that assumes all parents are directories.
- **Mitigation (Phase 1.2, Day 2)**:
  - Documented explicit model: ZIP files act as virtual directories
  - ZIP file nodes: `file_source='zip_file'`, can have children
  - Traversal rule: Treat `file_source='zip_file'` nodes as containers
  - Query pattern documented for getting ZIP contents
  - Implementation guide includes ZIP traversal handling

### Medium Risk (NOW MITIGATED): Snapshot Immutability Enforcement Gaps
- **Original Risk**: `get_index_session()` exposes writable session, allowing direct mutations despite NotImplementedError guards.
- **Mitigation (Phase 1.6.2B, Day 10)**:
  - Added `get_index_session(read_only=bool)` parameter
  - Read-only sessions use event listener to raise error on flush/commit
  - Query methods documented to prefer `read_only=True`
  - Test verifies read-only sessions prevent mutations

### High Risk (NOW MITIGATED): Cross-Database Referential Integrity
- **Original Risk**: SQLite cannot enforce FKs across database files, risking orphaned work.db rows
- **Mitigation (Phase 1.6.1)**:
  - Application-level validation on all cross-DB inserts
  - Deletion policy: cannot delete snapshots referenced by runs
  - Comprehensive unit tests for referential integrity
  - Explicit validation methods: `_validate_snapshot_exists()`, `_validate_node_exists()`

### High Risk (NOW MITIGATED): Snapshot Immutability Violations
- **Original Risk**: Unclear when snapshots are "complete", risk of post-creation modifications
- **Mitigation (Phase 1.6.2)**:
  - Atomic snapshot creation: all data written in single transaction
  - Feature computation happens BEFORE commit
  - Explicit immutability contract in code and docs
  - Tests verify atomicity and prevent modifications
  - NotImplementedError on any modification attempts

### Medium Risk: Performance Regression
- **Mitigation**:
  - Benchmark before/after
  - Optimize indexes (especially on `node.snapshot_id`, `node.parent_node_id`)
  - Use WAL mode for concurrent reads
  - Target: <20% slowdown acceptable
  - Note: Referential integrity checks add minimal overhead (single query per insert)

### Medium Risk: Frontend API Breakage
- **Mitigation**:
  - Update frontend to query new storage
  - Or create compatibility layer that exports work.db results in legacy format

### Low Risk: Disk Space Growth
- **Mitigation**:
  - Index.db is append-only (snapshots only deleted when no runs reference them)
  - Add `cleanup-snapshots` command for old snapshot removal
  - Document storage growth patterns
  - Deletion policy prevents accidental data loss

---

## Success Criteria

- ✅ Can create snapshots from filesystem (including nested ZIPs)
- ✅ Can run multiple grouping iterations on same snapshot
- ✅ Can query "what nodes changed between snapshot A and B?"
- ✅ Reproducible runs: identical inputs → identical outputs
- ✅ Pipeline execution time within 20% of baseline
- ✅ All existing tests pass with new storage backend
- ✅ SQLAlchemy models correctly generate database schemas
- ✅ **PRAGMA foreign_keys enforced**: Event listener sets PRAGMA on every connection (Phase 1.3)
- ✅ **Referential integrity enforced**: Cannot create runs with invalid snapshot_id
- ✅ **Referential integrity enforced**: Cannot create group entries with invalid node_id
- ✅ **Deletion policy enforced**: Cannot delete snapshots referenced by runs
- ✅ **snapshot_id consistency enforced**: mark_stage_complete and create_group_iteration validate consistency (Phase 1.2/3.1)
- ✅ **Snapshot immutability enforced**: Snapshots complete and frozen after ingest_filesystem()
- ✅ **Read-only sessions**: get_index_session(read_only=True) prevents accidental mutations (Phase 1.6.2B)
- ✅ **Atomic creation**: Failed ingestion leaves no partial snapshots
- ✅ **Schema versioning enforced**: Mismatched versions raise clear error messages
- ✅ **Path namespace uniqueness**: file_source is NOT NULL; can ingest filesystem and ZIP paths without collision
- ✅ **ZIP parent modeling**: ZIP files act as virtual directories; traversal handles kind='file' parents
- ✅ **Test infrastructure**: All :memory: tests use StaticPool to prevent separate DB issues
- ✅ **All referential integrity tests pass** (see Phase 1.6.1 section D)
- ✅ **All snapshot_id consistency tests pass** (Phase 1.6.1 section D)
- ✅ **All immutability tests pass** (see Phase 1.6.2 section E)
