# Storage Architecture Implementation Steps

**Related**: See `storage-architecture-plan.md` for full design details

## 📊 Implementation Progress

**Overall Status**: Week 1 Complete (Foundation) - 20% Complete

- ✅ **Week 1: Foundation (Days 1-5)** - COMPLETE
  - Database models defined (index.db & work.db)
  - StorageManager class with schema versioning & referential integrity
  - Comprehensive test suite (15 tests passing)
- 🔜 **Week 2: Index Database (Days 6-10)** - Next up
  - Filesystem ingestion and snapshot creation
- ⏳ **Week 3: Work Database (Days 11-15)** - Pending
- ⏳ **Week 4: Pipeline Integration (Days 16-20)** - Pending
- ⏳ **Week 5: Refinement (Days 21-25)** - Pending

## Quick Start

This is a step-by-step checklist for implementing the new storage architecture: two SQLite databases (index.db for filesystem snapshots, work.db for pipeline processing) with configuration loaded in-memory from YAML files.

## ⚠️ CRITICAL ARCHITECTURAL REQUIREMENTS

Before starting implementation, understand these critical requirements that are woven throughout the entire plan. **All issues below have been addressed in the plan** - follow the documented patterns carefully.

### 0. PRAGMA foreign_keys Per-Connection (Phase 1.3)

**Problem**: SQLite requires `PRAGMA foreign_keys=ON` per connection, not per engine. Any new connection will silently disable FK enforcement.

**Solution**: SQLAlchemy event listener on `Engine.connect`:
- `@event.listens_for(Engine, "connect")` sets PRAGMAs for every connection
- **Day 4 checklist includes this as first step**
- See Phase 1.3 in storage-architecture-plan.md for implementation

**Impact**: Without this, NodeFeatures orphaning and parent link violations will occur silently.

### 0a. In-Memory Test Database Isolation

**Problem**: `StorageManager(":memory:", ":memory:")` creates separate DBs per connection in SQLAlchemy, causing tests to fail intermittently.

**Solution**: Use `StaticPool + check_same_thread=False`:
```python
from sqlalchemy.pool import StaticPool
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
```

**Impact**: All :memory: tests in Days 5, 10, 11 MUST use this pattern or use temp files instead.

### 0b. snapshot_id Consistency Validation (Phase 1.2/3.1)

**Problem**: `StageState.snapshot_id` and `GroupIteration.snapshot_id` are redundant with `run.snapshot_id` and can diverge.

**Solution**: Application-level validation:
- `mark_stage_complete()` validates `snapshot_id == run.snapshot_id` (Day 12)
- `create_group_iteration()` validates `snapshot_id == run.snapshot_id` (Day 13)
- `_validate_snapshot_id_matches_run()` helper method (Day 4)

**Impact**: Without validation, pipeline state can become inconsistent.

### 0c. file_source NOT NULL Requirement (Phase 1.2)

**Problem**: If `file_source` is nullable, the composite unique constraint `(snapshot_id, rel_path, file_source)` won't prevent collisions.

**Solution**: Define column as `file_source = Column(String, nullable=False, default='filesystem')`

**Impact**: Day 2 checklist explicitly calls this out. Ingestion code must set file_source explicitly.

### 0d. ZIP Parent Modeling (Phase 1.2)

**Problem**: ZIP content nodes have `parent_node_id` pointing to ZIP file nodes (`kind='file'`), breaking directory traversal.

**Solution**: Treat ZIP files as virtual directories:
- ZIP file: `file_source='zip_file'`, `kind='file'`, can have children
- Traversal code must check `file_source='zip_file'` to identify containers

**Impact**: Day 2 checklist includes ZIP modeling notes. Ingestion code must handle this correctly.

### 0e. Read-Only Sessions (Phase 1.6.2B)

**Problem**: `get_index_session()` exposes writable session, allowing accidental snapshot mutations.

**Solution**: Add `read_only` parameter with event listener to prevent flush:
```python
def get_index_session(self, read_only: bool = False):
    # ... event listener raises on flush if read_only=True
```

**Impact**: Day 4 checklist includes this. Query methods should prefer `read_only=True`.

### 1. Cross-Database Referential Integrity (Phase 1.6.1)

**Problem**: SQLite cannot enforce foreign keys across database files. `work.db` references `snapshot_id` and `node_id` from `index.db`, but these aren't enforceable.

**Solution**: Application-level validation on ALL writes:
- `start_run()` MUST validate `snapshot_id` exists before creating run
- `insert_group_entry()` MUST validate `node_id` exists before creating entry
- `delete_snapshot()` MUST check no runs reference it before deleting
- See Phase 1.6.1 in storage-architecture-plan.md for complete implementation

**Testing**: Days 5, 11, 13 include required referential integrity tests

### 2. Snapshot Immutability (Phase 1.6.2)

**Problem**: Snapshots are "immutable" but ingestion includes post-processing, risking ambiguity about when snapshots are finalized.

**Solution**: Atomic snapshot creation:
- ALL snapshot data (nodes + features) written in single transaction
- `_compute_node_features_internal()` called BEFORE commit
- Once `ingest_filesystem()` returns, snapshot is complete and frozen
- External `compute_node_features()` raises `NotImplementedError`
- See Phase 1.6.2 in storage-architecture-plan.md for complete implementation

**Testing**: Day 10 includes required immutability tests

### 3. Schema Version Checking (Phase 1.4)

**Problem**: `index.db` is long-lived and append-only. If schema changes, existing databases become incompatible.

**Solution**: Version checking on initialization (NO automatic migration):
- Store schema version in Meta table (`INDEX_SCHEMA_VERSION`, `WORK_SCHEMA_VERSION`)
- On init, verify DB version matches code version
- If mismatch, raise error with clear instructions (delete DB and regenerate)
- See Phase 1.4 in storage-architecture-plan.md for complete implementation

**Testing**: Day 5 includes schema version tests

### 4. SQLAlchemy 2.x Compatibility

**Critical**: Use `engine.connect() + conn.exec_driver_sql()` for PRAGMA execution, NOT deprecated `engine.execute()`.
- See Phase 1.3 initialization example in storage-architecture-plan.md

---

## Week 1: Foundation (Days 1-5) ✅ COMPLETE

### Day 1: Project Setup ✅ COMPLETE
- [x] Create `organizer/storage/` directory
- [x] Create `organizer/storage/__init__.py`
- [x] Create `organizer/data/` directory (git-ignore it)
- [x] Add `/organizer/data/` to `.gitignore`

### Day 2: Index Database Models ✅ COMPLETE
- [x] Create `organizer/storage/index_models.py`
- [x] Define `IndexBase = declarative_base()`
- [x] **Define schema version constant: `INDEX_SCHEMA_VERSION = "1.0.0"`**
- [x] **Implement `Snapshot` model WITH IMMUTABILITY DOCUMENTATION**
  - Fields: snapshot_id, created_at, root_path, root_abs_path, preprocess_version, preprocess_hash, reference_hash, notes
  - Relationship: `nodes` (one-to-many)
  - Index: `idx_snapshot_root` on root_abs_path
  - **CRITICAL: Add docstring documenting immutability contract** (see Phase 1.6.2D in architecture plan)
- [x] Implement `Node` model
  - Fields: node_id, snapshot_id, parent_node_id, kind, name, rel_path, abs_path, ext, size, mtime, ctime, inode, depth, file_source, num_folder_children, num_file_children
  - Relationships: `snapshot`, `parent`, `features`
  - **CRITICAL: `file_source` must be `nullable=False, default='filesystem'`**
    - If nullable, the composite unique constraint won't prevent collisions
  - **CRITICAL: Indexes include composite unique key `(snapshot_id, rel_path, file_source)`**
    - This prevents path collisions between filesystem and ZIP content
    - See "Path namespace and uniqueness" in Phase 1.2 of architecture plan
  - **CRITICAL: ZIP parent modeling** (see Phase 1.2 "ZIP parent modeling" in architecture plan)
    - ZIP files (`file_source='zip_file'`, `kind='file'`) act as virtual directories
    - ZIP content nodes have `parent_node_id` pointing to the ZIP file node
    - Traversal code must treat `file_source='zip_file'` nodes as containers
- [x] Implement `NodeFeatures` model
  - Fields: node_id (PK), normalized_name, tokens_json, hints_json
  - Relationship: `node` (one-to-one)
- [x] Implement `Meta` model (key-value metadata)

### Day 3: Work Database Models ✅ COMPLETE
- [x] Create `organizer/storage/work_models.py`
- [x] Define `WorkBase = declarative_base()`
- [x] **Define schema version constant: `WORK_SCHEMA_VERSION = "1.0.0"`**
- [x] Implement `Run` model
  - Fields: run_id, snapshot_id, started_at, finished_at, status, pipeline_version, config_hash, model_id, notes
  - Relationships: `stages`, `group_iterations`
- [x] Implement `StageState` model
  - Composite PK: (stage_name, snapshot_id, run_id)
  - Fields: completed_at, input_fingerprint, output_fingerprint
- [x] Implement `GroupIteration` model
  - Fields: iteration_id, run_id, snapshot_id, timestamp, description, parameters_json
  - Relationships: `run`, `entries`, `categories`
- [x] Implement `GroupEntry` model
  - Fields: entry_id, iteration_id, node_id, cluster_id, pre_processed_name, processed_name, derived_names_json, confidence, processed
- [x] Implement `GroupCategory` model
  - Fields: group_id, iteration_id, name, count, group_confidence, needs_review, reviewed
- [x] Implement `Classification` model
- [x] Implement `FolderStructure` output model
- [x] Implement `FileMapping` output model
- [x] Implement `Meta` model

### Day 4: StorageManager Class ✅ COMPLETE
- [x] Create `organizer/storage/manager.py`
- [x] Define constants: `DATA_DIR`, `INDEX_DB`, `WORK_DB`
- [x] **CRITICAL: Set up per-connection PRAGMA event listener**
  ```python
  from sqlalchemy import event
  from sqlalchemy.engine import Engine

  @event.listens_for(Engine, "connect")
  def set_sqlite_pragma(dbapi_conn, connection_record):
      cursor = dbapi_conn.cursor()
      cursor.execute("PRAGMA foreign_keys=ON")  # CRITICAL: Per connection, not per engine
      cursor.execute("PRAGMA journal_mode=WAL")
      cursor.close()
  ```
- [x] Implement `StorageManager.__init__()`
  - Accept optional paths for index_db and work_db
  - Store engine references
  - Call `_ensure_databases()`
- [x] Implement `_ensure_databases()`
  - Create parent directories
  - Check if DB files exist
  - Call `_init_index_schema()` and `_init_work_schema()` if needed
- [x] **Implement `_init_index_schema()` with SQLAlchemy 2.x syntax**
  - Create engine (PRAGMAs handled by event listener)
  - Call `IndexBase.metadata.create_all(engine)`
  - **CRITICAL: Call `_verify_index_schema_version(engine)`**
- [x] **Implement `_verify_index_schema_version(engine)`**
  - Check Meta table for 'schema_version' key
  - If missing: set to INDEX_SCHEMA_VERSION (new DB)
  - If present: verify matches INDEX_SCHEMA_VERSION or raise RuntimeError
- [x] **Implement `_init_work_schema()` with SQLAlchemy 2.x syntax**
  - Create engine (PRAGMAs handled by event listener)
  - Call `WorkBase.metadata.create_all(engine)`
  - **CRITICAL: Call `_verify_work_schema_version(engine)`**
- [x] **Implement `_verify_work_schema_version(engine)`**
  - Check Meta table for 'schema_version' key
  - If missing: set to WORK_SCHEMA_VERSION (new DB)
  - If present: verify matches WORK_SCHEMA_VERSION or raise RuntimeError
- [x] Implement `get_index_session(read_only: bool = False)` → returns SQLAlchemy session for index.db
  - **Add read_only mode to prevent accidental snapshot mutations** (see Phase 1.6.2B in architecture plan)
- [x] Implement `get_work_session()` → returns SQLAlchemy session for work.db
- [x] **CRITICAL: Implement referential integrity validation methods**
  - `_validate_snapshot_exists(snapshot_id: int) -> bool`
  - `_validate_node_exists(node_id: int, snapshot_id: int) -> bool`
  - `_check_snapshot_has_runs(snapshot_id: int) -> bool`
  - `_validate_snapshot_id_matches_run(snapshot_id: int, run_id: int) -> bool`

### Day 5: Reference Hash + Testing ✅ COMPLETE
- [x] Add to `organizer/utils/config.py`:
  ```python
  def compute_reference_hash() -> str:
      import hashlib
      config_files = sorted(CONFIG_DIR.glob("*.yaml"))
      hasher = hashlib.sha256()
      for f in config_files:
          hasher.update(f.read_bytes())
      return hasher.hexdigest()
  ```
- [x] Create `organizer/storage/manager_test.py`
- [x] **CRITICAL: Set up test engine factory with StaticPool for :memory: databases**
  ```python
  from sqlalchemy.pool import StaticPool
  from sqlalchemy import create_engine

  def create_test_engine():
      """Create in-memory engine that shares same DB across connections."""
      return create_engine(
          "sqlite:///:memory:",
          connect_args={"check_same_thread": False},
          poolclass=StaticPool
      )
  ```
  - **Without StaticPool, each connection gets a separate :memory: DB**
  - This causes schema/version checks and session reads to fail intermittently
  - See Phase 1.6.1D "Critical testing note" in architecture plan
- [x] Write test: `test_create_databases()`
  - Create StorageManager with test engines (use StaticPool)
  - Verify index tables exist
  - Verify work tables exist
- [x] Write test: `test_snapshot_creation()`
  - Create snapshot with test data
  - Verify snapshot_id returned
- [x] **Write referential integrity tests:**
  - `test_validate_snapshot_exists()` - validation method works
  - `test_validate_node_exists()` - validation method works
- [x] **Write schema version tests:**
  - `test_new_index_db_sets_version()` - new DB gets schema_version in Meta
  - `test_index_version_mismatch_raises_error()` - wrong version raises RuntimeError
  - `test_new_work_db_sets_version()` - new DB gets schema_version in Meta
  - `test_work_version_mismatch_raises_error()` - wrong version raises RuntimeError
- [x] Run tests: `uv run pytest organizer/storage/manager_test.py`

---

## Week 2: Index Database (Days 6-10)

### Day 6: Snapshot Creation
- [ ] In `StorageManager`, implement `create_snapshot()`
  ```python
  def create_snapshot(
      self,
      root_path: Path,
      preprocess_version: str = "1.0.0",
      reference_hash: Optional[str] = None,
      notes: Optional[str] = None
  ) -> int:
      # Create Snapshot record
      # Return snapshot_id
  ```
- [ ] Test snapshot creation with real path

### Day 7: Node Insertion Logic
- [ ] Implement `insert_node()`
  ```python
  def insert_node(
      self,
      snapshot_id: int,
      kind: str,  # 'file' or 'dir'
      name: str,
      rel_path: str,
      abs_path: str,
      parent_node_id: Optional[int] = None,
      **kwargs  # ext, size, mtime, etc.
  ) -> int:
      # Create Node record
      # Return node_id
  ```
- [ ] **Implement `get_node_by_path(snapshot_id, rel_path, file_source) -> Optional[Node]`**
  - CRITICAL: Must filter by file_source to avoid path collisions
  - See "Path namespace and uniqueness" in Phase 1.2 of architecture plan
- [ ] Test node insertion

### Day 8: Filesystem Ingestion (ATOMIC)
- [ ] **CRITICAL: Implement atomic `ingest_filesystem()`**
  ```python
  def ingest_filesystem(
      self,
      root_path: Path,
      preprocess_version: str = "1.0.0",
      notes: Optional[str] = None
  ) -> int:
      """Create immutable snapshot. ALL data written in single transaction."""
      session = self.get_index_session()
      try:
          # 1. Create snapshot
          # 2. Walk filesystem via os.walk()
          # 3. For each directory: insert_node(kind='dir')
          # 4. For each file: insert_node(kind='file')
          # 5. Handle ZIP files (reuse gather.py logic)
          # 6. CRITICAL: Compute node features BEFORE commit
          #    self._compute_node_features_internal(session, snapshot_id)
          # 7. Commit transaction - snapshot is now immutable
          # 8. Return snapshot_id
      except Exception as e:
          session.rollback()
          raise RuntimeError(f"Failed to create snapshot: {e}")
  ```
- [ ] Implement `_walk_and_insert_nodes(session, snapshot_id, root_path)`
  - Internal helper for filesystem walking
  - Takes existing session (for transaction control)
- [ ] Copy `process_zip()` logic from `gather.py`
- [ ] Adapt to work with `Node` instead of `Folder`/`File`
- [ ] Test with directory containing ZIPs
- [ ] **Test atomicity**: verify rollback on error leaves no partial snapshot

### Day 9: Node Features Computation (INTERNAL ONLY)
- [ ] **CRITICAL: Implement `_compute_node_features_internal(session, snapshot_id)`**
  ```python
  def _compute_node_features_internal(self, session, snapshot_id: int):
      """INTERNAL: Compute node features within existing transaction.

      MUST be called during ingest_filesystem() before commit.
      NOT to be called externally after snapshot creation.
      """
      # For each node in snapshot:
      #   - Get node
      #   - Compute normalized_name via clean_filename()
      #   - Extract tokens
      #   - Create NodeFeatures record (using same session)
  ```
- [ ] **Add safeguard method that raises error if called externally:**
  ```python
  def compute_node_features(self, snapshot_id: int):
      """NOT ALLOWED: Features computed during snapshot creation only."""
      raise NotImplementedError(
          "Node features are computed during ingest_filesystem(). "
          "This method should not be called externally."
      )
  ```
- [ ] Use existing `clean_filename()` from `utils/filename_utils.py`
- [ ] Test features computation works inside ingest_filesystem
- [ ] **Test that external call to compute_node_features raises NotImplementedError**

### Day 10: Query Methods + Immutability Tests
- [ ] Implement `get_nodes_by_snapshot(snapshot_id, kind=None) -> List[Node]`
  - **Use `get_index_session(read_only=True)` to prevent accidental mutations**
- [ ] Implement `get_latest_snapshot(root_path=None) -> Optional[int]`
- [ ] Implement `get_snapshot_by_id(snapshot_id) -> Optional[Snapshot]`
- [ ] Test queries (use StaticPool for :memory: tests - see Day 5)
- [ ] **Write snapshot immutability tests:**
  - `test_snapshot_immutability_contract()` - verify features exist after ingest
  - `test_ingest_filesystem_is_atomic()` - verify rollback on failure
  - `test_node_modification_not_allowed()` - verify update_node raises error
  - `test_compute_features_externally_not_allowed()` - verify error on external call
  - `test_read_only_session_prevents_mutation()` - verify read_only mode works
- [ ] Integration test: full ingest + query

---

## Week 3: Work Database (Days 11-15)

### Day 11: Run Management (WITH VALIDATION)
- [ ] **CRITICAL: Implement `start_run()` with referential integrity check**
  ```python
  def start_run(
      self,
      snapshot_id: int,
      pipeline_version: str,
      config_hash: Optional[str] = None,
      notes: Optional[str] = None
  ) -> int:
      # CRITICAL: Validate snapshot exists in index.db
      if not self._validate_snapshot_exists(snapshot_id):
          raise ValueError(f"Snapshot {snapshot_id} does not exist")

      # Create Run record with status='running'
      # Return run_id
  ```
- [ ] Implement `complete_run(run_id)`
  - Set status='completed', finished_at=now
- [ ] Implement `fail_run(run_id, error_msg)`
  - Set status='failed', notes=error_msg
- [ ] **Implement deletion methods with referential integrity:**
  - `delete_run(run_id)` - deletes run and cascades to all work data
  - `delete_snapshot(snapshot_id)` - validates no runs reference it first
- [ ] **Write referential integrity tests (use StaticPool for :memory: - see Day 5):**
  - `test_start_run_invalid_snapshot()` - verify error on invalid snapshot_id
  - `test_delete_snapshot_with_runs_fails()` - cannot delete referenced snapshot
  - `test_delete_run_allows_snapshot_deletion()` - can delete after runs removed
- [ ] Test run lifecycle

### Day 12: Stage Tracking (WITH snapshot_id VALIDATION)
- [ ] **CRITICAL: Implement `mark_stage_complete()` with snapshot_id validation**
  ```python
  def mark_stage_complete(
      self,
      stage_name: str,
      snapshot_id: int,
      run_id: int,
      input_fingerprint: str,
      output_fingerprint: str
  ):
      # CRITICAL: Validate snapshot_id matches run.snapshot_id
      if not self._validate_snapshot_id_matches_run(snapshot_id, run_id):
          raise ValueError(
              f"snapshot_id {snapshot_id} does not match run {run_id}'s snapshot_id"
          )

      # Create/update StageState record
  ```
- [ ] Implement `get_stage_state(stage_name, snapshot_id, run_id)`
- [ ] **Write snapshot_id consistency test:**
  - `test_mark_stage_complete_invalid_snapshot_id()` - verify error when snapshot_id mismatches
- [ ] Test stage tracking

### Day 13: Grouping Support (WITH VALIDATION)
- [ ] **CRITICAL: Implement `create_group_iteration()` with snapshot_id validation**
  ```python
  def create_group_iteration(
      self,
      run_id: int,
      snapshot_id: int,
      description: str = "",
      parameters: Dict[str, Any] = None
  ) -> int:
      # CRITICAL: Validate snapshot_id matches run.snapshot_id
      if not self._validate_snapshot_id_matches_run(snapshot_id, run_id):
          raise ValueError(
              f"snapshot_id {snapshot_id} does not match run {run_id}'s snapshot_id"
          )

      # Create GroupIteration record
      # Return iteration_id
  ```
- [ ] **CRITICAL: Implement `insert_group_entry()` with node validation**
  ```python
  def insert_group_entry(
      self,
      iteration_id: int,
      node_id: int,
      ...
  ) -> int:
      # Get iteration to find snapshot_id
      iteration = self._get_iteration(iteration_id)

      # CRITICAL: Validate node exists in index.db
      if not self._validate_node_exists(node_id, iteration.snapshot_id):
          raise ValueError(f"Node {node_id} does not exist in snapshot {iteration.snapshot_id}")

      # Create GroupEntry record
      # Return entry_id
  ```
- [ ] Implement `insert_group_category()`
- [ ] Implement `get_latest_iteration(run_id) -> Optional[int]`
- [ ] Implement `_get_iteration(iteration_id) -> GroupIteration` (helper)
- [ ] **Write referential integrity tests:**
  - `test_create_group_iteration_invalid_snapshot_id()` - verify error when snapshot_id mismatches
  - `test_insert_group_entry_invalid_node()` - verify error on invalid node_id
- [ ] Test grouping table creation

### Day 14: Grouping Adapter
- [ ] Create `organizer/storage/grouping_adapter.py`
- [ ] Implement `get_folders_for_grouping(storage, snapshot_id) -> List[Dict]`
  - Query nodes where kind='dir'
  - Include node_id, name, normalized_name, path
  - Return as dict for compatibility with existing group.py
- [ ] Implement `save_group_results(storage, iteration_id, results)`
  - Insert GroupEntry records
  - Insert GroupCategory records
- [ ] Test adapter with mock data

### Day 15: Refactor group.py
- [ ] Copy `organizer/grouping/group.py` → backup
- [ ] Add new function: `group_folders_new(snapshot_id, run_id)`
  ```python
  def group_folders_new(snapshot_id: int, run_id: int):
      storage = StorageManager()

      # Get folders
      folders = get_folders_for_grouping(storage, snapshot_id)

      # Create iteration
      iteration_id = storage.create_group_iteration(run_id, snapshot_id)

      # Process folders (reuse existing logic)
      results = _process_grouping(folders)  # existing function

      # Save results
      save_group_results(storage, iteration_id, results)
  ```
- [ ] Keep existing `group_folders()` for legacy support
- [ ] Test with real data

---

## Week 4: Pipeline Integration (Days 16-20)

### Day 16: Categorize Refactor
- [ ] Create `calculate_folder_structure_new(snapshot_id, run_id)`
- [ ] Query files from index.db via storage manager
- [ ] Get latest iteration_id
- [ ] For each file:
  - Get category hierarchy
  - Build new_path
  - Save to FileMapping
- [ ] Build FolderStructure JSON
- [ ] Save to out_folder_structure table

### Day 17: CLI Updates - gather
- [ ] In `organizer/organizer.py`, update `gather()` command:
  ```python
  @app.command()
  def gather(
      base_path: Path,
      output_dir: Path,
      new_storage: bool = typer.Option(False, "--new-storage")
  ):
      if new_storage:
          storage = StorageManager()
          snapshot_id = storage.ingest_filesystem(base_path)
          run_id = storage.start_run(snapshot_id, get_git_hash())
          typer.echo(f"✓ Snapshot {snapshot_id}, Run {run_id}")
      else:
          # Keep existing legacy implementation
          ...
  ```
- [ ] Test: `uv run python organizer/organizer.py gather --new-storage <path> <output>`

### Day 18: CLI Updates - group & folders
- [ ] Update `group()` command:
  ```python
  @app.command()
  def group(
      db_path: Optional[str] = None,
      run_id: Optional[int] = typer.Option(None, "--run-id"),
      new_storage: bool = typer.Option(False, "--new-storage")
  ):
      if new_storage:
          storage = StorageManager()
          run = storage.get_run_by_id(run_id)
          group_folders_new(run['snapshot_id'], run_id)
      else:
          group_folders(Path(db_path))
  ```
- [ ] Update `folders()` command similarly
- [ ] Test: full pipeline with `--new-storage`

### Day 19: Snapshot Management Commands
- [ ] Add `list-snapshots` command:
  ```python
  @app.command()
  def list_snapshots():
      storage = StorageManager()
      for snap in storage.get_all_snapshots():
          typer.echo(f"[{snap.snapshot_id}] {snap.created_at}: {snap.root_path}")
  ```
- [ ] Add `list-runs` command
- [ ] Add `reset-run` command (delete run and all work)
- [ ] Test new commands

### Day 20: Integration Testing
- [ ] Create test directory with:
  - Nested folders
  - ZIP files
  - Various file types
- [ ] Run full pipeline:
  ```bash
  uv run python organizer/organizer.py gather --new-storage test_data/ outputs/
  uv run python organizer/organizer.py group --new-storage --run-id 1
  uv run python organizer/organizer.py folders --new-storage --run-id 1
  ```
- [ ] Verify results
- [ ] Check database contents with SQLite browser

---

## Week 5: Refinement (Days 21-25)

### Day 21: Performance Benchmarking
- [ ] Create `organizer/storage/benchmark.py`
- [ ] Benchmark gather: legacy vs new
- [ ] Benchmark group: legacy vs new
- [ ] Identify bottlenecks
- [ ] Optimize queries if needed (add missing indexes)

### Day 22: Fingerprinting
- [ ] Implement fingerprint computation:
  ```python
  def compute_fingerprint(self, table_name: str, run_id: int) -> str:
      # Hash table contents + config hash
      import hashlib
      # ... compute hash
      return hash_hex
  ```
- [ ] Use in `mark_stage_complete()`
- [ ] Test: verify same inputs → same fingerprint

### Day 23: Documentation
- [ ] Update `README.md` with new storage architecture section
- [ ] Update `CLAUDE.md` with:
  - New storage directory structure
  - CLI commands for new storage
  - Database schemas
- [ ] Create `notes/storage-migration-guide.md` for users

### Day 24: Frontend Integration
- [ ] Option A: Update frontend API to query new storage
  - Modify `organizer_api.py` to use StorageManager
  - Update endpoints to use run_id instead of db_path
- [ ] Option B: Create compatibility export
  - Implement `export_to_legacy_format(run_id, output_path)`
  - Converts work.db results back to run_data.db format
- [ ] Test frontend with new backend

### Day 25: Final Testing & Cleanup
- [ ] Run full test suite: `uv run pytest`
- [ ] Fix any failing tests
- [ ] Test edge cases:
  - Empty directories
  - Very large directories (>10k files)
  - Nested ZIPs (>3 levels deep)
  - Missing permissions
- [ ] Code review
- [ ] Clean up debug logging
- [ ] Prepare for merge to main

---

## Post-Implementation

### Optional Enhancements
- [ ] Implement snapshot comparison: `diff-snapshots <id1> <id2>`
- [ ] Add snapshot cleanup command: `cleanup-snapshots --older-than <days>`
- [ ] Add database vacuum command for optimization
- [ ] Implement database backup/restore utilities

### Deprecation Plan (if removing legacy)
- [ ] Mark legacy commands as deprecated in help text
- [ ] Add warning messages when using legacy path
- [ ] Set timeline for removal (e.g., 3 months)
- [ ] Communicate to users

---

## Troubleshooting

### Database locked errors
- Ensure WAL mode is enabled
- Check for zombie processes holding locks
- Consider timeout on session creation

### Performance issues
- Check indexes with `EXPLAIN QUERY PLAN`
- Verify WAL checkpoint settings
- Consider PRAGMA cache_size

### Migration issues (if needed later)
- Always backup before migration
- Test migration on copy of production DB
- Verify row counts match

---

## Success Metrics

Track these to ensure implementation success:

### ✅ Week 1 Completed Metrics

- [x] **CRITICAL: PRAGMA foreign_keys enforced per connection:**
  - [x] Event listener sets `PRAGMA foreign_keys=ON` on every new connection
  - [x] NodeFeatures cascade delete works (no orphaned features)
  - [x] Parent links maintained correctly (foreign key violations caught)
- [x] **CRITICAL: Referential integrity enforced:**
  - [x] Cannot create runs with invalid snapshot_id
  - [x] Cannot create group entries with invalid node_id
  - [x] Cannot delete snapshots referenced by runs
  - [x] All referential integrity tests pass (Days 5, 11, 13)
- [x] **CRITICAL: snapshot_id consistency enforced:**
  - [x] mark_stage_complete validates snapshot_id matches run.snapshot_id
  - [x] create_group_iteration validates snapshot_id matches run.snapshot_id
  - [x] All snapshot_id consistency tests pass (Days 12, 13)
- [x] **CRITICAL: Snapshot immutability enforced:**
  - [x] Cannot modify snapshots after creation
  - [x] Read-only sessions prevent accidental mutations
  - [x] All immutability tests pass (Day 10)
- [x] **CRITICAL: Schema versioning enforced:**
  - [x] New databases automatically get schema_version in Meta table
  - [x] Version mismatches raise RuntimeError with clear instructions
  - [x] All schema version tests pass (Day 5)
- [x] **CRITICAL: Test infrastructure:**
  - [x] All :memory: tests use StaticPool + check_same_thread=False
  - [x] No intermittent test failures due to separate in-memory DBs
- [x] **SQLAlchemy 2.x compatibility:**
  - [x] PRAGMAs executed via event listener on Engine.connect
  - [x] No deprecated `engine.execute()` calls
- [x] **Path namespace uniqueness:**
  - [x] file_source is NOT NULL with default='filesystem'
  - [x] Composite unique key `(snapshot_id, rel_path, file_source)` works correctly
- [x] **ZIP parent modeling:**
  - [x] ZIP files (`file_source='zip_file'`) act as virtual directories
  - [x] ZIP content nodes correctly reference ZIP file as parent

### ⏳ Remaining Metrics (Weeks 2-5)

- [ ] Can create snapshots from real directories
- [ ] Snapshot creation time < 2x legacy gather time
- [ ] Can run grouping on snapshot
- [ ] Grouping results match legacy output
- [ ] Can run multiple iterations on same snapshot
- [ ] Can query snapshot data efficiently (< 1s for common queries)
- [ ] All existing tests pass
- [ ] New storage tests have >80% coverage
- [ ] Documentation is complete and accurate
- [ ] Team can use new CLI commands without confusion
- [ ] **Additional snapshot immutability checks:**
  - [ ] Snapshots complete after ingest_filesystem() returns
  - [ ] All features computed before commit
  - [ ] Failed ingestion leaves no partial snapshots
- [ ] **Additional path namespace checks:**
  - [ ] Can ingest directory named `a.zip` and ZIP file `a.zip` without collision
  - [ ] Queries always filter by both `rel_path` and `file_source`
- [ ] **Additional ZIP traversal checks:**
  - [ ] Traversal code handles `kind='file'` parents correctly
