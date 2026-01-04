# Dual-State File System Design with Undo/Redo Support

## Overview

This document outlines the design for implementing a dual-state file system representative that enables the frontend to track changes, maintain an undo/redo queue, and send patch commands to the backend for applying modifications.

## Goals

1. **State Management**: Track both current and historical states of the folder tree
2. **Undo/Redo**: Allow users to undo and redo operations on the folder structure
3. **Efficient Sync**: Send only the changes (patches) to the backend rather than the entire tree
4. **Type Safety**: Maintain TypeScript type safety throughout the implementation
5. **Performance**: Keep the UI responsive even with large folder structures

## Current Architecture Analysis

### Frontend State (useFolderTree hook)

```typescript
interface FolderTreeState {
  originalTree: FolderV2 | null;     // The backend's current state
  modifiedTree: FolderV2 | null;     // The user's working copy
  hasModifications: boolean;          // Flag indicating unsaved changes
  expandedFolders: Set<string>;       // UI state for expanded folders
  selectedFileId: number | null;      // Currently selected file
  selectedFolderPaths: string[];      // Currently selected folder(s)
}
```

### Operations

Tree operations are performed via utility functions in `folderTreeOperations.ts`:
- `moveItems()` - Move files/folders
- `deleteItems()` - Delete files/folders
- `flattenItems()` - Flatten folder hierarchies
- `invertItems()` - Invert folder structures
- `createFolder()` - Create new folders

### Backend Storage

- **FolderStructure** table stores serialized folder trees
- **FileMapping** table stores original_path → new_path mappings
- Current API only supports full tree replacement via POST `/api/save-graph`

## Design: Dual-State System with Undo/Redo

### 1. Frontend: Enhanced State Management

#### 1.1 History Stack Structure

```typescript
/**
 * Represents a single operation that can be undone/redone
 */
interface HistoryEntry {
  // The complete tree state after this operation
  tree: FolderV2;

  // Metadata about the operation
  operation: OperationMetadata;

  // Timestamp for debugging and history display
  timestamp: number;

  // Optional: The patch that was applied (for efficient backend sync)
  patch?: TreePatch;
}

interface OperationMetadata {
  type: OperationType;
  description: string;  // Human-readable description for UI
  affectedPaths: string[];  // Paths affected by this operation
}

type OperationType =
  | 'move'
  | 'delete'
  | 'create'
  | 'rename'
  | 'flatten'
  | 'invert'
  | 'confidence-change';

/**
 * The history state manages undo/redo stacks
 */
interface HistoryState {
  // Stack of previous states (for undo)
  past: HistoryEntry[];

  // Current state
  present: HistoryEntry;

  // Stack of future states (for redo after undo)
  future: HistoryEntry[];

  // Configuration
  maxHistorySize: number;  // Prevent unbounded memory growth
}
```

#### 1.2 Updated FolderTreeState

```typescript
interface FolderTreeState {
  // Original tree from backend (read-only reference)
  originalTree: FolderV2 | null;

  // History management
  history: HistoryState;

  // Computed properties
  hasModifications: boolean;  // present !== originalTree
  canUndo: boolean;           // past.length > 0
  canRedo: boolean;           // future.length > 0

  // UI state (unchanged)
  expandedFolders: Set<string>;
  selectedFileId: number | null;
  selectedFolderPaths: string[];

  // Sync state
  isSyncing: boolean;         // True while sending patches to backend
  lastSyncedEntry: HistoryEntry | null;  // Last state synced to backend
}
```

#### 1.3 History Management Actions

```typescript
interface HistoryActions {
  /**
   * Execute an operation and add it to history
   * This is the main entry point for all tree modifications
   */
  executeOperation: (
    operation: OperationMetadata,
    treeMutation: (tree: FolderV2) => FolderV2
  ) => void;

  /**
   * Undo the last operation
   */
  undo: () => void;

  /**
   * Redo a previously undone operation
   */
  redo: () => void;

  /**
   * Clear all history (useful after save or reset)
   */
  clearHistory: () => void;

  /**
   * Jump to a specific point in history
   */
  jumpToHistory: (index: number) => void;
}
```

### 2. Patch System for Efficient Backend Sync

Instead of sending the entire tree to the backend, we'll send only the changes (patches).

#### 2.1 Patch Structure

```typescript
/**
 * Represents a change to the folder tree
 */
interface TreePatch {
  version: number;  // Patch format version for future compatibility
  baseTreeId: string;  // ID/hash of the tree this patch applies to
  operations: PatchOperation[];
}

/**
 * A single atomic operation in a patch
 */
type PatchOperation =
  | MovePatchOperation
  | DeletePatchOperation
  | CreatePatchOperation
  | RenamePatchOperation
  | UpdatePatchOperation;

interface MovePatchOperation {
  type: 'move';
  itemType: 'file' | 'folder';
  itemId?: number;  // For files
  sourcePath: string;
  targetPath: string;
}

interface DeletePatchOperation {
  type: 'delete';
  itemType: 'file' | 'folder';
  itemId?: number;  // For files
  path: string;
}

interface CreatePatchOperation {
  type: 'create';
  path: string;
  folderData: {
    name: string;
    confidence: number;
  };
}

interface RenamePatchOperation {
  type: 'rename';
  path: string;
  oldName: string;
  newName: string;
}

interface UpdatePatchOperation {
  type: 'update';
  path: string;
  updates: {
    confidence?: number;
    // Add other mutable properties as needed
  };
}
```

#### 2.2 Patch Generation

The frontend will generate patches by comparing the current tree with the last synced tree:

```typescript
/**
 * Generate a patch by comparing two trees
 */
function generatePatch(
  from: FolderV2,
  to: FolderV2,
  baseTreeId: string
): TreePatch {
  const operations: PatchOperation[] = [];

  // Recursively compare trees and generate operations
  compareAndGenerateOps(from, to, '', operations);

  return {
    version: 1,
    baseTreeId,
    operations
  };
}

/**
 * Recursively compare two folder nodes and generate patch operations
 */
function compareAndGenerateOps(
  fromNode: FolderV2,
  toNode: FolderV2,
  currentPath: string,
  operations: PatchOperation[]
): void {
  // Implementation would:
  // 1. Detect moved items (by comparing IDs at different paths)
  // 2. Detect deleted items (items in 'from' but not in 'to')
  // 3. Detect created items (items in 'to' but not in 'from')
  // 4. Detect renamed items (same path, different name)
  // 5. Detect updates (same path and name, different properties)

  // This is a complex algorithm that needs careful implementation
  // to handle edge cases like moves that also rename
}
```

#### 2.3 Alternative: Operation Recording

Instead of comparing trees, we can record operations as they happen:

```typescript
/**
 * Wrapper around tree operations that records patches
 */
class PatchRecorder {
  private operations: PatchOperation[] = [];

  recordMove(itemType: 'file' | 'folder', itemId: number | undefined,
             sourcePath: string, targetPath: string): void {
    this.operations.push({
      type: 'move',
      itemType,
      itemId,
      sourcePath,
      targetPath
    });
  }

  recordDelete(itemType: 'file' | 'folder', itemId: number | undefined,
               path: string): void {
    this.operations.push({
      type: 'delete',
      itemType,
      itemId,
      path
    });
  }

  // ... other record methods

  toPatch(baseTreeId: string): TreePatch {
    return {
      version: 1,
      baseTreeId,
      operations: this.operations
    };
  }

  reset(): void {
    this.operations = [];
  }
}
```

**Recommendation**: Use operation recording (2.3) as it's simpler and more accurate than tree comparison.

### 3. Integration with Existing Operations

Modify existing operations to work with the history system:

```typescript
// Before (current):
const moveItems = (tree, sourcePath, targetPath) => {
  // ... mutation logic
  return newTree;
};

// After (with history):
const moveItems = (
  executeOperation: HistoryActions['executeOperation'],
  patchRecorder: PatchRecorder,
  sourcePath: string,
  targetPath: string
) => {
  executeOperation(
    {
      type: 'move',
      description: `Move from ${sourcePath} to ${targetPath}`,
      affectedPaths: [sourcePath, targetPath]
    },
    (tree) => {
      // Record the operation for patch generation
      patchRecorder.recordMove('folder', undefined, sourcePath, targetPath);

      // Perform the actual mutation
      return performMove(tree, sourcePath, targetPath);
    }
  );
};
```

### 4. Backend: Patch Application API

#### 4.1 New API Endpoint

```python
# organizer/api/models.py

class PatchOperation(BaseModel):
    type: str  # 'move' | 'delete' | 'create' | 'rename' | 'update'
    itemType: str | None = None  # 'file' | 'folder'
    itemId: int | None = None
    path: str | None = None
    sourcePath: str | None = None
    targetPath: str | None = None
    oldName: str | None = None
    newName: str | None = None
    folderData: dict | None = None
    updates: dict | None = None

class TreePatch(BaseModel):
    version: int
    baseTreeId: str
    operations: list[PatchOperation]

class ApplyPatchRequest(BaseModel):
    run_id: int
    structure_type: StructureType
    patch: TreePatch

class ApplyPatchResponse(BaseModel):
    success: bool
    message: str
    new_tree_id: str | None = None
    conflicts: list[str] = []  # Paths where conflicts occurred
```

```python
# organizer/organizer_api.py

@app.patch("/api/folder-structure")
async def apply_patch(request: ApplyPatchRequest) -> ApplyPatchResponse:
    """
    Apply a patch to the folder structure.

    This endpoint applies incremental changes to the folder structure
    instead of replacing the entire tree, enabling efficient updates
    and conflict detection.
    """
    try:
        storage = StorageManager(Path(db_path).parent)

        # Load the current tree
        current_tree = get_folder_structure_from_db(
            db_path,
            request.structure_type
        )

        if not current_tree:
            raise HTTPException(
                status_code=404,
                message="No folder structure found"
            )

        # Verify base tree ID matches (optimistic locking)
        current_tree_id = compute_tree_hash(current_tree)
        if current_tree_id != request.patch.baseTreeId:
            return ApplyPatchResponse(
                success=False,
                message="Base tree mismatch - structure was modified by another source",
                conflicts=[]
            )

        # Apply the patch
        try:
            updated_tree, conflicts = apply_patch_to_tree(
                current_tree,
                request.patch
            )
        except PatchApplicationError as e:
            return ApplyPatchResponse(
                success=False,
                message=str(e),
                conflicts=e.conflicts
            )

        # Save the updated tree
        new_tree_id = save_folder_structure(
            storage,
            request.run_id,
            request.structure_type,
            updated_tree
        )

        # Update file mappings based on patch operations
        update_file_mappings_from_patch(
            storage,
            request.run_id,
            request.patch
        )

        return ApplyPatchResponse(
            success=True,
            message="Patch applied successfully",
            new_tree_id=new_tree_id,
            conflicts=[]
        )

    except Exception as e:
        logger.exception("Error applying patch")
        raise HTTPException(status_code=500, detail=str(e))
```

#### 4.2 Patch Application Logic

```python
# organizer/stages/patch_operations.py

from api.models import TreePatch, PatchOperation, FolderV2, File
import hashlib
import json

class PatchApplicationError(Exception):
    def __init__(self, message: str, conflicts: list[str]):
        super().__init__(message)
        self.conflicts = conflicts

def compute_tree_hash(tree: dict) -> str:
    """
    Compute a stable hash of the tree structure for optimistic locking.
    """
    # Sort keys to ensure stable serialization
    tree_json = json.dumps(tree, sort_keys=True)
    return hashlib.sha256(tree_json.encode()).hexdigest()

def apply_patch_to_tree(
    tree: dict,
    patch: TreePatch
) -> tuple[dict, list[str]]:
    """
    Apply a patch to a folder tree.

    Returns:
        (updated_tree, conflicts) where conflicts is a list of paths
        that couldn't be modified due to concurrent changes
    """
    # Deep copy the tree to avoid mutations
    tree_copy = json.loads(json.dumps(tree))
    conflicts = []

    for operation in patch.operations:
        try:
            if operation.type == 'move':
                apply_move_operation(tree_copy, operation)
            elif operation.type == 'delete':
                apply_delete_operation(tree_copy, operation)
            elif operation.type == 'create':
                apply_create_operation(tree_copy, operation)
            elif operation.type == 'rename':
                apply_rename_operation(tree_copy, operation)
            elif operation.type == 'update':
                apply_update_operation(tree_copy, operation)
            else:
                raise ValueError(f"Unknown operation type: {operation.type}")
        except (PathNotFoundError, ItemNotFoundError) as e:
            # Record conflict but continue with other operations
            conflicts.append(str(e))

    return tree_copy, conflicts

def apply_move_operation(tree: dict, op: PatchOperation) -> None:
    """Move an item from sourcePath to targetPath."""
    # Find the item at sourcePath
    item = remove_item_at_path(tree, op.sourcePath)

    # Insert it at targetPath
    insert_item_at_path(tree, op.targetPath, item)

def apply_delete_operation(tree: dict, op: PatchOperation) -> None:
    """Delete an item at the specified path."""
    remove_item_at_path(tree, op.path)

def apply_create_operation(tree: dict, op: PatchOperation) -> None:
    """Create a new folder at the specified path."""
    new_folder = FolderV2(
        name=op.folderData['name'],
        confidence=op.folderData.get('confidence', 1.0),
        count=0,
        children=[]
    )
    insert_item_at_path(tree, op.path, new_folder.model_dump())

def apply_rename_operation(tree: dict, op: PatchOperation) -> None:
    """Rename an item at the specified path."""
    item = find_item_at_path(tree, op.path)
    if item:
        item['name'] = op.newName

def apply_update_operation(tree: dict, op: PatchOperation) -> None:
    """Update properties of an item at the specified path."""
    item = find_item_at_path(tree, op.path)
    if item:
        for key, value in op.updates.items():
            item[key] = value

# Helper functions for tree navigation

def find_item_at_path(tree: dict, path: str) -> dict | None:
    """Find an item at the given path."""
    parts = path.split('/') if path else []
    current = tree

    for part in parts:
        children = current.get('children', [])
        found = False
        for child in children:
            if child.get('name') == part:
                current = child
                found = True
                break
        if not found:
            raise PathNotFoundError(f"Path not found: {path}")

    return current

def remove_item_at_path(tree: dict, path: str) -> dict:
    """Remove and return an item at the given path."""
    parts = path.split('/')
    parent_path = '/'.join(parts[:-1])
    item_name = parts[-1]

    parent = find_item_at_path(tree, parent_path) if parent_path else tree
    children = parent.get('children', [])

    for i, child in enumerate(children):
        if child.get('name') == item_name or child.get('id') == item_name:
            return children.pop(i)

    raise ItemNotFoundError(f"Item not found: {path}")

def insert_item_at_path(tree: dict, path: str, item: dict) -> None:
    """Insert an item at the given path."""
    parts = path.split('/')
    parent_path = '/'.join(parts[:-1])

    parent = find_item_at_path(tree, parent_path) if parent_path else tree

    if 'children' not in parent:
        parent['children'] = []

    parent['children'].append(item)

class PathNotFoundError(Exception):
    pass

class ItemNotFoundError(Exception):
    pass
```

### 5. Frontend: API Integration

#### 5.1 Patch API Client

```typescript
// frontend/src/api.ts

export interface PatchOperation {
  type: 'move' | 'delete' | 'create' | 'rename' | 'update';
  itemType?: 'file' | 'folder';
  itemId?: number;
  path?: string;
  sourcePath?: string;
  targetPath?: string;
  oldName?: string;
  newName?: string;
  folderData?: {
    name: string;
    confidence: number;
  };
  updates?: Record<string, unknown>;
}

export interface TreePatch {
  version: number;
  baseTreeId: string;
  operations: PatchOperation[];
}

export interface ApplyPatchRequest {
  run_id: number;
  structure_type: 'old' | 'new';
  patch: TreePatch;
}

export interface ApplyPatchResponse {
  success: boolean;
  message: string;
  new_tree_id?: string;
  conflicts: string[];
}

export const applyPatch = async (
  request: ApplyPatchRequest
): Promise<ApplyPatchResponse> => {
  const response = await fetch(`${env.apiUrl}/api/folder-structure`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Failed to apply patch: ${response.statusText}`);
  }

  return await response.json();
};
```

#### 5.2 Integration with useFolderTree Hook

```typescript
// frontend/src/hooks/useFolderTree.ts

export const useFolderTree = (): UseFolderTreeReturn => {
  // ... existing state setup

  const [history, setHistory] = useState<HistoryState>({
    past: [],
    present: null,
    future: [],
    maxHistorySize: 50
  });

  const [patchRecorder] = useState(() => new PatchRecorder());

  // Execute an operation and add to history
  const executeOperation = useCallback(
    (
      operation: OperationMetadata,
      treeMutation: (tree: FolderV2) => FolderV2
    ) => {
      if (!history.present) return;

      // Reset the patch recorder
      patchRecorder.reset();

      // Apply the mutation
      const newTree = treeMutation(deepClone(history.present.tree));

      // Create a new history entry
      const newEntry: HistoryEntry = {
        tree: newTree,
        operation,
        timestamp: Date.now(),
        patch: patchRecorder.toPatch(computeTreeHash(history.present.tree))
      };

      // Update history (add to past, update present, clear future)
      setHistory(prev => {
        const newPast = [...prev.past, prev.present]
          .slice(-prev.maxHistorySize); // Limit history size

        return {
          ...prev,
          past: newPast,
          present: newEntry,
          future: []  // Clear redo stack on new operation
        };
      });
    },
    [history, patchRecorder]
  );

  // Undo operation
  const undo = useCallback(() => {
    setHistory(prev => {
      if (prev.past.length === 0) return prev;

      const newPresent = prev.past[prev.past.length - 1];
      const newPast = prev.past.slice(0, -1);
      const newFuture = [prev.present, ...prev.future];

      return {
        ...prev,
        past: newPast,
        present: newPresent,
        future: newFuture
      };
    });
  }, []);

  // Redo operation
  const redo = useCallback(() => {
    setHistory(prev => {
      if (prev.future.length === 0) return prev;

      const newPresent = prev.future[0];
      const newPast = [...prev.past, prev.present];
      const newFuture = prev.future.slice(1);

      return {
        ...prev,
        past: newPast,
        present: newPresent,
        future: newFuture
      };
    });
  }, []);

  // Save changes to backend
  const saveChanges = useCallback(async () => {
    if (!history.present || !state.lastSyncedEntry) return;

    setState(prev => ({ ...prev, isSyncing: true }));

    try {
      // Generate patch from last synced to current
      const patch = generatePatchFromHistory(
        state.lastSyncedEntry,
        history.present
      );

      // Apply patch to backend
      const response = await applyPatch({
        run_id: currentRunId,
        structure_type: 'new',
        patch
      });

      if (response.success) {
        // Update last synced entry
        setState(prev => ({
          ...prev,
          lastSyncedEntry: history.present,
          isSyncing: false
        }));

        // Optionally clear history after successful save
        // setHistory(prev => ({ ...prev, past: [], future: [] }));
      } else {
        // Handle conflicts
        console.error('Patch conflicts:', response.conflicts);
        // Show conflict resolution UI
      }
    } catch (error) {
      console.error('Failed to save changes:', error);
    } finally {
      setState(prev => ({ ...prev, isSyncing: false }));
    }
  }, [history, state.lastSyncedEntry]);

  return {
    // ... existing returns

    // History management
    executeOperation,
    undo,
    redo,
    canUndo: history.past.length > 0,
    canRedo: history.future.length > 0,

    // Save
    saveChanges,
    isSyncing: state.isSyncing,
    hasUnsavedChanges: history.present !== state.lastSyncedEntry
  };
};
```

### 6. UI Components

#### 6.1 Undo/Redo Toolbar

```typescript
// frontend/src/components/UndoRedoToolbar.tsx

interface UndoRedoToolbarProps {
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  hasUnsavedChanges: boolean;
  onSave: () => void;
  isSaving: boolean;
}

export const UndoRedoToolbar: React.FC<UndoRedoToolbarProps> = ({
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  hasUnsavedChanges,
  onSave,
  isSaving
}) => {
  return (
    <div className="undo-redo-toolbar">
      <button
        onClick={onUndo}
        disabled={!canUndo}
        title="Undo (Ctrl+Z)"
      >
        ↶ Undo
      </button>

      <button
        onClick={onRedo}
        disabled={!canRedo}
        title="Redo (Ctrl+Y)"
      >
        ↷ Redo
      </button>

      <div className="separator" />

      <button
        onClick={onSave}
        disabled={!hasUnsavedChanges || isSaving}
        className={hasUnsavedChanges ? 'has-changes' : ''}
        title="Save changes (Ctrl+S)"
      >
        {isSaving ? 'Saving...' : 'Save'}
        {hasUnsavedChanges && '*'}
      </button>
    </div>
  );
};
```

#### 6.2 Keyboard Shortcuts

```typescript
// frontend/src/hooks/useKeyboardShortcuts.ts

export const useKeyboardShortcuts = (
  undo: () => void,
  redo: () => void,
  save: () => void
) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Z or Cmd+Z: Undo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      }

      // Ctrl+Y or Ctrl+Shift+Z or Cmd+Shift+Z: Redo
      if (
        ((e.ctrlKey || e.metaKey) && e.key === 'y') ||
        ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'z')
      ) {
        e.preventDefault();
        redo();
      }

      // Ctrl+S or Cmd+S: Save
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        save();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo, save]);
};
```

## Implementation Phases

### Phase 1: History Management (Frontend)
- [ ] Implement `HistoryState` and history management logic
- [ ] Integrate with `useFolderTree` hook
- [ ] Add undo/redo actions
- [ ] Modify existing operations to work with `executeOperation`
- [ ] Add keyboard shortcuts
- [ ] Create undo/redo UI toolbar

### Phase 2: Patch System (Frontend)
- [ ] Implement `PatchRecorder` class
- [ ] Integrate patch recording with tree operations
- [ ] Create patch generation utilities
- [ ] Add tree hashing for optimistic locking
- [ ] Implement patch API client

### Phase 3: Patch Application (Backend)
- [ ] Create patch models (`TreePatch`, `PatchOperation`, etc.)
- [ ] Implement `/api/folder-structure` PATCH endpoint
- [ ] Create patch application logic
- [ ] Add tree hashing utility
- [ ] Update `FileMapping` based on patch operations
- [ ] Add conflict detection and resolution

### Phase 4: Integration & Testing
- [ ] Connect frontend save to backend patch API
- [ ] Add conflict resolution UI
- [ ] Add loading/saving states
- [ ] Test all operations with undo/redo
- [ ] Test patch generation and application
- [ ] Test conflict scenarios
- [ ] Add error handling and user feedback

### Phase 5: Polish & Optimization
- [ ] Optimize history size management
- [ ] Add history visualization UI (optional)
- [ ] Implement batch operations
- [ ] Add analytics/telemetry for operations
- [ ] Performance testing with large trees
- [ ] Documentation and user guide

## Edge Cases & Considerations

### 1. Conflict Resolution
When the backend tree has changed since the last sync:
- Detect via tree hash mismatch
- Options:
  - **Pull and retry**: Fetch latest tree, rebase local changes
  - **Force push**: Overwrite backend (with confirmation)
  - **Manual merge**: Show conflict UI

### 2. Memory Management
Large folder structures and long history can consume memory:
- Limit history size (e.g., 50 operations)
- Optionally persist history to localStorage
- Compress old history entries

### 3. Complex Operations
Some operations affect multiple paths:
- Ensure patches capture all changes
- Test operations like "flatten all low-confidence folders"

### 4. Concurrent Edits
Multiple users/tabs editing the same structure:
- Use optimistic locking (tree hash)
- Implement proper conflict resolution
- Consider WebSocket for real-time sync (future)

### 5. Performance
Large trees with many operations:
- Debounce save operations
- Use immutable data structures (immer.js)
- Lazy-load history visualization

## Testing Strategy

### Unit Tests
- Patch generation logic
- Patch application logic
- History state management
- Tree hashing

### Integration Tests
- Full undo/redo flow
- Save and reload
- Conflict detection
- All tree operations with history

### E2E Tests
- User performs series of operations
- User undoes and redoes
- User saves changes
- Handle network errors

## Future Enhancements

1. **Collaborative Editing**: Real-time sync between multiple users
2. **History Visualization**: Timeline view of all operations
3. **Branching**: Create multiple "branches" of the folder structure
4. **Snapshots**: Named checkpoints in history
5. **Auto-save**: Periodic background saving
6. **Operation Batching**: Group related operations
7. **Intelligent Conflict Resolution**: Auto-merge compatible changes

## References

- Current codebase: `/frontend/src/hooks/useFolderTree.ts`
- Tree operations: `/frontend/src/utils/folderTreeOperations.ts`
- Backend API: `/organizer/organizer_api.py`
- Storage models: `/organizer/storage/work_models.py`
