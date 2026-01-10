**Note:** These test cases validate the backend implementation based on the revised design specified in `notes/dual_representation_design.md`. This design uses a nested `HierarchyRecord` object for hierarchy entries to support metadata, rather than a simple flattened map of ID arrays.

# Test Cases: Backend Dual Representation

This document provides a suite of test cases to validate the backend implementation of the dual representation feature.

## 1. Data & Hierarchy Construction Logic (`utils/dual_representation.py`)

These tests validate the core logic that transforms database records into the `DualRepresentation` structure.

---

### **TC-BUILD-01:** Empty snapshot
- **Given:** An empty `Snapshot` exists with no associated `Node` records.
- **When:** `build_dual_representation` is called for the `original` stage.
- **Then:** 
  - The `items` store contains only the root item for the hierarchy's root record.
  - The `hierarchies.original` tree has no children.

---

### **TC-BUILD-02:** Single root node
- **Given:** A `Snapshot` contains a single file `Node` with no parent.
- **When:** `build_dual_representation` is called for the `original` stage.
- **Then:**
  - The `items` store contains the root item and one `node-*` item.
  - The `hierarchies.original` root has one child corresponding to the file node.

---

### **TC-BUILD-03:** Deeply nested node hierarchy
- **Given:** A `Snapshot` contains a nested folder structure (e.g., `dir1/dir2/file.txt`).
- **When:** `build_dual_representation` is called for the `original` stage.
- **Then:** The returned `DualRepresentation` object should reflect the following:

  **1. `items` Store Verification:**
  All nodes are present in the flattened `items` store. The `HierarchyItem` should only contain intrinsic, shared data.

  *Example for `dir1/dir2/file.txt`:*
  ```json
  {
    "items": {
      "root-item": { "id": "root-item", "type": "node", ... },
      "node-1": { "id": "node-1", "type": "node", "originalPath": "/path/to/dir1", ... },
      "node-2": { "id": "node-2", "type": "node", "originalPath": "/path/to/dir1/dir2", ... },
      "node-3": { "id": "node-3", "type": "node", "originalPath": "/path/to/dir1/dir2/file.txt", ... }
    },
    "hierarchies": { ... }
  }
  ```

  **2. `hierarchies` Tree Verification:**
  The `hierarchies.original` tree contains contextual data, including the `name`. Each `HierarchyRecord` links to the `ItemStore` via `itemId`.

  *Example for `dir1/dir2/file.txt`:*
  ```json
  {
    "items": { ... },
    "hierarchies": {
      "original": {
        "stage": "original",
        "root": {
          "itemId": "root-item",
          "name": "root",
          "children": [
            {
              "itemId": "node-1",
              "name": "dir1",
              "children": [
                {
                  "itemId": "node-2",
                  "name": "dir2",
                  "children": [
                    {
                      "itemId": "node-3",
                      "name": "file.txt",
                      "children": []
                    }
                  ]
                }
              ]
            }
          ]
        }
      }
    }
  }
  ```

  **What happens if `dir2` has siblings?**

  - **Given:** A snapshot with `dir1/dir2/file.txt` and `dir1/dir3/another.log`.

  - **Expected Output:** The `children` array of the `HierarchyRecord` for `dir1` now contains two objects, correctly reflecting the sibling relationship.
    ```json
    {
        "itemId": "node-1",
        "name": "dir1",
        "children": [
            {
                "itemId": "node-2",
                "name": "dir2",
                "children": [
                    { "itemId": "node-3", "name": "file.txt", "children": [] }
                ]
            },
            {
                "itemId": "node-4",
                "name": "dir3",
                "children": [
                    { "itemId": "node-5", "name": "another.log", "children": [] }
                ]
            }
        ]
    }
    ```

---

### **TC-BUILD-04:** No categories exist
- **Given:** A `Run` exists, but there are no `GroupCategory` or `GroupCategoryEntry` records.
- **When:** `build_dual_representation` is called for the `organized` stage.
- **Then:**
  - The `items` store contains only the root item.
  - The `hierarchies.organized` tree has no children.

---

### **TC-BUILD-05:** Simple category hierarchy
- **Given:** A `Run` has one `GroupCategory` which contains two file `Node`s via `GroupCategoryEntry` records.
- **When:** `build_dual_representation` is called for the `organized` stage.
- **Then:**
  - The `items` store contains the root item, one `category-*` item, and two `node-*` items.
  - The `hierarchies.organized` root has one category child, which in turn has two node children.

---

### **TC-BUILD-06:** Uncategorized nodes and pruned ItemStore
- **Given:** A `Run` has nodes that are assigned to a category and nodes that are not.
- **When:** `build_dual_representation` is called for the **`organized` stage only**.
- **Then:**
  - Uncategorized nodes are absent from the `hierarchies.organized` tree.
  - The `ItemStore` **only contains items that are part of the `organized` hierarchy**. It must **not** contain `HierarchyItem`s for the uncategorized nodes.
  - Uncategorized nodes are not included in the `items` store

---

### **TC-BUILD-07:** ItemStore as a Union of Requested Hierarchies
- **Given:** Data exists for both a `Snapshot` and a categorized `Run`, with some nodes appearing in both the `original` and `organized` hierarchies, and some only in `original`.
- **When:** `build_dual_representation` is called for both `original` and `organized` stages.
- **Then:**
  - The `ItemStore` contains the **union** of all items required for both hierarchies.
  - An item for a node that is uncategorized (and thus not in the `organized` hierarchy) **will still be included** in the `ItemStore` because it is required by the `original` hierarchy.
  - The `hierarchies` dictionary contains both `original` and `organized` keys, each with a valid, distinct tree structure.

---

### **TC-BUILD-08:** Data integrity and metadata
- **Given:** Nodes and Categories have associated metadata (e.g., paths, confidence scores).
- **When:** `build_dual_representation` is called.
- **Then:** 
  - The `HierarchyItem` objects in the `items` store correctly reflect shared metadata like `originalPath`.
  - Tree-specific metadata should be present in the `metadata` field of the `HierarchyRecord`.

---
---

## 2. API Endpoint Validation (`GET /api/v2/folder-structure`)

These tests validate the behavior of the public-facing API endpoint.

---

### **TC-API-01:** No runs exist in DB
- **Given:** The `run` table in the database is empty.
- **When:** A `GET` request is made to `/api/v2/folder-structure`.
- **Then:** The API returns an **HTTP 404 Not Found** status with a descriptive error message.

---

### **TC-API-02:** Invalid `stages` parameter
- **Given:** A valid `Run` exists.
- **When:** A `GET` request is made with an invalid stage name, e.g., `?stages=foo`.
- **Then:** The API returns an **HTTP 400 Bad Request** status with an error message listing valid stages.

---

### **TC-API-03:** Default `stages` parameter
- **Given:** A valid `Run` with both node and category data exists.
- **When:** A `GET` request is made with no `stages` query parameter.
- **Then:**
  - The API returns **HTTP 200 OK**.
  - The response body contains hierarchies for both `original` and `organized` stages by default.

---

### **TC-API-04:** Single stage request
- **Given:** A valid `Run` exists.
- **When:** A `GET` request is made for a single stage, e.g., `?stages=original`.
- **Then:**
  - The API returns **HTTP 200 OK**.
  - The response's `hierarchies` dictionary contains **only** the `original` key.

---

### **TC-API-05:** Multiple specific stages
- **Given:** A valid `Run` exists.
- **When:** A `GET` request is made for multiple valid stages, e.g., `?stages=organized,original`.
- **Then:**
  - The API returns **HTTP 200 OK**.
  - The response's `hierarchies` dictionary contains **both** `organized` and `original` keys.

---

### **TC-API-06:** Graceful error on build failure
- **Given:** A valid `Run` exists.
- **When:** The `build_dual_representation` function is mocked to raise a database exception.
- **Then:** The API returns an **HTTP 500 Internal Server Error** status with a descriptive error message.