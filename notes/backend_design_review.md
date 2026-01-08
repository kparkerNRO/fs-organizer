# Backend Design Review - Final

**Document Purpose**: To provide a final analysis of the backend implementation against the clarified and updated design specification in `notes/dual_representation_design.md`.

## 1. Summary

Following a detailed design discussion, the target architecture has been clarified. The goal is a "git-like" model that separates intrinsic, shared data (`ItemStore`) from contextual, tree-specific data (`Hierarchy` and `HierarchyRecord`). The `notes/dual_representation_design.md` document has been updated to serve as the new source of truth for this model.

The current backend implementation is **partially compliant** with this clarified design. It correctly implements the nested `HierarchyRecord` structure and the flexible, stage-keyed `hierarchies` dictionary. However, a critical deviation remains that prevents the system from meeting its core user story.

## 2. Key Deviation from Final Design

The backend implementation incorrectly assigns data ownership between the shared `ItemStore` and the contextual `HierarchyRecord`.

*   **Clarified Design**: The `name` of an item must be a **contextual property** that can differ between hierarchies. Therefore, it must reside in the `HierarchyRecord`. The `HierarchyRecord` uses an `itemId` to link back to the shared, intrinsic data in the `ItemStore`.

*   **Actual Implementation**: The backend currently places the `name` property inside the shared `HierarchyItem` model. The `HierarchyRecord` uses the item's full ID (`node-123`) as its own ID, rather than having a separate `itemId` field.

### Consequence of this Deviation

This design flaw makes the primary user story impossible to implement. Because `name` is in the shared `ItemStore`, renaming an item in one hierarchy would incorrectly rename it in all other hierarchies. The system cannot support a "mutable" tree with names that differ from the "original" tree.

## 3. Required Refactoring

To become compliant with the final design, the backend implementation (`organizer/utils/dual_representation.py` and `organizer/api/models.py`) must be refactored:

1.  **Update `HierarchyRecord` model**:
    *   Remove the `id` field.
    *   Add an `itemId: str` field to reference the `ItemStore`.
    *   Add a `name: str` field to hold the contextual name.

2.  **Update `HierarchyItem` model**:
    *   Remove the `name` field. It is no longer a shared property.

3.  **Update `build_dual_representation` logic**:
    *   When creating the `HierarchyRecord` tree, the logic must populate the new `name` field from the appropriate source (e.g., `Node.name` for the original tree, or potentially another source for a categorized tree).
    *   The `itemId` field must be populated with the correct ID that links to the `ItemStore` (e.g., `"node-123"`).
