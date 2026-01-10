import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useDualRepresentation } from "./useDualRepresentation";
import * as api from "../api";

// Mock the API module
vi.mock("../api");

const mockDualRepresentation = {
  items: {
    "node-root": { id: "node-root", name: "root", type: "node" as const },
    "node-1": {
      id: "node-1",
      name: "Documents",
      type: "node" as const,
      originalPath: "/Documents",
    },
    "category-root": {
      id: "category-root",
      name: "Categories",
      type: "category" as const,
    },
    "category-1": { id: "category-1", name: "Work", type: "category" as const },
  },
  node_hierarchy: {
    "node-root": ["node-1"],
  },
  category_hierarchy: {
    "category-root": ["category-1"],
  },
};

describe("useDualRepresentation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("initializes with default state", () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    expect(result.current.dualRep).toBeNull();
    // isLoading is true because the hook auto-fetches on mount
    expect(result.current.isLoading).toBe(true);
    expect(result.current.error).toBeNull();
    expect(result.current.hasPendingChanges).toBe(false);
    expect(result.current.highlightedItemId).toBeNull();
    expect(result.current.selectedView).toBe("node");
  });

  it("fetches dual representation on mount", async () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    expect(result.current.dualRep).toEqual(mockDualRepresentation);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("handles fetch errors", async () => {
    const error = new Error("Failed to fetch");
    vi.mocked(api.getDualRepresentation).mockRejectedValue(error);

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.error).not.toBeNull();
    });

    expect(result.current.error?.message).toBe("Failed to fetch");
    expect(result.current.dualRep).toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  it("can refresh data", async () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    // Clear the mock call count
    vi.clearAllMocks();

    // Call refresh
    await act(async () => {
      await result.current.refresh();
    });

    expect(api.getDualRepresentation).toHaveBeenCalledTimes(1);
  });

  it("adds items to parent in pending diff", () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    act(() => {
      result.current.addToParent("category-1", "node-1");
    });

    expect(result.current.pendingDiff.added["category-1"]).toContain("node-1");
    expect(result.current.hasPendingChanges).toBe(true);
  });

  it("removes items from parent in pending diff", () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    act(() => {
      result.current.removeFromParent("category-1", "node-1");
    });

    expect(result.current.pendingDiff.deleted["category-1"]).toContain(
      "node-1",
    );
    expect(result.current.hasPendingChanges).toBe(true);
  });

  it("moves items between parents", () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    act(() => {
      result.current.moveItem("node-1", "category-1", "category-2");
    });

    expect(result.current.pendingDiff.deleted["category-1"]).toContain(
      "node-1",
    );
    expect(result.current.pendingDiff.added["category-2"]).toContain("node-1");
    expect(result.current.hasPendingChanges).toBe(true);
  });

  it("does not add duplicate children to pending diff", () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    act(() => {
      result.current.addToParent("category-1", "node-1");
      result.current.addToParent("category-1", "node-1"); // Add same item again
    });

    // Should only contain one instance
    expect(result.current.pendingDiff.added["category-1"]).toEqual(["node-1"]);
  });

  it("applies pending changes and refreshes", async () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );
    vi.mocked(api.applyHierarchyDiff).mockResolvedValue({
      message: "Success",
      log_id: 1,
    });

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    act(() => {
      result.current.addToParent("category-1", "node-1");
    });

    expect(result.current.hasPendingChanges).toBe(true);

    await act(async () => {
      await result.current.applyPendingChanges();
    });

    expect(api.applyHierarchyDiff).toHaveBeenCalledWith({
      added: { "category-1": ["node-1"] },
      deleted: {},
    });
    expect(result.current.hasPendingChanges).toBe(false);
    expect(result.current.pendingDiff).toEqual({ added: {}, deleted: {} });
  });

  it("does not apply when there are no pending changes", async () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );
    vi.mocked(api.applyHierarchyDiff).mockResolvedValue({
      message: "Success",
      log_id: 1,
    });

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    await act(async () => {
      await result.current.applyPendingChanges();
    });

    expect(api.applyHierarchyDiff).not.toHaveBeenCalled();
  });

  it("handles errors when applying changes", async () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );
    vi.mocked(api.applyHierarchyDiff).mockRejectedValue(
      new Error("Apply failed"),
    );

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    act(() => {
      result.current.addToParent("category-1", "node-1");
    });

    await act(async () => {
      await result.current.applyPendingChanges();
    });

    expect(result.current.error?.message).toBe("Apply failed");
    expect(result.current.hasPendingChanges).toBe(true); // Should keep changes on error
  });

  it("clears pending changes", () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    act(() => {
      result.current.addToParent("category-1", "node-1");
      result.current.removeFromParent("category-2", "node-2");
    });

    expect(result.current.hasPendingChanges).toBe(true);

    act(() => {
      result.current.clearPendingChanges();
    });

    expect(result.current.hasPendingChanges).toBe(false);
    expect(result.current.pendingDiff).toEqual({ added: {}, deleted: {} });
  });

  it("highlights items", () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    act(() => {
      result.current.highlightItem("node-1");
    });

    expect(result.current.highlightedItemId).toBe("node-1");

    act(() => {
      result.current.highlightItem(null);
    });

    expect(result.current.highlightedItemId).toBeNull();
  });

  it("sets the current view", () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    expect(result.current.selectedView).toBe("node");

    act(() => {
      result.current.setView("category");
    });

    expect(result.current.selectedView).toBe("category");
  });

  it("gets an item by ID", async () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    const item = result.current.getItem("node-1");

    expect(item).toEqual({
      id: "node-1",
      name: "Documents",
      type: "node",
      originalPath: "/Documents",
    });
  });

  it("returns undefined for non-existent item", async () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    const item = result.current.getItem("non-existent");

    expect(item).toBeUndefined();
  });

  it("gets children of a parent in node hierarchy", async () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    const children = result.current.getChildren("node-root", "node");

    expect(children).toEqual(["node-1"]);
  });

  it("gets children of a parent in category hierarchy", async () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    const children = result.current.getChildren("category-root", "category");

    expect(children).toEqual(["category-1"]);
  });

  it("returns empty array for non-existent parent", async () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    const children = result.current.getChildren("non-existent", "node");

    expect(children).toEqual([]);
  });

  it("finds item in both hierarchies", async () => {
    const dualRep = {
      items: {
        "node-1": { id: "node-1", name: "File", type: "node" as const },
      },
      node_hierarchy: {
        "node-root": ["node-1"],
      },
      category_hierarchy: {
        "category-1": ["node-1"],
        "category-2": ["node-1"],
      },
    };

    vi.mocked(api.getDualRepresentation).mockResolvedValue(dualRep);

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    const location = result.current.findItemInBothHierarchies("node-1");

    expect(location.inNodeHierarchy).toBe(true);
    expect(location.inCategoryHierarchy).toBe(true);
    expect(location.nodeParents).toEqual(["node-root"]);
    expect(location.categoryParents).toEqual(["category-1", "category-2"]);
  });

  it("handles item not in any hierarchy", async () => {
    vi.mocked(api.getDualRepresentation).mockResolvedValue(
      mockDualRepresentation,
    );

    const { result } = renderHook(() => useDualRepresentation());

    await waitFor(() => {
      expect(result.current.dualRep).not.toBeNull();
    });

    const location = result.current.findItemInBothHierarchies("non-existent");

    expect(location.inNodeHierarchy).toBe(false);
    expect(location.inCategoryHierarchy).toBe(false);
    expect(location.nodeParents).toEqual([]);
    expect(location.categoryParents).toEqual([]);
  });
});
