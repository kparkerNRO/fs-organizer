/**
 * Hook for managing dual representation data and operations.
 *
 * This hook provides:
 * - Fetching and caching of dual representation data
 * - Applying hierarchy diffs
 * - Tracking pending changes
 * - Synchronized state management
 */

import { useState, useCallback, useEffect } from "react";
import {
  DualRepresentation,
  HierarchyDiff,
  HierarchyItem,

} from "../types/types";
import { getDualRepresentation, applyHierarchyDiff } from "../api";

export interface DualRepresentationState {
  // Data
  dualRep: DualRepresentation | null;
  isLoading: boolean;
  error: Error | null;

  // Pending changes
  pendingDiff: HierarchyDiff;
  hasPendingChanges: boolean;

  // UI state
  highlightedItemId: string | null;
  selectedView: 'node' | 'category';
}

export interface DualRepresentationActions {
  // Data management
  fetchDualRepresentation: () => Promise<void>;
  refresh: () => Promise<void>;

  // Hierarchy operations
  addToParent: (parentId: string, childId: string) => void;
  removeFromParent: (parentId: string, childId: string) => void;
  moveItem: (itemId: string, fromParentId: string, toParentId: string) => void;

  // Diff management
  applyPendingChanges: () => Promise<void>;
  clearPendingChanges: () => void;

  // UI operations
  highlightItem: (itemId: string | null) => void;
  setView: (view: 'node' | 'category') => void;

  // Utility functions
  getItem: (itemId: string) => HierarchyItem | undefined;
  getChildren: (parentId: string, hierarchy: 'node' | 'category') => string[];
  findItemInBothHierarchies: (itemId: string) => {
    inNodeHierarchy: boolean;
    inCategoryHierarchy: boolean;
    nodeParents: string[];
    categoryParents: string[];
  };
}

export type UseDualRepresentationReturn = DualRepresentationState & DualRepresentationActions;

export const useDualRepresentation = (): UseDualRepresentationReturn => {
  const [state, setState] = useState<DualRepresentationState>({
    dualRep: null,
    isLoading: false,
    error: null,
    pendingDiff: { added: {}, deleted: {} },
    hasPendingChanges: false,
    highlightedItemId: null,
    selectedView: 'node',
  });

  // Fetch dual representation from the API
  const fetchDualRepresentation = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));
    try {
      const data = await getDualRepresentation();
      setState(prev => ({
        ...prev,
        dualRep: data,
        isLoading: false,
        error: null,
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error : new Error('Failed to fetch dual representation'),
      }));
    }
  }, []);

  // Refresh data (alias for fetchDualRepresentation)
  const refresh = useCallback(async () => {
    await fetchDualRepresentation();
  }, [fetchDualRepresentation]);

  // Add a child to a parent in the pending diff
  const addToParent = useCallback((parentId: string, childId: string) => {
    setState(prev => {
      const newDiff = { ...prev.pendingDiff };

      if (!newDiff.added[parentId]) {
        newDiff.added[parentId] = [];
      }

      if (!newDiff.added[parentId].includes(childId)) {
        newDiff.added[parentId] = [...newDiff.added[parentId], childId];
      }

      return {
        ...prev,
        pendingDiff: newDiff,
        hasPendingChanges: true,
      };
    });
  }, []);

  // Remove a child from a parent in the pending diff
  const removeFromParent = useCallback((parentId: string, childId: string) => {
    setState(prev => {
      const newDiff = { ...prev.pendingDiff };

      if (!newDiff.deleted[parentId]) {
        newDiff.deleted[parentId] = [];
      }

      if (!newDiff.deleted[parentId].includes(childId)) {
        newDiff.deleted[parentId] = [...newDiff.deleted[parentId], childId];
      }

      return {
        ...prev,
        pendingDiff: newDiff,
        hasPendingChanges: true,
      };
    });
  }, []);

  // Move an item from one parent to another
  const moveItem = useCallback((itemId: string, fromParentId: string, toParentId: string) => {
    removeFromParent(fromParentId, itemId);
    addToParent(toParentId, itemId);
  }, [removeFromParent, addToParent]);

  // Apply pending changes to the backend
  const applyPendingChanges = useCallback(async () => {
    if (!state.hasPendingChanges) {
      return;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));
    try {
      await applyHierarchyDiff(state.pendingDiff);

      // Clear pending changes and refresh
      setState(prev => ({
        ...prev,
        pendingDiff: { added: {}, deleted: {} },
        hasPendingChanges: false,
        isLoading: false,
      }));

      await fetchDualRepresentation();
    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error : new Error('Failed to apply changes'),
      }));
    }
  }, [state.hasPendingChanges, state.pendingDiff, fetchDualRepresentation]);

  // Clear pending changes without applying
  const clearPendingChanges = useCallback(() => {
    setState(prev => ({
      ...prev,
      pendingDiff: { added: {}, deleted: {} },
      hasPendingChanges: false,
    }));
  }, []);

  // Highlight an item (for synchronized highlighting)
  const highlightItem = useCallback((itemId: string | null) => {
    setState(prev => ({ ...prev, highlightedItemId: itemId }));
  }, []);

  // Set the current view (node or category)
  const setView = useCallback((view: 'node' | 'category') => {
    setState(prev => ({ ...prev, selectedView: view }));
  }, []);

  // Get an item by ID
  const getItem = useCallback((itemId: string): HierarchyItem | undefined => {
    return state.dualRep?.items[itemId];
  }, [state.dualRep]);

  // Get children of a parent
  const getChildren = useCallback((parentId: string, hierarchy: 'node' | 'category'): string[] => {
    if (!state.dualRep) return [];

    const hierarchyData = hierarchy === 'node'
      ? state.dualRep.node_hierarchy
      : state.dualRep.category_hierarchy;

    return hierarchyData[parentId] || [];
  }, [state.dualRep]);

  // Find an item in both hierarchies
  const findItemInBothHierarchies = useCallback((itemId: string) => {
    const result = {
      inNodeHierarchy: false,
      inCategoryHierarchy: false,
      nodeParents: [] as string[],
      categoryParents: [] as string[],
    };

    if (!state.dualRep) return result;

    // Check node hierarchy
    for (const [parentId, children] of Object.entries(state.dualRep.node_hierarchy)) {
      if (children.includes(itemId)) {
        result.inNodeHierarchy = true;
        result.nodeParents.push(parentId);
      }
    }

    // Check category hierarchy
    for (const [parentId, children] of Object.entries(state.dualRep.category_hierarchy)) {
      if (children.includes(itemId)) {
        result.inCategoryHierarchy = true;
        result.categoryParents.push(parentId);
      }
    }

    return result;
  }, [state.dualRep]);

  // Auto-fetch on mount
  useEffect(() => {
    fetchDualRepresentation();
  }, [fetchDualRepresentation]);

  return {
    ...state,
    fetchDualRepresentation,
    refresh,
    addToParent,
    removeFromParent,
    moveItem,
    applyPendingChanges,
    clearPendingChanges,
    highlightItem,
    setView,
    getItem,
    getChildren,
    findItemInBothHierarchies,
  };
};
