import { useState, useCallback, useMemo } from "react";
import { FolderV2 } from "../types/types";
import {
  FolderTreeNode,
  FolderTreePath,
  FolderTreeOperation,
  FolderTreeOperationResult,
  renameNode,
  mergeFolders,
  moveNode,
  findNodeByPath,
  findParentByPath,
  flattenFolders,
  invertFolder,
  deleteFolders,
  createFolder,
  isFileNode,
  getFilePathInTree,
  getFoldersAtSameLevel,
  hasLowConfidenceChildren,
  setConfidenceToMax,
  buildNodePath,
} from "../utils/folderTreeOperations";

export interface FolderTreeState {
  // Tree data
  originalTree: FolderV2 | null;
  modifiedTree: FolderV2 | null;
  hasModifications: boolean;

  // UI state
  expandedFolders: Set<string>;
  selectedFileId: number | null;
  selectedFolderPaths: string[];

  // Operation state
  isOperationInProgress: boolean;
  lastOperation: FolderTreeOperation | null;
  operationHistory: FolderTreeOperation[];
}

export interface FolderTreeActions {
  // Tree management
  setTreeData: (tree: FolderV2 | null) => void;
  resetToOriginal: () => void;

  // Node operations
  renameItem: (
    targetPath: FolderTreePath,
    newName: string,
  ) => Promise<FolderTreeOperationResult>;
  createFolder: (
    parentPath: FolderTreePath,
    folderName: string,
  ) => Promise<FolderTreeOperationResult>;
  mergeItems: (
    sourcePaths: FolderTreePath[],
    targetName: string,
  ) => Promise<FolderTreeOperationResult>;
  moveItem: (
    sourcePath: FolderTreePath,
    targetPath: FolderTreePath,
  ) => Promise<FolderTreeOperationResult>;
  deleteItems: (
    targetPaths: FolderTreePath[],
  ) => Promise<FolderTreeOperationResult>;
  flattenItems: (
    sourcePaths: FolderTreePath[],
  ) => Promise<FolderTreeOperationResult>;
  invertItems: (
    targetPath: FolderTreePath,
  ) => Promise<FolderTreeOperationResult>;

  // Selection management
  selectFile: (fileId: number | null) => void;
  selectFolder: (folderPath: string | null) => void;
  selectMultipleFolders: (folderPaths: string[]) => void;
  selectFolderRange: (startPath: string, endPath: string) => void;
  clearSelection: () => void;

  // Tree navigation
  expandFolder: (folderPath: string) => void;
  collapseFolder: (folderPath: string) => void;
  toggleFolder: (folderPath: string) => void;
  expandToFile: (fileId: number) => void;
  expandAllParents: (nodePath: string) => void;

  // Utility functions
  findNode: (targetPath: FolderTreePath) => FolderTreeNode | null;
  getNodeParent: (targetPath: FolderTreePath) => {
    parent: FolderV2 | null;
    parentPath: string;
  };
  isNodeSelected: (node: FolderTreeNode, nodePath: string) => boolean;
  hasLowConfidenceChildrenInPath: (folderPath: string) => boolean;
  setFolderConfidence: (folderPath: string, confidence: number) => void;
  getFoldersAtLevel: (folderPath: string) => string[];
}

export type UseFolderTreeReturn = FolderTreeState & FolderTreeActions;

export const useFolderTree = (): UseFolderTreeReturn => {
  // State management
  const [state, setState] = useState<FolderTreeState>({
    originalTree: null,
    modifiedTree: null,
    hasModifications: false,
    expandedFolders: new Set(),
    selectedFileId: null,
    selectedFolderPaths: [],
    isOperationInProgress: false,
    lastOperation: null,
    operationHistory: [],
  });

  // Get the current active tree (modified if available, otherwise original)
  const activeTree = useMemo(() => {
    return state.modifiedTree || state.originalTree;
  }, [state.modifiedTree, state.originalTree]);

  // Tree management actions
  const setTreeData = useCallback((tree: FolderV2 | null) => {
    setState((prev) => ({
      ...prev,
      originalTree: tree,
      modifiedTree: null,
      hasModifications: false,
      selectedFileId: null,
      selectedFolderPaths: [],
      operationHistory: [],
    }));
  }, []);

  const resetToOriginal = useCallback(() => {
    setState((prev) => ({
      ...prev,
      modifiedTree: null,
      hasModifications: false,
      operationHistory: [],
    }));
  }, []);

  // Node operations
  const renameItem = useCallback(
    async (
      targetPath: FolderTreePath,
      newName: string,
    ): Promise<FolderTreeOperationResult> => {
      if (!activeTree) {
        return { success: false, error: "No tree data available" };
      }

      setState((prev) => ({ ...prev, isOperationInProgress: true }));

      try {
        const result = renameNode(activeTree, targetPath, newName);

        if (result.success && result.newTree) {
          const operation: FolderTreeOperation = {
            type: "rename",
            sourcePath: targetPath,
            newName,
          };

          setState((prev) => ({
            ...prev,
            modifiedTree: result.newTree!,
            hasModifications: true,
            lastOperation: operation,
            operationHistory: [...prev.operationHistory, operation],
            isOperationInProgress: false,
            selectedFolderPaths: [], // Clear selection after tree modification
            selectedFileId: null, // Clear file selection too
          }));
        } else {
          setState((prev) => ({ ...prev, isOperationInProgress: false }));
        }

        return result;
      } catch {
        setState((prev) => ({ ...prev, isOperationInProgress: false }));
        return { success: false, error: "Failed to rename item" };
      }
    },
    [activeTree],
  );

  const mergeItems = useCallback(
    async (
      sourcePaths: FolderTreePath[],
      targetName: string,
    ): Promise<FolderTreeOperationResult> => {
      if (!activeTree) {
        return { success: false, error: "No tree data available" };
      }

      setState((prev) => ({ ...prev, isOperationInProgress: true }));

      try {
        const result = mergeFolders(activeTree, sourcePaths);

        if (result.success && result.newTree) {
          const operation: FolderTreeOperation = {
            type: "merge",
            sourcePath: sourcePaths[0], // Primary source
            targetNodes: sourcePaths
              .map((path) => findNodeByPath(activeTree, path))
              .filter(Boolean) as FolderTreeNode[],
            newName: targetName,
          };

          setState((prev) => ({
            ...prev,
            modifiedTree: result.newTree!,
            hasModifications: true,
            lastOperation: operation,
            operationHistory: [...prev.operationHistory, operation],
            isOperationInProgress: false,
            selectedFolderPaths: [], // Clear selection after tree modification
            selectedFileId: null, // Clear file selection too
          }));
        } else {
          setState((prev) => ({ ...prev, isOperationInProgress: false }));
        }

        return result;
      } catch {
        setState((prev) => ({ ...prev, isOperationInProgress: false }));
        return { success: false, error: "Failed to merge items" };
      }
    },
    [activeTree],
  );

  const moveItem = useCallback(
    async (
      sourcePath: FolderTreePath,
      targetPath: FolderTreePath,
    ): Promise<FolderTreeOperationResult> => {
      if (!activeTree) {
        return { success: false, error: "No tree data available" };
      }

      setState((prev) => ({ ...prev, isOperationInProgress: true }));

      try {
        const result = moveNode(activeTree, sourcePath, targetPath);

        if (result.success && result.newTree) {
          const operation: FolderTreeOperation = {
            type: "move",
            sourcePath,
            targetPath,
          };

          setState((prev) => ({
            ...prev,
            modifiedTree: result.newTree!,
            hasModifications: true,
            lastOperation: operation,
            operationHistory: [...prev.operationHistory, operation],
            isOperationInProgress: false,
            selectedFolderPaths: [], // Clear selection after tree modification
            selectedFileId: null, // Clear file selection too
          }));
        } else {
          setState((prev) => ({ ...prev, isOperationInProgress: false }));
        }

        return result;
      } catch {
        setState((prev) => ({ ...prev, isOperationInProgress: false }));
        return { success: false, error: "Failed to move item" };
      }
    },
    [activeTree],
  );

  const flattenItems = useCallback(
    async (
      sourcePaths: FolderTreePath[],
    ): Promise<FolderTreeOperationResult> => {
      if (!activeTree) {
        return { success: false, error: "No tree data available" };
      }

      setState((prev) => ({ ...prev, isOperationInProgress: true }));

      try {
        const result = flattenFolders(activeTree, sourcePaths);

        if (result.success && result.newTree) {
          const operation: FolderTreeOperation = {
            type: "flatten",
            sourcePath: sourcePaths[0], // Primary source
            targetNodes: sourcePaths
              .map((path) => findNodeByPath(activeTree, path))
              .filter(Boolean) as FolderTreeNode[],
          };

          setState((prev) => ({
            ...prev,
            modifiedTree: result.newTree!,
            hasModifications: true,
            lastOperation: operation,
            operationHistory: [...prev.operationHistory, operation],
            isOperationInProgress: false,
            selectedFolderPaths: [], // Clear selection after tree modification
            selectedFileId: null, // Clear file selection too
          }));
        } else {
          setState((prev) => ({ ...prev, isOperationInProgress: false }));
        }

        return result;
      } catch {
        setState((prev) => ({ ...prev, isOperationInProgress: false }));
        return { success: false, error: "Failed to flatten items" };
      }
    },
    [activeTree],
  );

  const deleteItems = useCallback(
    async (
      targetPaths: FolderTreePath[],
    ): Promise<FolderTreeOperationResult> => {
      if (!activeTree) {
        return { success: false, error: "No tree data available" };
      }

      setState((prev) => ({ ...prev, isOperationInProgress: true }));

      try {
        const result = deleteFolders(activeTree, targetPaths);

        if (result.success && result.newTree) {
          const operation: FolderTreeOperation = {
            type: "delete",
            sourcePath: targetPaths[0], // Use first path as primary
          };

          setState((prev) => ({
            ...prev,
            modifiedTree: result.newTree!,
            hasModifications: true,
            lastOperation: operation,
            operationHistory: [...prev.operationHistory, operation],
            isOperationInProgress: false,
            selectedFolderPaths: [], // Clear selection after tree modification
            selectedFileId: null, // Clear file selection too
          }));
        } else {
          setState((prev) => ({ ...prev, isOperationInProgress: false }));
        }

        return result;
      } catch {
        setState((prev) => ({ ...prev, isOperationInProgress: false }));
        return { success: false, error: "Failed to delete item" };
      }
    },
    [activeTree],
  );

  const invertItems = useCallback(
    async (targetPath: FolderTreePath): Promise<FolderTreeOperationResult> => {
      if (!activeTree) {
        return { success: false, error: "No tree data available" };
      }

      setState((prev) => ({ ...prev, isOperationInProgress: true }));

      try {
        const result = invertFolder(activeTree, targetPath);

        if (result.success && result.newTree) {
          const operation: FolderTreeOperation = {
            type: "invert",
            sourcePath: targetPath,
          };

          setState((prev) => ({
            ...prev,
            modifiedTree: result.newTree!,
            hasModifications: true,
            lastOperation: operation,
            operationHistory: [...prev.operationHistory, operation],
            isOperationInProgress: false,
            selectedFolderPaths: [], // Clear selection after tree modification
            selectedFileId: null, // Clear file selection too
          }));
        } else {
          setState((prev) => ({ ...prev, isOperationInProgress: false }));
        }

        return result;
      } catch {
        setState((prev) => ({ ...prev, isOperationInProgress: false }));
        return { success: false, error: "Failed to invert folder" };
      }
    },
    [activeTree],
  );

  const createFolderMethod = useCallback(
    async (
      parentPath: FolderTreePath,
      folderName: string,
    ): Promise<FolderTreeOperationResult> => {
      if (!activeTree) {
        return { success: false, error: "No tree data available" };
      }

      setState((prev) => ({ ...prev, isOperationInProgress: true }));

      try {
        const result = createFolder(activeTree, parentPath, folderName);

        if (result.success && result.newTree) {
          const operation: FolderTreeOperation = {
            type: "create",
            sourcePath: parentPath,
            newName: folderName,
          };

          setState((prev) => ({
            ...prev,
            modifiedTree: result.newTree!,
            hasModifications: true,
            lastOperation: operation,
            operationHistory: [...prev.operationHistory, operation],
            isOperationInProgress: false,
            selectedFolderPaths: [], // Clear selection after tree modification
            selectedFileId: null, // Clear file selection too
          }));
        } else {
          setState((prev) => ({ ...prev, isOperationInProgress: false }));
        }

        return result;
      } catch {
        setState((prev) => ({ ...prev, isOperationInProgress: false }));
        return { success: false, error: "Failed to create folder" };
      }
    },
    [activeTree],
  );

  // Utility functions
  const findNode = useCallback(
    (targetPath: FolderTreePath): FolderTreeNode | null => {
      return activeTree ? findNodeByPath(activeTree, targetPath) : null;
    },
    [activeTree],
  );

  const getNodeParent = useCallback(
    (targetPath: FolderTreePath) => {
      return activeTree
        ? findParentByPath(activeTree, targetPath)
        : { parent: null, parentPath: "" };
    },
    [activeTree],
  );

  // Selection management
  const selectFile = useCallback((fileId: number | null) => {
    setState((prev) => ({
      ...prev,
      selectedFileId: fileId,
      selectedFolderPaths: fileId ? [] : prev.selectedFolderPaths,
    }));
  }, []);

  const selectFolder = useCallback((folderPath: string | null) => {
    setState((prev) => ({
      ...prev,
      selectedFolderPaths: folderPath ? [folderPath] : [],
      selectedFileId: folderPath ? null : prev.selectedFileId,
    }));
  }, []);

  const selectMultipleFolders = useCallback((folderPaths: string[]) => {
    setState((prev) => ({
      ...prev,
      selectedFolderPaths: folderPaths,
      selectedFileId: folderPaths.length > 0 ? null : prev.selectedFileId,
    }));
  }, []);

  const selectFolderRange = useCallback(
    (startPath: string, endPath: string) => {
      if (!activeTree) return;

      // Get folders at same level using helper
      const sameLevelFolders = getFoldersAtSameLevel(activeTree, startPath);
      const startIndex = sameLevelFolders.indexOf(startPath);
      const endIndex = sameLevelFolders.indexOf(endPath);

      if (startIndex !== -1 && endIndex !== -1) {
        const start = Math.min(startIndex, endIndex);
        const end = Math.max(startIndex, endIndex);
        const range = sameLevelFolders.slice(start, end + 1);

        setState((prev) => ({
          ...prev,
          selectedFolderPaths: range,
          selectedFileId: null,
        }));
      }
    },
    [activeTree],
  );

  const clearSelection = useCallback(() => {
    setState((prev) => ({
      ...prev,
      selectedFileId: null,
      selectedFolderPaths: [],
    }));
  }, []);

  // Tree navigation
  const expandFolder = useCallback((folderPath: string) => {
    setState((prev) => ({
      ...prev,
      expandedFolders: new Set([...prev.expandedFolders, folderPath]),
    }));
  }, []);

  const collapseFolder = useCallback((folderPath: string) => {
    setState((prev) => {
      const newSet = new Set(prev.expandedFolders);
      newSet.delete(folderPath);
      return { ...prev, expandedFolders: newSet };
    });
  }, []);

  const toggleFolder = useCallback((folderPath: string) => {
    setState((prev) => {
      const newSet = new Set(prev.expandedFolders);
      if (newSet.has(folderPath)) {
        newSet.delete(folderPath);
      } else {
        newSet.add(folderPath);
      }
      return { ...prev, expandedFolders: newSet };
    });
  }, []);

  const expandToFile = useCallback(
    (fileId: number) => {
      if (!activeTree) return;

      const path = getFilePathInTree(activeTree, fileId);
      if (path) {
        setState((prev) => {
          const newSet = new Set(prev.expandedFolders);
          let currentPath = "";
          for (const segment of path) {
            currentPath = currentPath ? `${currentPath}/${segment}` : segment;
            newSet.add(currentPath);
          }
          return { ...prev, expandedFolders: newSet };
        });
      }
    },
    [activeTree],
  );

  const expandAllParents = useCallback(
    (nodePath: string) => {
      if (!activeTree) return;

      setState((prev) => {
        const newSet = new Set(prev.expandedFolders);
        const pathParts = nodePath.split("/");
        let currentPath = "";

        for (const part of pathParts) {
          currentPath = currentPath ? `${currentPath}/${part}` : part;
          newSet.add(currentPath);
        }

        return { ...prev, expandedFolders: newSet };
      });
    },
    [activeTree],
  );

  // Additional utility functions
  const isNodeSelected = useCallback(
    (node: FolderTreeNode, nodePath: string): boolean => {
      if (isFileNode(node)) {
        return state.selectedFileId === node.id;
      } else {
        return state.selectedFolderPaths.includes(nodePath);
      }
    },
    [state.selectedFileId, state.selectedFolderPaths],
  );

  const hasLowConfidenceChildrenInPath = useCallback(
    (folderPath: string): boolean => {
      const folder = findNode(folderPath);
      if (!folder || isFileNode(folder)) return false;

      return hasLowConfidenceChildren(folder as FolderV2);
    },
    [findNode],
  );

  const setFolderConfidence = useCallback(
    (folderPath: string, confidence: number) => {
      if (!activeTree) return;

      const newTree = { ...activeTree } as FolderV2;
      const folder = findNodeByPath(newTree, folderPath);
      if (folder && !isFileNode(folder)) {
        if (confidence === 1) {
          setConfidenceToMax(folder);
        } else {
          (folder as FolderV2).confidence = confidence;
        }

        setState((prev) => ({
          ...prev,
          modifiedTree: newTree,
          hasModifications: true,
        }));
      }
    },
    [activeTree],
  );

  const getFoldersAtLevel = useCallback(
    (folderPath: string): string[] => {
      if (!activeTree) return [];
      return getFoldersAtSameLevel(activeTree, folderPath);
    },
    [activeTree],
  );

  return {
    // State
    ...state,

    // Actions
    setTreeData,
    resetToOriginal,
    renameItem,
    createFolder: createFolderMethod,
    moveItem,
    mergeItems,
    deleteItems,
    flattenItems,
    invertItems,

    // Selection management
    selectFile,
    selectFolder,
    selectMultipleFolders,
    selectFolderRange,
    clearSelection,

    // Tree navigation
    expandFolder,
    collapseFolder,
    toggleFolder,
    expandToFile,
    expandAllParents,

    // Utility functions
    findNode,
    getNodeParent,
    isNodeSelected,
    hasLowConfidenceChildrenInPath,
    setFolderConfidence,
    getFoldersAtLevel,
  };
};
