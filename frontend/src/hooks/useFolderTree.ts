import { useState, useCallback, useMemo } from "react";
import { FolderV2, File, FolderViewResponse } from "../types/types";
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
  isFileNode,
  flattenFolders,
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
  setTreeData: (
    folderViewResponse: FolderViewResponse | null,
    viewType: "ORIGINAL" | "NEW"
  ) => void;
  resetToOriginal: () => void;

  // Node operations
  renameItem: (
    targetPath: FolderTreePath,
    newName: string
  ) => Promise<FolderTreeOperationResult>;
  moveItem: (
    sourcePath: FolderTreePath,
    targetPath: FolderTreePath
  ) => Promise<FolderTreeOperationResult>;
  mergeItems: (
    sourcePaths: FolderTreePath[],
    targetName: string
  ) => Promise<FolderTreeOperationResult>;
  deleteItem: (
    targetPath: FolderTreePath
  ) => Promise<FolderTreeOperationResult>;
  flattenItems: (
    sourcePaths: FolderTreePath[]
  ) => Promise<FolderTreeOperationResult>;

  // Selection management
  selectFile: (fileId: number | null) => void;
  selectFolder: (folderPath: string | null) => void;
  selectMultipleFolders: (folderPaths: string[]) => void;
  clearSelection: () => void;

  // Tree navigation
  expandFolder: (folderPath: string) => void;
  collapseFolder: (folderPath: string) => void;
  toggleFolder: (folderPath: string) => void;
  expandToFile: (fileId: number) => void;

  // Utility functions
  findNode: (targetPath: FolderTreePath) => FolderTreeNode | null;
  getNodeParent: (targetPath: FolderTreePath) => {
    parent: FolderV2 | null;
    parentPath: string;
  };
  isNodeSelected: (node: FolderTreeNode, nodePath: string) => boolean;
}

export type UseFolderTreeReturn = FolderTreeState & FolderTreeActions;

// Helper function to get file path in tree
const getFilePathInTree = (
  tree: FolderV2 | File,
  fileId: number,
  path: string[] = []
): string[] | null => {
  if (isFileNode(tree) && tree.id === fileId) {
    return path;
  }
  if (!isFileNode(tree) && tree.children) {
    for (const child of tree.children) {
      const childPath = getFilePathInTree(child, fileId, [...path, tree.name]);
      if (childPath) return childPath;
    }
  }
  return null;
};

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
  const setTreeData = useCallback(
    (
      folderViewResponse: FolderViewResponse | null,
      viewType: "ORIGINAL" | "NEW"
    ) => {
      const tree =
        folderViewResponse &&
        (viewType === "NEW"
          ? folderViewResponse.new
          : folderViewResponse.original);
      setState((prev) => ({
        ...prev,
        originalTree: tree,
        modifiedTree: null,
        hasModifications: false,
        selectedFileId: null,
        selectedFolderPaths: [],
        operationHistory: [],
      }));
    },
    []
  );

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
      newName: string
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
    [activeTree]
  );

  const mergeItems = useCallback(
    async (
      sourcePaths: FolderTreePath[],
      targetName: string
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
    [activeTree]
  );

  const moveItem = useCallback(
    async (
      sourcePath: FolderTreePath,
      targetPath: FolderTreePath
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
    [activeTree]
  );

  const flattenItems = useCallback(
    async (
      sourcePaths: FolderTreePath[]
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
    [activeTree]
  );

  const deleteItem = useCallback(
    async (): Promise<FolderTreeOperationResult> => {
      // Placeholder for delete operation
      return { success: false, error: "Delete operation not implemented yet" };
    },
    []
  );

  // Selection management
  const selectFile = useCallback((fileId: number | null) => {
    setState((prev) => ({
      ...prev,
      selectedFileId: fileId,
      selectedFolderPaths: fileId ? [] : prev.selectedFolderPaths, // Clear folder selection when file is selected
    }));
  }, []);

  const selectFolder = useCallback((folderPath: string | null) => {
    setState((prev) => ({
      ...prev,
      selectedFolderPaths: folderPath ? [folderPath] : [],
      selectedFileId: folderPath ? null : prev.selectedFileId, // Clear file selection when folder is selected
    }));
  }, []);

  const selectMultipleFolders = useCallback((folderPaths: string[]) => {
    setState((prev) => ({
      ...prev,
      selectedFolderPaths: folderPaths,
      selectedFileId: folderPaths.length > 0 ? null : prev.selectedFileId,
    }));
  }, []);

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
    [activeTree]
  );

  // Utility functions
  const findNode = useCallback(
    (targetPath: FolderTreePath): FolderTreeNode | null => {
      return activeTree ? findNodeByPath(activeTree, targetPath) : null;
    },
    [activeTree]
  );

  const getNodeParent = useCallback(
    (targetPath: FolderTreePath) => {
      return activeTree
        ? findParentByPath(activeTree, targetPath)
        : { parent: null, parentPath: "" };
    },
    [activeTree]
  );

  const isNodeSelected = useCallback(
    (node: FolderTreeNode, nodePath: string): boolean => {
      if (isFileNode(node)) {
        return state.selectedFileId === node.id;
      } else {
        return state.selectedFolderPaths.includes(nodePath);
      }
    },
    [state.selectedFileId, state.selectedFolderPaths]
  );

  return {
    // State
    ...state,

    // Actions
    setTreeData,
    resetToOriginal,
    renameItem,
    moveItem,
    mergeItems,
    deleteItem,
    flattenItems,
    selectFile,
    selectFolder,
    selectMultipleFolders,
    clearSelection,
    expandFolder,
    collapseFolder,
    toggleFolder,
    expandToFile,
    findNode,
    getNodeParent,
    isNodeSelected,
  };
};
