import { FolderV2, File } from "../types/types";

// Types for folder tree operations
export type FolderTreeNode = FolderV2 | File;
export type FolderTreePath = string;

export interface FolderTreeOperation {
  type: "rename" | "move" | "merge" | "flatten" | "delete" | "create";
  sourcePath: FolderTreePath;
  targetPath?: FolderTreePath;
  newName?: string;
  targetNodes?: FolderTreeNode[];
}

export interface FolderTreeOperationResult {
  success: boolean;
  newTree?: FolderV2;
  error?: string;
  affectedPaths?: FolderTreePath[];
}

// Helper function to check if a node is a file
export const isFileNode = (node: FolderTreeNode): node is File => {
  return "id" in node;
};

// Helper function to build node path
export const buildNodePath = (parentPath: string, nodeName: string): string => {
  return parentPath ? `${parentPath}/${nodeName}` : nodeName;
};

// Find a node by path in the tree
export const findNodeByPath = (
  tree: FolderV2,
  targetPath: FolderTreePath,
  currentPath: string = ""
): FolderTreeNode | null => {
  const nodePath = buildNodePath(currentPath, tree.name);

  if (nodePath === targetPath) {
    return tree;
  }

  if (tree.children) {
    for (const child of tree.children) {
      if (isFileNode(child)) {
        const childPath = buildNodePath(nodePath, child.name);
        if (childPath === targetPath) {
          return child;
        }
      } else {
        const found = findNodeByPath(child, targetPath, nodePath);
        if (found) return found;
      }
    }
  }

  return null;
};

// Find parent folder of a node by path
export const findParentByPath = (
  tree: FolderV2,
  targetPath: FolderTreePath,
  currentPath: string = ""
): { parent: FolderV2 | null; parentPath: string } => {
  const nodePath = buildNodePath(currentPath, tree.name);

  if (tree.children) {
    for (const child of tree.children) {
      const childPath = buildNodePath(nodePath, child.name);
      if (childPath === targetPath) {
        return { parent: tree, parentPath: nodePath };
      }

      if (!isFileNode(child)) {
        const result = findParentByPath(child, targetPath, nodePath);
        if (result.parent) return result;
      }
    }
  }

  return { parent: null, parentPath: "" };
};

// Deep clone a tree node
export const cloneTreeNode = (node: FolderTreeNode): FolderTreeNode => {
  if (isFileNode(node)) {
    return { ...node };
  }

  return {
    ...node,
    children: node.children ? node.children.map(cloneTreeNode) : undefined,
  };
};

// Check if a name is valid for files/folders
export const validateNodeName = (
  name: string
): { valid: boolean; error?: string } => {
  if (!name.trim()) {
    return { valid: false, error: "Name cannot be empty" };
  }

  if (name.includes("/")) {
    return { valid: false, error: "Name cannot contain forward slashes" };
  }

  if (name.startsWith(".") && name.length === 1) {
    return { valid: false, error: "Name cannot be just a dot" };
  }

  return { valid: true };
};

// Rename a node in the tree
export const renameNode = (
  tree: FolderV2,
  targetPath: FolderTreePath,
  newName: string
): FolderTreeOperationResult => {
  const nameValidation = validateNodeName(newName);
  if (!nameValidation.valid) {
    return { success: false, error: nameValidation.error };
  }

  const newTree = cloneTreeNode(tree) as FolderV2;
  const node = findNodeByPath(newTree, targetPath);

  if (!node) {
    return { success: false, error: `Node not found at path: ${targetPath}` };
  }

  const oldName = node.name;
  node.name = newName.trim();
  
  // Set confidence to 100% if it's a folder
  if (!isFileNode(node)) {
    setConfidenceToMax(node);
  }

  // Calculate new path for the renamed node
  const pathParts = targetPath.split("/");
  pathParts[pathParts.length - 1] = newName.trim();
  const newPath = pathParts.join("/");

  return {
    success: true,
    newTree,
    affectedPaths: [targetPath, newPath],
  };
};

// Get all descendant paths of a folder
export const getDescendantPaths = (
  tree: FolderV2,
  targetPath: FolderTreePath,
  currentPath: string = ""
): FolderTreePath[] => {
  const nodePath = buildNodePath(currentPath, tree.name);
  const paths: FolderTreePath[] = [];

  if (nodePath === targetPath && tree.children) {
    const collectPaths = (node: FolderTreeNode, parentPath: string) => {
      const childPath = buildNodePath(parentPath, node.name);
      paths.push(childPath);

      if (!isFileNode(node) && node.children) {
        node.children.forEach((child) => collectPaths(child, childPath));
      }
    };

    tree.children.forEach((child) => collectPaths(child, nodePath));
  } else if (tree.children) {
    for (const child of tree.children) {
      if (!isFileNode(child)) {
        paths.push(...getDescendantPaths(child, targetPath, nodePath));
      }
    }
  }

  return paths;
};

// Check if path1 is ancestor of path2
export const isAncestorPath = (
  ancestorPath: string,
  descendantPath: string
): boolean => {
  return (
    descendantPath.startsWith(ancestorPath + "/") ||
    descendantPath === ancestorPath
  );
};

// Check if the selected folders can be flattened
export const canFlattenFolders = (
  tree: FolderV2,
  sourcePaths: FolderTreePath[]
): { canFlatten: boolean; reason?: string } => {
  if (sourcePaths.length < 2) {
    return {
      canFlatten: false,
      reason: "At least 2 folders required for flattening",
    };
  }

  // Validate that all source folders exist and are folders
  for (const path of sourcePaths) {
    const node = findNodeByPath(tree, path);
    if (!node || isFileNode(node)) {
      return { canFlatten: false, reason: `Folder not found at path: ${path}` };
    }
  }

  // Sort paths by length to establish hierarchy order
  const sortedPaths = [...sourcePaths].sort((a, b) => a.length - b.length);

  // Validate that folders form a single hierarchy chain
  for (let i = 0; i < sortedPaths.length - 1; i++) {
    const currentPath = sortedPaths[i];
    const nextPath = sortedPaths[i + 1];

    if (!isAncestorPath(currentPath, nextPath)) {
      return {
        canFlatten: false,
        reason: `Folders must form a single hierarchy chain: "${nextPath}" is not a descendant of "${currentPath}"`,
      };
    }

    // Check if there are any sibling folders in the selection
    for (let j = i + 2; j < sortedPaths.length; j++) {
      const laterPath = sortedPaths[j];

      if (
        isAncestorPath(currentPath, laterPath) &&
        !isAncestorPath(nextPath, laterPath)
      ) {
        return {
          canFlatten: false,
          reason: `Cannot have sibling folders in selection: "${nextPath}" and "${laterPath}" are siblings`,
        };
      }
    }
  }

  // Check if any selected folder has unselected siblings
  for (const selectedPath of sourcePaths) {
    const { parent, parentPath } = findParentByPath(tree, selectedPath);

    // Only check siblings if the parent is also selected (part of the hierarchy)
    if (parent && parent.children && sourcePaths.includes(parentPath)) {
      const siblings: string[] = [];
      for (const child of parent.children) {
        if (!isFileNode(child)) {
          const siblingPath = buildNodePath(parentPath, child.name);
          if (siblingPath !== selectedPath) {
            siblings.push(siblingPath);
          }
        }
      }

      // Check if any sibling is not in the selected paths
      for (const siblingPath of siblings) {
        if (!sourcePaths.includes(siblingPath)) {
          return {
            canFlatten: false,
            reason: `All sibling folders must be selected: "${selectedPath}" has unselected sibling "${siblingPath}"`,
          };
        }
      }
    }
  }

  return { canFlatten: true };
};

export const pathNamesToRootPath = (pathNames: string[]): string[] => {
  return pathNames.map((path) => {
    const pathParts = path.split("/");
    return pathParts[pathParts.length - 1];
  });
};

export const generateFlattenedName = (sortedPaths: FolderTreePath[]): string => {
  // Helper function to generate target name from selected folders
  // Extract the base folder name from each path
  const pathNames = pathNamesToRootPath(sortedPaths);

  const new_name = pathNames.join(" ");

  const nameValidation = validateNodeName(new_name);
  if (!nameValidation.valid) {
    return pathNames[0];
  }
  return new_name;
};

export const flattenFolders = (
  tree: FolderV2,
  sourcePaths: FolderTreePath[]
): FolderTreeOperationResult => {
  // Use the helper function to validate if folders can be flattened
  const canFlatten = canFlattenFolders(tree, sourcePaths);
  if (!canFlatten.canFlatten) {
    return { success: false, error: canFlatten.reason! };
  }

  const sortedPaths = [...sourcePaths].sort((a, b) => a.length - b.length);

  const targetName = generateFlattenedName(sortedPaths);
  const topLevelPath = sortedPaths[0];

  const newTree = cloneTreeNode(tree) as FolderV2;

  // Find the target folder where all files will be moved
  const targetFolder = findNodeByPath(newTree, topLevelPath) as FolderV2;
  if (!targetFolder) {
    return {
      success: false,
      error: `Target folder not found at path: ${topLevelPath}`,
    };
  }

  // Collect all children from nested folders that will be flattened
  const processedPaths = new Set<string>();
  for (const path of sortedPaths) {
    if (path !== topLevelPath && !processedPaths.has(path)) {
      const folder = findNodeByPath(newTree, path) as FolderV2;
      if (folder) {
        for (const child of folder.children) {
          targetFolder.children.push(child);
        }
        processedPaths.add(path);
      }
    }
  }

  // Remove the nested folder structure by updating the target folder's children
  targetFolder.children = targetFolder.children.filter((child) => {
    if (isFileNode(child)) {
      return true;
    } else {
      // Keep folders that are not in the flatten list (except the target itself)
      const childPath = buildNodePath(topLevelPath, child.name);
      return !sourcePaths.includes(childPath) || childPath === topLevelPath;
    }
  });

  targetFolder.name = targetName;
  
  // Set confidence to 100% for the target folder
  setConfidenceToMax(targetFolder);

  const pathParts = topLevelPath.split("/");
  pathParts[pathParts.length - 1] = targetName;
  const newPath = pathParts.join("/");

  return {
    success: true,
    newTree,
    affectedPaths: [
      topLevelPath,
      newPath,
      ...sortedPaths.filter((p) => p !== topLevelPath),
    ],
  };
};

/**
 * Calculate the longest string shared between all values in sourcePaths
 * @param sourcePaths
 * @returns
 */
export const findSharedString = (sourcePaths: FolderTreePath[]): string => {
  if (sourcePaths.length < 2) return "";

  const pathNames = pathNamesToRootPath(sourcePaths);

  // Helper function to find all possible contiguous word sequences in a name
  const getAllWordSequences = (name: string): string[] => {
    const words = name.split(/[\s]+/).filter(word => word.length > 0);
    const sequences: string[] = [];
    
    // Generate all contiguous subsequences
    for (let start = 0; start < words.length; start++) {
      for (let end = start + 1; end <= words.length; end++) {
        sequences.push(words.slice(start, end).join(" "));
      }
    }
    
    return sequences;
  };

  // Get all possible sequences from the first name
  const allSequencesFromFirstName = getAllWordSequences(pathNames[0]);
  const commonSequences: string[] = [];

  for (const sequence of allSequencesFromFirstName) {
    // Check if this sequence appears in all other names
    const appearsInAll = pathNames.slice(1).every(name => {
      // Check if the sequence appears as complete words (word boundaries)
      const regex = new RegExp(`\\b${sequence.replace(/\s+/g, '\\s+')}\\b`);
      return regex.test(name);
    });
    
    if (appearsInAll) {
      commonSequences.push(sequence);
    }
  }

  if (commonSequences.length > 0) {
    // Return the longest common sequence
    return commonSequences.sort((a, b) => b.length - a.length)[0];
  }

  return "";
};

// Helper function to set confidence to 100% for impacted folders
export const setConfidenceToMax = (node: FolderTreeNode): void => {
  if (!isFileNode(node)) {
    node.confidence = 1;
    if (node.children) {
      node.children.forEach(child => {
        if (!isFileNode(child)) {
          setConfidenceToMax(child);
        }
      });
    }
  }
};

// Helper function to recursively merge a node into a target folder
const mergeNodeIntoFolder = (targetFolder: FolderV2, nodeToMerge: FolderTreeNode): void => {
  if (!targetFolder.children) {
    targetFolder.children = [];
  }

  // If it's a file, just add it (files can have duplicate names)
  if (isFileNode(nodeToMerge)) {
    targetFolder.children.push(nodeToMerge);
    return;
  }

  // It's a folder - check if a folder with the same name already exists
  const existingFolder = targetFolder.children.find(
    child => !isFileNode(child) && child.name === nodeToMerge.name
  ) as FolderV2 | undefined;

  if (existingFolder) {
    // Merge the contents of nodeToMerge into the existing folder
    if (nodeToMerge.children) {
      for (const childToMerge of nodeToMerge.children) {
        mergeNodeIntoFolder(existingFolder, childToMerge);
      }
    }
  } else {
    // No conflict, just add the folder
    targetFolder.children.push(nodeToMerge);
  }
};

// Merge multiple folders into one
export const mergeFolders = (
  tree: FolderV2,
  sourcePaths: FolderTreePath[]
): FolderTreeOperationResult => {

  const newTree = cloneTreeNode(tree) as FolderV2;

  if (sourcePaths.length < 2) {
    return { success: false, error: "At least 2 folders required for merging" };
  }

  // Get the common string for all folders
  const commonString = findSharedString(sourcePaths);
  if (commonString === "") {
    return {
      success: false,
      error: `Unable to find common string to merge to`,
    };
  }

  // Find the target folder where all files will be moved
  const sortedPaths = [...sourcePaths].sort((a, b) => a.length - b.length);
  const topLevelPath = sortedPaths[0];
  const targetFolder = findNodeByPath(newTree, topLevelPath) as FolderV2;
  if (!targetFolder) {
    return {
      success: false,
      error: `Target folder not found at path: ${topLevelPath}`,
    };
  }

  // If the target folder isn't named the common string, move the rest of the name to the children
  if (targetFolder.name !== commonString) {
    const targetFolderCopy = cloneTreeNode(targetFolder) as FolderV2;
    targetFolderCopy.name = targetFolder.name.replace(commonString, "").trim();
    targetFolder.name = commonString;
    targetFolder.children = [targetFolderCopy];
  }

  for (const path of sourcePaths) {
    // Skip the target folder itself
    if (path === topLevelPath) continue;
    
    const node = findNodeByPath(newTree, path);
    if (!node) continue;

    const { parent } = findParentByPath(newTree, path);

    if (parent && parent.children) {
      parent.children = parent.children.filter((child) => child !== node);
    }

    if (node && typeof node.name === "string") {
      node.name = node.name.replace(commonString, "").trim();
    }

    mergeNodeIntoFolder(targetFolder, node);
  }

  // Set confidence to 100% for the target folder
  setConfidenceToMax(targetFolder);

  return {
    success: true,
    newTree: newTree,
    affectedPaths: sourcePaths,
  };
};
