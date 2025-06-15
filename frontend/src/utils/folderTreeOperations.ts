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

const generateFlattenedName = (sortedPaths: FolderTreePath[]): string => {
  // Helper function to generate target name from selected folders
  // Extract the base folder name from each path
  const pathNames = sortedPaths.map((path) => {
    const pathParts = path.split("/");
    return pathParts[pathParts.length - 1];
  });

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

  const targetName = generateFlattenedName(sourcePaths);
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

const findSharedString = (
  sourcePaths: FolderTreePath[]
): string => {
  if (sourcePaths.length === 0) return "";

  const pathNames = sourcePaths.map((path) => {
    const pathParts = path.split("/");
    return pathParts[pathParts.length - 1];
  });

  // Split each name into words (by spaces, underscores, dashes, dots)
  const wordArrays = pathNames.map((name) => {
    return name.split(/[\s_\-\.]+/).filter(word => word.length > 0);
  });

  if (wordArrays.length === 0 || wordArrays.some(arr => arr.length === 0)) {
    return "";
  }

  let longestSharedSequence = "";

  // Try all possible consecutive word sequences from the first folder
  const firstArray = wordArrays[0];
  
  for (let start = 0; start < firstArray.length; start++) {
    for (let end = start; end < firstArray.length; end++) {
      const candidateSequence = firstArray.slice(start, end + 1);
      
      // Check if this sequence exists consecutively in all other arrays
      const isSharedByAll = wordArrays.slice(1).every(wordArray => {
        return containsConsecutiveSequence(wordArray, candidateSequence);
      });
      
      if (isSharedByAll && candidateSequence.length > longestSharedSequence.split(' ').length) {
        longestSharedSequence = candidateSequence.join(' ');
      }
    }
  }

  return longestSharedSequence;
};

// Helper function to check if an array contains a consecutive sequence
const containsConsecutiveSequence = (haystack: string[], needle: string[]): boolean => {
  if (needle.length === 0) return true;
  if (haystack.length < needle.length) return false;
  
  for (let i = 0; i <= haystack.length - needle.length; i++) {
    let matches = true;
    for (let j = 0; j < needle.length; j++) {
      if (haystack[i + j].toLowerCase() !== needle[j].toLowerCase()) {
        matches = false;
        break;
      }
    }
    if (matches) return true;
  }
  
  return false;
};

// Merge multiple folders into one
export const mergeFolders = (
  tree: FolderV2,
  sourcePaths: FolderTreePath[],
  targetName: string
): FolderTreeOperationResult => {
  const nameValidation = validateNodeName(targetName);
  if (!nameValidation.valid) {
    return { success: false, error: nameValidation.error };
  }

  if (sourcePaths.length < 2) {
    return { success: false, error: "At least 2 folders required for merging" };
  }

  // Find all source folders
  const sourceNodes: FolderV2[] = [];
  for (const path of sourcePaths) {
    const node = findNodeByPath(tree, path);
    if (!node || isFileNode(node)) {
      return { success: false, error: `Folder not found at path: ${path}` };
    }
    sourceNodes.push(node);
  }

  // For now, this is a placeholder - actual merge logic would be more complex
  return {
    success: true,
    newTree: cloneTreeNode(tree) as FolderV2,
    affectedPaths: sourcePaths,
  };
};
