import { FolderV2, File } from "../types/types";

// Types for folder tree operations
export type FolderTreeNode = FolderV2 | File;
export type FolderTreePath = string;

export interface FolderTreeOperation {
  type:
    | "rename"
    | "move"
    | "merge"
    | "flatten"
    | "delete"
    | "create"
    | "invert";
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

export const getFilePathInTree = (
  tree: FolderV2 | File,
  fileId: number,
  path: string[] = [],
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

// Helper function to get folders at the same level as the given folder path
export const getFoldersAtSameLevel = (
  tree: FolderV2,
  targetPath: string,
): string[] => {
  if (!tree) return [];

  // Get the parent path of the target
  const pathParts = targetPath.split("/");
  const parentPath =
    pathParts.length > 1 ? pathParts.slice(0, -1).join("/") : "";

  // Find the parent folder using the existing utility
  const parentFolder = parentPath ? findNodeByPath(tree, parentPath) : tree;
  if (!parentFolder || isFileNode(parentFolder)) return [];

  // Get all folder children at this level
  const sameLevelPaths: string[] = [];
  if ((parentFolder as FolderV2).children) {
    for (const child of (parentFolder as FolderV2).children!) {
      if (!isFileNode(child)) {
        const childPath = buildNodePath(parentPath, child.name);
        sameLevelPaths.push(childPath);
      }
    }
  }

  return sameLevelPaths;
};

// Find a node by path in the tree
export const findNodeByPath = (
  tree: FolderV2,
  targetPath: FolderTreePath,
  currentPath: string = "",
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
  currentPath: string = "",
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
    children: node.children ? node.children.map(cloneTreeNode) : [],
  };
};

// Check if a name is valid for files/folders
export const validateNodeName = (
  name: string,
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
  newName: string,
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

  node.name = newName.trim();

  // Set confidence to 100% if it's a folder
  if (!isFileNode(node)) {
    setConfidenceToMax(node);
  }

  // Sort the parent folder's children since renaming might affect alphabetical order
  const { parent } = findParentByPath(newTree, targetPath);
  if (parent && parent.children) {
    sortFolderChildren(parent);
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
  currentPath: string = "",
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
  descendantPath: string,
): boolean => {
  return (
    descendantPath.startsWith(ancestorPath + "/") ||
    descendantPath === ancestorPath
  );
};

// Check if the selected folders can be flattened
export const canFlattenFolders = (
  tree: FolderV2,
  sourcePaths: FolderTreePath[],
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

export const generateFlattenedName = (
  sortedPaths: FolderTreePath[],
): string => {
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
  sourcePaths: FolderTreePath[],
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
  const workingFolder = cloneTreeNode(targetFolder) as FolderV2;

  // By definition, we should be able to just move the children from the bottom to the top,
  //  and drop the top's children
  const lowestPath = sortedPaths[sortedPaths.length - 1];
  const bottomFolder = findNodeByPath(newTree, lowestPath) as FolderV2;
  if (!bottomFolder) {
    return {
      success: false,
      error: `Lowest folder not found at path: ${lowestPath}`,
    };
  }
  workingFolder.children = [...bottomFolder.children];

  //add the files of all the children
  const addChildren = (node: FolderV2) => {
    for (const child of node.children || []) {
      if (isFileNode(child)) {
        workingFolder.children.push(child);
      } else if (
        child.path &&
        sortedPaths.includes(child.path) &&
        child.path !== lowestPath
      ) {
        addChildren(child);
      }
    }
  };
  addChildren(targetFolder);

  const { parent: parentNode } = findParentByPath(newTree, topLevelPath);
  if (parentNode) {
    parentNode.children = parentNode.children.filter((p) => p !== targetFolder);
  }

  const pathParts = topLevelPath.split("/");
  pathParts[pathParts.length - 1] = targetName;
  const newPath = pathParts.join("/");

  const existingTarget = findNodeByPath(newTree, newPath) as FolderV2;
  if (existingTarget) {
    if (parentNode) {
      existingTarget.children = [
        ...existingTarget.children,
        ...workingFolder.children,
      ];
      sortFolderChildren(parentNode);
    }
  } else {
    workingFolder.name = targetName;
    if (parentNode) {
      parentNode.children.push(workingFolder);
      sortFolderChildren(parentNode);
    }
  }

  const placedFolder = findNodeByPath(newTree, newPath) as FolderV2;
  if (!placedFolder) {
    return {
      success: false,
      error: `Failed to find placed folder at path: ${newPath}`,
    };
  }
  placedFolder.name = targetName;
  // Set confidence to 100% for the target folder
  setConfidenceToMax(placedFolder);

  // Sort the target folder's children alphabetically
  sortFolderChildren(placedFolder);

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
    const words = name.split(/[\s]+/).filter((word) => word.length > 0);
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
    const appearsInAll = pathNames.slice(1).every((name) => {
      // Check if the sequence appears as complete words (word boundaries)
      const regex = new RegExp(`\\b${sequence.replace(/\s+/g, "\\s+")}\\b`);
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
      node.children.forEach((child) => {
        if (!isFileNode(child)) {
          setConfidenceToMax(child);
        }
      });
    }
  }
};

// Helper function to check if a folder has any child folders with confidence < 1
export const hasLowConfidenceChildren = (folder: FolderV2): boolean => {
  if (!folder.children) return false;

  for (const child of folder.children) {
    if (!isFileNode(child)) {
      const childFolder = child as FolderV2;
      if (childFolder.confidence < 1) {
        return true;
      }
      // Recursively check child folders
      if (hasLowConfidenceChildren(childFolder)) {
        return true;
      }
    }
  }
  return false;
};

// Helper function to sort children alphabetically (folders first, then files)
export const sortFolderChildren = (folder: FolderV2): void => {
  if (!folder.children) return;

  folder.children.sort((a, b) => {
    const aIsFile = isFileNode(a);
    const bIsFile = isFileNode(b);

    // Folders come first
    if (!aIsFile && bIsFile) return -1;
    if (aIsFile && !bIsFile) return 1;

    // Within the same type (both files or both folders), sort alphabetically by name
    return a.name.localeCompare(b.name, undefined, {
      numeric: true,
      sensitivity: "base",
    });
  });

  // Recursively sort all subfolders
  folder.children.forEach((child) => {
    if (!isFileNode(child)) {
      sortFolderChildren(child);
    }
  });
};

// Helper function to recursively merge a node into a target folder
const mergeNodeIntoFolder = (
  targetFolder: FolderV2,
  nodeToMerge: FolderTreeNode,
): void => {
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
    (child) => !isFileNode(child) && child.name === nodeToMerge.name,
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
// Move a folder or file to a different location in the tree
export const moveNode = (
  tree: FolderV2,
  sourcePath: FolderTreePath,
  targetPath: FolderTreePath,
): FolderTreeOperationResult => {
  // Clone the tree to avoid mutation
  const newTree = cloneTreeNode(tree) as FolderV2;

  // Find the source node
  const sourceNode = findNodeByPath(newTree, sourcePath);
  if (!sourceNode) {
    return {
      success: false,
      error: `Source node not found at path: ${sourcePath}`,
    };
  }

  // Find the target folder
  const targetFolder = findNodeByPath(newTree, targetPath);
  if (!targetFolder || isFileNode(targetFolder)) {
    return {
      success: false,
      error: `Target must be a folder at path: ${targetPath}`,
    };
  }

  // Check if the source is being moved to a descendant of itself (circular reference)
  if (!isFileNode(sourceNode) && isAncestorPath(sourcePath, targetPath)) {
    return {
      success: false,
      error: "Cannot move folder into its own descendant",
    };
  }

  // Find the source parent and remove the node
  const { parent: sourceParent } = findParentByPath(newTree, sourcePath);
  if (!sourceParent || !sourceParent.children) {
    return { success: false, error: "Source parent not found" };
  }

  const sourceIndex = sourceParent.children.findIndex(
    (child) => child === sourceNode,
  );
  if (sourceIndex === -1) {
    return {
      success: false,
      error: "Source node not found in parent's children",
    };
  }

  sourceParent.children.splice(sourceIndex, 1);

  // Add the node to the target folder
  if (!targetFolder.children) {
    targetFolder.children = [];
  }

  // Check for name conflicts and handle them
  const existingChild = targetFolder.children.find(
    (child) => child.name === sourceNode.name,
  );
  if (existingChild) {
    if (isFileNode(sourceNode) && isFileNode(existingChild)) {
      // For files, append a number to make the name unique
      let counter = 1;
      let newName = `${sourceNode.name} (${counter})`;
      while (targetFolder.children.some((child) => child.name === newName)) {
        counter++;
        newName = `${sourceNode.name} (${counter})`;
      }
      sourceNode.name = newName;
    } else if (!isFileNode(sourceNode) && !isFileNode(existingChild)) {
      // For folders, merge them
      mergeNodeIntoFolder(existingChild as FolderV2, sourceNode);

      // Sort the merged folder's children
      sortFolderChildren(existingChild as FolderV2);

      // Also sort the source parent folder
      if (
        sourceParent &&
        sourceParent.children &&
        sourceParent.children.length > 0
      ) {
        sortFolderChildren(sourceParent);
      }

      return {
        success: true,
        newTree,
        affectedPaths: [sourcePath, targetPath],
      };
    } else {
      return {
        success: false,
        error: "Cannot move: name conflict between file and folder",
      };
    }
  }

  targetFolder.children.push(sourceNode);

  // Sort the target folder's children alphabetically
  sortFolderChildren(targetFolder);

  // Also sort the source parent folder if it's different from target
  if (
    sourceParent !== targetFolder &&
    sourceParent.children &&
    sourceParent.children.length > 0
  ) {
    sortFolderChildren(sourceParent);
  }

  return {
    success: true,
    newTree,
    affectedPaths: [sourcePath, targetPath],
  };
};

export const mergeFolders = (
  tree: FolderV2,
  sourcePaths: FolderTreePath[],
): FolderTreeOperationResult => {
  const newTree = cloneTreeNode(tree) as FolderV2;

  if (sourcePaths.length < 2) {
    return { success: false, error: "At least 2 folders required for merging" };
  }

  // Validate that all source paths exist and are folders
  for (const path of sourcePaths) {
    const node = findNodeByPath(newTree, path);
    if (!node) {
      return {
        success: false,
        error: `Folder not found at path: ${path}`,
      };
    }
    if (isFileNode(node)) {
      return {
        success: false,
        error: `Path is not a folder: ${path}`,
      };
    }
  }

  // Find the target folder where all files will be moved
  const sortedPaths = [...sourcePaths].sort((a, b) => a.length - b.length);
  const topLevelPath = sortedPaths[0];
  const targetFolder = findNodeByPath(newTree, topLevelPath) as FolderV2;

  // Get the common string for all folders
  const commonString = findSharedString(sourcePaths);

  // If we found a common string, rename the target folder to it
  if (commonString !== "") {
    // If the target folder isn't named the common string, move the rest of the name to the children
    if (targetFolder.name !== commonString) {
      const targetFolderCopy = cloneTreeNode(targetFolder) as FolderV2;
      targetFolderCopy.name = targetFolder.name
        .replace(commonString, "")
        .trim();
      targetFolder.name = commonString;
      targetFolder.children = [targetFolderCopy];
    }
  }
  // If no common string found, just use the target folder as-is

  for (const path of sourcePaths) {
    // Skip the target folder itself
    if (path === topLevelPath) continue;

    const node = findNodeByPath(newTree, path);
    if (!node) continue;

    const { parent } = findParentByPath(newTree, path);

    // remove the node from the parent
    if (parent && parent.children) {
      parent.children = parent.children.filter((child) => child !== node);
    }

    // Only modify the node name if we found a common string to remove
    if (node && typeof node.name === "string" && commonString !== "") {
      node.name = node.name.replace(commonString, "").trim();
    }

    mergeNodeIntoFolder(targetFolder, node);
  }

  // Set confidence to 100% for the target folder
  setConfidenceToMax(targetFolder);

  // Sort the target folder's children alphabetically
  sortFolderChildren(targetFolder);

  return {
    success: true,
    newTree: newTree,
    affectedPaths: sourcePaths,
  };
};

// Helper function to check if folder can be inverted with children
export const canInvertFolder = (
  tree: FolderV2,
  targetPath: FolderTreePath,
): { canInvert: boolean; reason?: string } => {
  const folder = findNodeByPath(tree, targetPath);

  if (!folder || isFileNode(folder)) {
    return { canInvert: false, reason: "Target must be a folder" };
  }

  const folderNode = folder as FolderV2;

  if (!folderNode.children || folderNode.children.length === 0) {
    return { canInvert: false, reason: "Folder has no children" };
  }

  // Check if folder has any file children
  const hasFiles = folderNode.children.some((child) => isFileNode(child));
  if (hasFiles) {
    return { canInvert: false, reason: "Folder must not contain any files" };
  }

  // Check if folder has at least one folder child
  const hasFolders = folderNode.children.some((child) => !isFileNode(child));
  if (!hasFolders) {
    return {
      canInvert: false,
      reason: "Folder must contain at least one subfolder",
    };
  }

  return { canInvert: true };
};

// Invert folder with children operation
export const invertFolder = (
  tree: FolderV2,
  targetPath: FolderTreePath,
): FolderTreeOperationResult => {
  // Validate that the folder can be inverted
  const canInvert = canInvertFolder(tree, targetPath);
  if (!canInvert.canInvert) {
    return { success: false, error: canInvert.reason! };
  }

  const newTree = cloneTreeNode(tree) as FolderV2;

  // Find the target folder and its parent
  const targetFolder = findNodeByPath(newTree, targetPath) as FolderV2;
  const { parent: targetParent } = findParentByPath(newTree, targetPath);

  if (!targetFolder || !targetParent) {
    return { success: false, error: "Target folder or parent not found" };
  }

  // Store the original folder name
  const originalFolderName = targetFolder.name;

  // Get all child folders (we already validated there are no files)
  const childFolders = targetFolder.children!.filter(
    (child) => !isFileNode(child),
  ) as FolderV2[];

  // Remove the target folder from its parent
  const targetIndex = targetParent.children!.findIndex(
    (child) => child === targetFolder,
  );
  targetParent.children!.splice(targetIndex, 1);

  // Process each child folder
  for (const childFolder of childFolders) {
    // Clone the child folder to avoid mutation issues
    const childClone = cloneTreeNode(childFolder) as FolderV2;

    // Check if a folder with the same name already exists in the parent
    const existingFolder = targetParent.children!.find(
      (child) => !isFileNode(child) && child.name === childClone.name,
    ) as FolderV2 | undefined;

    if (existingFolder) {
      // If existing folder exists, we need to merge carefully
      // Create a new folder with the original name to hold the child's original children
      const newChildFolder: FolderV2 = {
        name: originalFolderName,
        children: childClone.children || [],
        confidence: 1,
        count: (childClone.children || []).length,
      };

      // Add the new child folder directly to the existing folder
      if (!existingFolder.children) {
        existingFolder.children = [];
      }
      existingFolder.children.push(newChildFolder);

      setConfidenceToMax(existingFolder);
      sortFolderChildren(existingFolder);
    } else {
      // No existing folder, create the full inverted structure
      // Create a new folder with the original name that will contain the child's original children
      const newChildFolder: FolderV2 = {
        name: originalFolderName,
        children: childClone.children || [],
        confidence: 1,
        count: (childClone.children || []).length,
      };

      // Set the child's children to contain the new folder with original name
      childClone.children = [newChildFolder];

      // Set confidence for the new structure
      setConfidenceToMax(childClone);
      sortFolderChildren(childClone);

      // Add the child folder directly to the parent
      targetParent.children!.push(childClone);
    }
  }

  // Sort the parent folder's children
  sortFolderChildren(targetParent);

  return {
    success: true,
    newTree,
    affectedPaths: [
      targetPath,
      ...childFolders.map((child) => `${targetPath}/${child.name}`),
    ],
  };
};

export const deleteFolders = (
  tree: FolderV2,
  targetPaths: FolderTreePath[],
): FolderTreeOperationResult => {
  const newTree = cloneTreeNode(tree) as FolderV2;

  for (const path of targetPaths) {
    const { parent } = findParentByPath(newTree, path);
    if (!parent || !parent.children) {
      continue; // Skip if parent not found or has no children
    }
    const baseName = path.split("/").slice(-1)[0];
    parent.children = parent.children.filter(
      (child) => child.name !== baseName,
    );
  }

  return {
    success: true,
    newTree,
    affectedPaths: targetPaths,
  };
};

export const createFolder = (
  tree: FolderV2,
  parentPath: FolderTreePath,
  folderName: string,
): FolderTreeOperationResult => {
  const newTree = cloneTreeNode(tree) as FolderV2;

  // Find the parent folder where we want to create the new folder
  const parentFolder = parentPath
    ? (findNodeByPath(newTree, parentPath) as FolderV2)
    : newTree;

  if (!parentFolder || isFileNode(parentFolder)) {
    return {
      success: false,
      error: `Parent folder not found: ${parentPath}`,
    };
  }

  // Check if a folder with the same name already exists
  const existingChild = parentFolder.children?.find(
    (child) => child.name === folderName,
  );

  if (existingChild) {
    return {
      success: false,
      error: `A folder with the name "${folderName}" already exists`,
    };
  }

  // Create the new folder
  const newFolder: FolderV2 = {
    name: folderName,
    count: 0,
    confidence: 1.0,
    children: [],
    isSelected: false,
    isExpanded: false,
  };

  // Add the new folder to the parent's children
  if (!parentFolder.children) {
    parentFolder.children = [];
  }
  parentFolder.children.push(newFolder);

  const newFolderPath = parentPath ? `${parentPath}/${folderName}` : folderName;

  return {
    success: true,
    newTree,
    affectedPaths: [newFolderPath],
  };
};
