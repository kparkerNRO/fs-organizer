import React, { useState, useEffect, useRef } from "react";
import styled from "styled-components";
import { FolderV2, File } from "../types/types";
import { ChevronDown, ChevronRight, FileIcon, FolderIcon } from "lucide-react";
import { ContextMenu } from "./ContextMenu";
import { useFolderTree } from "../hooks/useFolderTree";
import {
  isFileNode,
  buildNodePath,
  canFlattenFolders,
  getFilePathInTree,
  canInvertFolder,
  findNodeByPath,
  setConfidenceToMax,
} from "../utils/folderTreeOperations";
import { TREE_TYPE } from "../types/enums";

interface FolderBrowserProps {
  folderTree: FolderV2 | null;
  onSelectItem: (file_id: number | null) => void;
  onSelectFolder?: (folder_path: string | null) => void;
  externalSelectedFile: number | null;
  shouldSync?: boolean;
  showConfidence?: boolean;
  treeType: TREE_TYPE;
}

class FolderBrowserErrorBoundary extends React.Component<
  React.PropsWithChildren<Record<string, never>>,
  { hasError: boolean }
> {
  constructor(props: React.PropsWithChildren<Record<string, never>>) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("FolderBrowser error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: "20px", textAlign: "center", color: "#ef4444" }}>
          <h3>Something went wrong with the folder browser.</h3>
          <button
            onClick={() => this.setState({ hasError: false })}
            style={{ padding: "8px 16px", marginTop: "10px" }}
          >
            Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export const FolderBrowser: React.FC<FolderBrowserProps> = ({
  folderTree: propFolderTree,
  onSelectItem,
  onSelectFolder,
  externalSelectedFile,
  shouldSync = true,
  showConfidence = false,
}) => {
  const folderTreeHook = useFolderTree();

  // Get the active tree from the hook or use the prop
  const folderTree = folderTreeHook.modifiedTree || propFolderTree;

  // Only UI-specific state remains
  const [creatingFolder, setCreatingFolder] = useState<{
    parentPath: string;
    folderName: string;
  } | null>(null);

  // Initialize tree data when propFolderTree changes
  useEffect(() => {
    if (propFolderTree) {
      folderTreeHook.setTreeData(propFolderTree);
      // Ensure root folder is expanded
      if (propFolderTree.name) {
        folderTreeHook.expandFolder(propFolderTree.name);
      }
    }
  }, [propFolderTree, folderTreeHook]);

  // Context Menu
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    item: FolderV2 | File;
    itemPath: string;
  } | null>(null);

  const [renamingItem, setRenamingItem] = useState<{
    item: FolderV2 | File;
    itemPath: string;
    newName: string;
  } | null>(null);

  const [dragState, setDragState] = useState<{
    isDragging: boolean;
    draggedItem: { item: FolderV2 | File; itemPath: string } | null;
    dropTarget: string | null;
  }>({
    isDragging: false,
    draggedItem: null,
    dropTarget: null,
  });

  // Synchronized scrolling
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const shouldScrollToSelection = useRef<boolean>(false);
  const scrollToSelectedFile = React.useCallback((fileId: number) => {
    if (!shouldSync || !scrollContainerRef.current) return;

    setTimeout(() => {
      const fileElement = scrollContainerRef.current?.querySelector(
        `[data-file-id="${fileId}"]`,
      ) as HTMLElement;

      if (fileElement && scrollContainerRef.current) {
        const container = scrollContainerRef.current;
        const containerRect = container.getBoundingClientRect();
        const elementRect = fileElement.getBoundingClientRect();

        // Calculate if element is not fully visible
        const isAboveView = elementRect.top < containerRect.top;
        const isBelowView = elementRect.bottom > containerRect.bottom;

        if (isAboveView || isBelowView) {
          // Scroll to center the element
          const scrollTop =
            fileElement.offsetTop -
            container.offsetTop -
            container.clientHeight / 2 +
            fileElement.clientHeight / 2;

          container.scrollTo({
            top: scrollTop,
            behavior: "smooth",
          });
        }
      }
    }, 100); // Small delay to allow DOM updates
  }, [shouldSync]);

  // Keep selectedFileId in sync with externalSelectedFile
  useEffect(() => {
    if (externalSelectedFile !== folderTreeHook.selectedFileId) {
      folderTreeHook.selectFile(externalSelectedFile);
      shouldScrollToSelection.current = true;

      // Expand all parent folders to the selected item
      if (externalSelectedFile && folderTree) {
        folderTreeHook.expandToFile(externalSelectedFile);
        scrollToSelectedFile(externalSelectedFile);
      }
    }
  }, [externalSelectedFile, folderTree, shouldSync, folderTreeHook, scrollToSelectedFile]);

  // Keep externalSelectedFile in sync if selectedFileId changes internally
  useEffect(() => {
    if (folderTreeHook.selectedFileId !== externalSelectedFile) {
      onSelectItem(folderTreeHook.selectedFileId);

      // Expand all parent folders to the selected item
      if (folderTreeHook.selectedFileId && folderTree) {
        folderTreeHook.expandToFile(folderTreeHook.selectedFileId);
        // Only scroll if this change originated from external selection
        if (shouldScrollToSelection.current) {
          scrollToSelectedFile(folderTreeHook.selectedFileId);
          shouldScrollToSelection.current = false;
        }
      }
    }
  }, [folderTreeHook.selectedFileId, externalSelectedFile, onSelectItem, folderTree, folderTreeHook, scrollToSelectedFile]);

  // Handle escape key to clear selection
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (renamingItem) {
          setRenamingItem(null);
          return;
        }
        // Clear all selections
        folderTreeHook.clearSelection();
        onSelectItem(null);
        if (onSelectFolder) {
          onSelectFolder(null);
        }
        setContextMenu(null);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onSelectItem, onSelectFolder, renamingItem, folderTreeHook]);

  const handleChevronClick = (event: React.MouseEvent, folderPath: string) => {
    event.stopPropagation(); // Prevent the folder row click from triggering
    folderTreeHook.toggleFolder(folderPath);
  };

  const handleItemClick = (
    item: FolderV2 | File,
    itemPath: string,
    e: React.MouseEvent,
  ) => {
    const isFile = isFileNode(item);

    if (isFile) {
      if (!(e.ctrlKey || e.metaKey)) {
        shouldScrollToSelection.current = true; // Allow scrolling for user-initiated selection
        folderTreeHook.selectFile(item.id);
      }
    } else {
      if (e.ctrlKey || e.metaKey) {
        // Multi-select folders
        const currentSelection = folderTreeHook.selectedFolderPaths;
        const isAlreadySelected = currentSelection.includes(itemPath);
        const newSelection = isAlreadySelected
          ? currentSelection.filter((f) => f !== itemPath)
          : [...currentSelection, itemPath];

        folderTreeHook.selectMultipleFolders(newSelection);
        if (onSelectFolder) {
          onSelectFolder(itemPath);
        }
      } else if (e.shiftKey) {
        // Range selection
        const lastSelected =
          folderTreeHook.selectedFolderPaths[
            folderTreeHook.selectedFolderPaths.length - 1
          ];

        if (lastSelected) {
          folderTreeHook.selectFolderRange(lastSelected, itemPath);
        } else {
          folderTreeHook.selectFolder(itemPath);
        }

        if (onSelectFolder) {
          onSelectFolder(itemPath);
        }
      } else {
        // Single selection
        folderTreeHook.selectFolder(itemPath);
        if (onSelectFolder) {
          onSelectFolder(itemPath);
        }
      }
    }
  };

  const handleItemRightClick = (
    item: FolderV2 | File,
    itemPath: string,
    e: React.MouseEvent,
  ) => {
    e.preventDefault();

    const isFile = isFileNode(item);

    // Select the item if it's not already selected
    if (isFile) {
      if (folderTreeHook.selectedFileId !== item.id) {
        folderTreeHook.selectFile(item.id);
        if (onSelectFolder) {
          onSelectFolder(null);
        }
      }
    } else {
      if (!folderTreeHook.selectedFolderPaths.includes(itemPath)) {
        folderTreeHook.selectFolder(itemPath);
        if (onSelectFolder) {
          onSelectFolder(itemPath);
        }
        onSelectItem(null);
      }
    }

    setContextMenu({
      x: e.clientX,
      y: e.clientY,
      item,
      itemPath,
    });
  };

  const closeContextMenu = () => {
    setContextMenu(null);
  };

  const startRenaming = React.useCallback(
    (item: FolderV2 | File, itemPath: string) => {
      setRenamingItem({
        item,
        itemPath,
        newName: item.name,
      });
      closeContextMenu();
    },
    [],
  );

  const handleRenameSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!renamingItem || !renamingItem.newName.trim()) {
      setRenamingItem(null);
      return;
    }

    const result = await folderTreeHook.renameItem(
      renamingItem.itemPath,
      renamingItem.newName.trim(),
    );

    if (result?.success) {
      console.log(`Successfully renamed to "${renamingItem.newName.trim()}"`);
    } else if (result) {
      console.error("Rename failed:", result.error);
      // You could show a toast/notification here
    }

    setRenamingItem(null);
  };

  const handleRenameKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleRenameSubmit(e as React.FormEvent);
    } else if (e.key === "Escape") {
      setRenamingItem(null);
    }
  };

  const startCreatingFolder = React.useCallback((parentPath: string) => {
    setCreatingFolder({
      parentPath,
      folderName: "New Folder",
    });

    // Ensure the parent folder is expanded so the new folder input is visible
    setTreeState((prev) => ({
      ...prev,
      expandedFolders: new Set([...prev.expandedFolders, parentPath]),
    }));

    closeContextMenu();
  }, []);

  const handleCreateFolderSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!creatingFolder || !creatingFolder.folderName.trim()) {
      setCreatingFolder(null);
      return;
    }

    const result = await folderTreeHook.createFolder(
      creatingFolder.parentPath,
      creatingFolder.folderName.trim(),
    );

    if (result?.success) {
      console.log(
        `Successfully created folder "${creatingFolder.folderName.trim()}"`,
      );
    } else if (result) {
      console.error("Create folder failed:", result.error);
      // You could show a toast/notification here
    }

    setCreatingFolder(null);
  };

  const handleCreateFolderKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleCreateFolderSubmit(e as React.FormEvent);
    } else if (e.key === "Escape") {
      setCreatingFolder(null);
    }
  };

  const getContextMenuItems = React.useCallback(
    (item: FolderV2 | File, itemPath: string) => {
      const isFile = isFileNode(item);
      const menuItems = [];

      if (isFile) {
        menuItems.push(
          {
            text: "Rename file",
            onClick: () => {
              startRenaming(item, itemPath);
            },
          },
          {
            text: "View File Details",
            onClick: () => {
              console.log("View file details:", item);
              closeContextMenu();
            },
          },
          {
            text: "Copy File Path",
            onClick: () => {
              navigator.clipboard.writeText(itemPath);
              closeContextMenu();
            },
          },
        );
      } else {
        // Get the current active tree to ensure we have the most up-to-date reference
        const currentTree =
          folderTreeHook.modifiedTree ||
          folderTreeHook.originalTree ||
          propFolderTree;

        if (folderTreeHook.selectedFolderPaths.length === 1) {
          menuItems.push({
            text: "Rename folder",
            onClick: () => {
              startRenaming(item, itemPath);
            },
          });
          menuItems.push({
            text: "New folder",
            onClick: () => {
              startCreatingFolder(itemPath);
            },
          });

          if (item.confidence < 1) {
            menuItems.push({
              text: "Mark as valid",
              onClick: () => {
                folderTreeHook.setFolderConfidence(itemPath, 1);
                closeContextMenu();
              },
            });
          }

          if (currentTree) {
            const invertCheck = canInvertFolder(currentTree, itemPath);

            // Add invert with children option for single folder selection
            if (invertCheck && invertCheck.canInvert) {
              menuItems.push({
                text: "Invert with children",
                onClick: () => {
                  closeContextMenu();
                  // Use setTimeout to ensure the context menu closes before starting the async operation
                  setTimeout(async () => {
                    try {
                      const result = await folderTreeHook.invertItems(itemPath);
                      if (result.success) {
                        console.log(
                          "Successfully inverted folder with children",
                        );
                      } else {
                        console.error("Invert failed:", result.error);
                      }
                    } catch (error) {
                      console.error("Invert operation failed:", error);
                    }
                  }, 0);
                },
              });
            }
          }
        }
        if (folderTreeHook.selectedFolderPaths.length >= 2) {
          menuItems.push({
            text: "Merge Folders",
            onClick: () => {
              closeContextMenu();
              setTimeout(async () => {
                try {
                  const result = await folderTreeHook.mergeItems(
                    folderTreeHook.selectedFolderPaths,
                    "Merged Folder",
                  );
                  if (result.success) {
                    console.log("Successfully merged folders");
                  } else {
                    console.error("Merge failed:", result.error);
                  }
                } catch (error) {
                  console.error("Merge operation failed:", error);
                }
              }, 0);
            },
          });

          if (currentTree) {
            // Check if the selected folders can be flattened using the validation function
            const flattenCheck = canFlattenFolders(
              currentTree,
              folderTreeHook.selectedFolderPaths,
            );

            if (flattenCheck && flattenCheck.canFlatten) {
              menuItems.push({
                text: "Flatten folders",
                onClick: () => {
                  closeContextMenu();
                  // Use setTimeout to ensure the context menu closes before starting the async operation
                  setTimeout(async () => {
                    try {
                      const result = await folderTreeHook.flattenItems(
                        folderTreeHook.selectedFolderPaths,
                      );
                      if (result.success) {
                        console.log("Successfully flattened folders");
                      } else {
                        console.error("Flatten failed:", result.error);
                      }
                    } catch (error) {
                      console.error("Flatten operation failed:", error);
                    }
                  }, 0);
                },
              });
            }
          }
        }
      }

      // Add standard folder options
      menuItems.push({
        text: "Copy Folder Path",
        onClick: () => {
          navigator.clipboard.writeText((item as FolderV2).path || itemPath);
          closeContextMenu();
        },
      });

      menuItems.push({
        text: "Delete",
        onClick: async () => {
          // Delete all selected folder paths at once
          await folderTreeHook.deleteItems(folderTreeHook.selectedFolderPaths);
          closeContextMenu();
        },
      });

      return menuItems;
    },
    [
      folderTreeHook,
      propFolderTree,
      startCreatingFolder,
      startRenaming,
    ],
  );

  // Drag and drop handlers
  const handleDragStart = (
    e: React.DragEvent,
    item: FolderV2 | File,
    itemPath: string,
  ) => {
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", itemPath);

    setDragState({
      isDragging: true,
      draggedItem: { item, itemPath },
      dropTarget: null,
    });
  };

  const handleDragEnd = () => {
    setDragState({
      isDragging: false,
      draggedItem: null,
      dropTarget: null,
    });
  };

  const handleDragOver = (
    e: React.DragEvent,
    targetPath: string,
    targetItem: FolderV2 | File,
  ) => {
    e.preventDefault();

    // Only allow dropping on folders
    if (isFileNode(targetItem)) {
      return;
    }

    // Don't allow dropping on the dragged item itself or its descendants
    if (
      dragState.draggedItem &&
      (dragState.draggedItem.itemPath === targetPath ||
        targetPath.startsWith(dragState.draggedItem.itemPath + "/"))
    ) {
      return;
    }

    e.dataTransfer.dropEffect = "move";
    setDragState((prev) => ({ ...prev, dropTarget: targetPath }));
  };

  const handleDragLeave = (e: React.DragEvent) => {
    // Only clear drop target if we're leaving the current target element
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const x = e.clientX;
    const y = e.clientY;

    if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
      setDragState((prev) => ({ ...prev, dropTarget: null }));
    }
  };

  const handleDrop = async (
    e: React.DragEvent,
    targetPath: string,
    targetItem: FolderV2 | File,
  ) => {
    e.preventDefault();

    if (!dragState.draggedItem || isFileNode(targetItem)) {
      return;
    }

    const sourcePath = dragState.draggedItem.itemPath;

    // Don't move if source and target are the same
    if (sourcePath === targetPath) {
      return;
    }

    try {
      const result = await folderTreeHook.moveItem(sourcePath, targetPath);
      if (result.success) {
        console.log(`Successfully moved "${sourcePath}" to "${targetPath}"`);
      } else {
        console.error("Move failed:", result.error);
      }
    } catch (error) {
      console.error("Move operation failed:", error);
    }

    setDragState({
      isDragging: false,
      draggedItem: null,
      dropTarget: null,
    });
  };

  const renderNode = (
    node: FolderV2 | File,
    level: number = 0,
    expandedFolders: Set<string>,
    parentPath: string = "",
  ): React.ReactNode => {
    const nodePath = buildNodePath(parentPath, node.name);
    const isFile = isFileNode(node);
    const isExpanded = !isFile && expandedFolders.has(nodePath);
    const hasChildren =
      !isFile &&
      (node as FolderV2).children &&
      (node as FolderV2).children!.length > 0;
    const isHighlighted = folderTreeHook.isNodeSelected(node, nodePath);

    // Check if this folder is in the path to the selected file
    const isInSelectedPath =
      !isFile && folderTreeHook.selectedFileId && folderTree
        ? (() => {
            const pathToFile = getFilePathInTree(
              folderTree!,
              folderTreeHook.selectedFileId,
            );
            if (!pathToFile) return false;

            // Build the full path segments to compare against nodePath
            let currentPath = "";
            for (const segment of pathToFile) {
              currentPath = currentPath ? `${currentPath}/${segment}` : segment;
              if (currentPath === nodePath) {
                return true;
              }
            }
            return false;
          })()
        : false;

    // Drag and drop state
    const isDraggedOver = dragState.dropTarget === nodePath;
    const isDragging =
      dragState.isDragging && dragState.draggedItem?.itemPath === nodePath;

    return (
      <div key={nodePath}>
        <FolderItem
          $level={level}
          $isFile={isFile}
          $isSelected={isHighlighted}
          $isDraggedOver={isDraggedOver}
          $isInHighlightedPath={isInSelectedPath}
          data-file-id={isFile ? node.id : undefined}
          data-folder-path={!isFile ? nodePath : undefined}
          draggable={true}
          onDragStart={(e) => handleDragStart(e, node, nodePath)}
          onDragEnd={handleDragEnd}
          onDragOver={(e) => handleDragOver(e, nodePath, node)}
          onDragLeave={handleDragLeave}
          onDrop={(e) => handleDrop(e, nodePath, node)}
          onClick={(e) => handleItemClick(node, nodePath, e)}
          onContextMenu={(e) => handleItemRightClick(node, nodePath, e)}
          style={{
            opacity: isDragging ? 0.5 : 1,
            cursor: isDragging ? "grabbing" : "pointer",
          }}
        >
          {!isFile && hasChildren ? (
            <ExpandIcon
              onClick={(e) => handleChevronClick(e, nodePath)}
              role="button"
            >
              {isExpanded ? (
                <ChevronDown size={10} />
              ) : (
                <ChevronRight size={10} />
              )}
            </ExpandIcon>
          ) : (
            <ExpandIcon style={{ opacity: 0.7 }}>
              {isFile ? <FileIcon size={10} /> : <FolderIcon size={10} />}
            </ExpandIcon>
          )}
          <FolderName
            $isFile={isFile}
            $confidence={
              !isFile && showConfidence
                ? (node as FolderV2).confidence
                : undefined
            }
            style={{ display: "flex", alignItems: "center" }}
          >
            {renamingItem &&
            renamingItem.itemPath === nodePath &&
            renamingItem.item === node ? (
              <form onSubmit={handleRenameSubmit} style={{ flex: 1 }}>
                <RenameInput
                  type="text"
                  value={renamingItem.newName}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setRenamingItem({
                      ...renamingItem,
                      newName: e.target.value,
                    })
                  }
                  onKeyDown={handleRenameKeyDown}
                  onBlur={() => setRenamingItem(null)}
                  autoFocus
                />
              </form>
            ) : (
              <span
                style={{
                  flex: 1,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  display: "flex",
                  alignItems: "center",
                }}
              >
                {node.name}
                {!isFile && folderTreeHook.hasLowConfidenceChildrenInPath(nodePath) && (
                  <RedDotIndicator />
                )}
              </span>
            )}

            {showConfidence &&
              !isFile &&
              (node as FolderV2).confidence !== undefined && (
                <ConfidenceInline $confidence={(node as FolderV2).confidence}>
                  (Confidence: {Math.round((node as FolderV2).confidence * 100)}
                  %)
                </ConfidenceInline>
              )}
          </FolderName>
          {isFile && (node as File).fileType ? (
            <FileType>{(node as File).fileType}</FileType>
          ) : null}
          {isFile && (node as File).size ? (
            <FileSize>{(node as File).size}</FileSize>
          ) : null}
          {!isFile && (node as FolderV2).path && nodePath !== "root" ? (
            <FolderPath>{(node as FolderV2).path}</FolderPath>
          ) : null}
        </FolderItem>

        {!isFile && isExpanded && (
          <div>
            {hasChildren &&
              (node as FolderV2).children?.map((child) =>
                renderNode(child, level + 1, expandedFolders, nodePath),
              )}
            {/* Render new folder input if creating folder in this location */}
            {creatingFolder && creatingFolder.parentPath === nodePath && (
              <div
                style={{
                  marginLeft: `${(level + 1) * 16}px`,
                  display: "flex",
                  alignItems: "center",
                  height: "32px",
                  paddingLeft: "0.5rem",
                }}
              >
                <FolderIcon size={16} style={{ marginRight: "0.5rem" }} />
                <form onSubmit={handleCreateFolderSubmit} style={{ flex: 1 }}>
                  <RenameInput
                    type="text"
                    value={creatingFolder.folderName}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setCreatingFolder({
                        ...creatingFolder,
                        folderName: e.target.value,
                      })
                    }
                    onKeyDown={handleCreateFolderKeyDown}
                    onBlur={() => setCreatingFolder(null)}
                    autoFocus
                  />
                </form>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  if (!propFolderTree) {
    return <ErrorMessage>Failed to load folder structure</ErrorMessage>;
  }

  return (
    <FolderBrowserErrorBoundary>
      <ContentContainer>
        {folderTree ? (
          <>
            <FolderTree ref={scrollContainerRef}>
              <div
                style={{
                  display: "flex",
                  gap: "0.75rem",
                  height: "100%",
                  width: "100%",
                }}
              >
                <div style={{ flex: 1 }}>
                  {folderTree &&
                    renderNode(
                      folderTree,
                      0,
                      folderTreeHook.expandedFolders,
                      "",
                    )}
                </div>
              </div>
            </FolderTree>
          </>
        ) : (
          <ErrorMessage>Failed to load folder structure</ErrorMessage>
        )}
        {contextMenu && (
          <ContextMenu
            x={contextMenu.x}
            y={contextMenu.y}
            onClose={closeContextMenu}
            menu_items={getContextMenuItems(
              contextMenu.item,
              contextMenu.itemPath,
            )}
          />
        )}
      </ContentContainer>
    </FolderBrowserErrorBoundary>
  );
};

const ContentContainer = styled.div`
  width: 100%;
  background-color: white;
  border-radius: 0.25rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  padding: 0.25rem;
  box-sizing: border-box;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0; /* Critical for proper flexbox behavior with scrolling */
`;

const FolderTree = styled.div`
  margin-top: 0.125rem;
  flex: 1;
  overflow-y: auto;
  min-height: 0; /* Critical for proper flexbox behavior with scrolling */
  width: 100%;

  /* Modern scrollbar styling */
  &::-webkit-scrollbar {
    width: 4px;
  }

  &::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 2px;
  }

  &::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 2px;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
  }
`;

const FolderItem = styled.div<{
  $level: number;
  $isFile?: boolean;
  $isSelected?: boolean;
  $isDraggedOver?: boolean;
  $isInHighlightedPath?: boolean;
}>`
  display: flex;
  align-items: center;
  width: 100%;
  box-sizing: border-box;
  padding: 0.1875rem 0.3rem;
  padding-left: ${(props) => props.$level * 0.625 + 0.3}rem;
  cursor: pointer;
  border-radius: 0.125rem;
  transition: all 0.15s ease;
  margin: 0.25px 0;
  font-size: 0.8125rem;
  line-height: 1.3;
  color: ${(props) => (props.$isFile ? "#4b5563" : "#000")};
  background-color: ${(props) => {
    if (props.$isDraggedOver) return "#e2f0fd";
    if (props.$isSelected) return "#dbeafe";
    if (props.$isInHighlightedPath) return "#f0f9ff"; // Very subtle blue highlight for path
    return "transparent";
  }};
  border: 1px solid transparent; /* Always maintain border space */
  border-color: ${(props) => {
    if (props.$isDraggedOver) return "#3b82f6";
    if (props.$isSelected) return "#60a5fa";
    if (props.$isInHighlightedPath) return "#e0f2fe"; // Subtle border for path highlight
    return "transparent";
  }};
  border-style: ${(props) => (props.$isDraggedOver ? "dashed" : "solid")};
  position: relative;

  &:hover {
    background-color: ${(props) => (props.$isSelected ? "#c7d2fe" : "#e7eefa")};
  }

  &:active {
    background-color: #dce6f8;
  }
`;

const ExpandIcon = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-right: 0.1875rem;
  width: 12px;
  height: 12px;
`;

const FolderName = styled.span<{
  $isFile: boolean;
  $confidence?: number;
}>`
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
  background-color: ${(props) => {
    if (props.$isFile) return "transparent";
    if (props.$confidence !== undefined) {
      // Calculate confidence background color
      // const confidence = Math.max(0, Math.min(100, props.$confidence));
      const normalizedConfidence = props.$confidence;
      const red = Math.round(220 + (255 - 220) * normalizedConfidence);
      const green = Math.round(38 + (255 - 38) * normalizedConfidence);
      const blue = Math.round(38 + (255 - 38) * normalizedConfidence);
      return `rgb(${red}, ${green}, ${blue})`;
    }
    return "transparent";
  }};
  padding: 2px 6px;
  border-radius: 3px;
  margin-right: 4px;
`;

const FolderPath = styled.span`
  margin-left: 0.5rem;
  color: #6b7280;
  font-size: 0.75rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 180px;
`;

const FileType = styled.span`
  margin-left: 0.5rem;
  color: #6b7280;
  font-size: 0.75rem;
  background-color: #f3f4f6;
  padding: 0.1rem 0.4rem;
  border-radius: 0.25rem;
`;

const FileSize = styled.span`
  margin-left: 0.5rem;
  color: #6b7280;
  font-size: 0.75rem;
`;

const ConfidenceInline = styled.span<{ $confidence: number }>`
  font-size: 0.7rem;
  font-weight: 400;
  color: rgb(51, 55, 61); /* Grey text */
  margin-left: auto; /* Push to the right */
  padding-left: 0.5rem;
`;

const ErrorMessage = styled.div`
  text-align: center;
  padding: 2rem;
  color: #ef4444;
`;

const RenameInput = styled.input`
  width: 100%;
  background: white;
  border: 2px solid #3b82f6;
  border-radius: 3px;
  padding: 2px 4px;
  font-size: 0.8125rem;
  font-weight: 500;
  outline: none;

  &:focus {
    border-color: #1d4ed8;
  }
`;

const RedDotIndicator = styled.div`
  width: 6px;
  height: 6px;
  background-color: rgb(220, 38, 38); /* Match low-certainty folder color */
  border-radius: 50%;
  margin-left: 4px;
  flex-shrink: 0;
`;
