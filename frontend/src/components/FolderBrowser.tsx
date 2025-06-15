import React, { useState, useEffect, useRef } from "react";
import styled from "styled-components";
import { FolderV2, File, FolderViewResponse } from "../types/types";
import { ChevronDown, ChevronRight, FileIcon, FolderIcon } from "lucide-react";
import { ContextMenu } from "./ContextMenu";
import { useFolderTree } from "../hooks/useFolderTree";
import {
  isFileNode,
  buildNodePath,
  canFlattenFolders,
} from "../utils/folderTreeOperations";

export enum FolderBrowserViewType {
  ORIGINAL = "ORIGINAL",
  NEW = "NEW",
}

// Helper function to get the path to a file
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

interface FolderBrowserProps {
  folderViewResponse: FolderViewResponse | null;
  onSelectItem: (file_id: number | null) => void;
  onSelectFolder?: (folder_path: string | null) => void;
  viewType: FolderBrowserViewType;
  externalSelectedFile: number | null;
  shouldSync?: boolean;
  showConfidence?: boolean;
}

export const FolderBrowser: React.FC<FolderBrowserProps> = ({
  folderViewResponse,
  onSelectItem,
  onSelectFolder,
  viewType,
  externalSelectedFile,
  shouldSync = true,
  showConfidence = false,
}) => {
  const folderTreeHook = useFolderTree();

  // // New tracking
  // Get the active tree from the hook
  const folderTree = folderTreeHook.modifiedTree || folderTreeHook.originalTree;

  // Initialize tree data when folderViewResponse changes
  useEffect(() => {
    folderTreeHook.setTreeData(folderViewResponse, viewType);
  }, [folderViewResponse]);
  // </ end new tracking

  // Old folder tree tracking
  // const folderTree =
  //   folderViewResponse &&
  //   (viewType === FolderBrowserViewType.NEW
  //     ? folderViewResponse.new
  //     : folderViewResponse.original);

  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    new Set()
  );
  const [selectedFileId, setSelectedFileId] = useState<number | null>(null);
  const [selectedFolderPath, setSelectedFolderPath] = useState<string | null>(
    null
  );

  const synchronizeFolders = (folderTree: FolderV2, selectedId: number) => {
    const path = getFilePathInTree(folderTree, selectedId);
    if (path) {
      setExpandedFolders((prev) => {
        const newSet = new Set(prev);
        let currentPath = "";
        for (const segment of path) {
          currentPath = currentPath ? `${currentPath}/${segment}` : segment;
          newSet.add(currentPath);
        }
        return newSet;
      });
    }
  };

  // </ end folder tree tracking>

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const shouldScrollToSelection = useRef<boolean>(false);
  const [selectedFolderPaths, setSelectedFolderPaths] = useState<string[]>([]);
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

  const scrollToSelectedFile = (fileId: number) => {
    if (!shouldSync || !scrollContainerRef.current) return;

    setTimeout(() => {
      const fileElement = scrollContainerRef.current?.querySelector(
        `[data-file-id="${fileId}"]`
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
  };

  // Keep selectedFileId in sync with externalSelectedFile and expand parent folders
  useEffect(() => {
    if (externalSelectedFile !== selectedFileId) {
      setSelectedFileId(externalSelectedFile);
      shouldScrollToSelection.current = true; // Mark that we should scroll

      // Clear folder selection when file is selected externally
      if (externalSelectedFile && selectedFolderPath) {
        setSelectedFolderPath(null);
        setSelectedFolderPaths([]);
      }

      // Expand all parent folders to the selected item
      if (externalSelectedFile && folderViewResponse && folderTree) {
        synchronizeFolders(folderTree, externalSelectedFile);
        // Scroll to the selected file if shouldSync is enabled
        scrollToSelectedFile(externalSelectedFile);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [externalSelectedFile, folderViewResponse, folderTree, shouldSync]);

  // Keep externalSelectedFile in sync if selectedFileId changes internally
  useEffect(() => {
    if (selectedFileId !== externalSelectedFile) {
      onSelectItem(selectedFileId);

      // Clear folder selection when file is selected
      if (selectedFileId && selectedFolderPath) {
        setSelectedFolderPath(null);
        setSelectedFolderPaths([]);
        if (onSelectFolder) {
          onSelectFolder(null);
        }
      }

      // Expand all parent folders to the selected item
      if (selectedFileId && folderViewResponse && folderTree) {
        synchronizeFolders(folderTree, selectedFileId);
        // Only scroll if this change originated from external selection
        if (shouldScrollToSelection.current) {
          scrollToSelectedFile(selectedFileId);
          shouldScrollToSelection.current = false; // Reset the flag
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFileId, shouldSync]);

  // Handle escape key to clear selection
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (renamingItem) {
          setRenamingItem(null);
          return;
        }
        // Clear all selections
        setSelectedFileId(null);
        setSelectedFolderPath(null);
        setSelectedFolderPaths([]);
        onSelectItem(null);
        if (onSelectFolder) {
          onSelectFolder(null);
        }
        // Close context menu if open
        setContextMenu(null);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onSelectItem, onSelectFolder, renamingItem]);

  const toggleFolder = (folderPath: string) => {
    setExpandedFolders((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(folderPath)) {
        newSet.delete(folderPath);
      } else {
        newSet.add(folderPath);
      }
      return newSet;
    });
  };

  const handleChevronClick = (event: React.MouseEvent, folderPath: string) => {
    event.stopPropagation(); // Prevent the folder row click from triggering
    toggleFolder(folderPath);
  };

  const handleItemClick = (
    item: FolderV2 | File,
    itemPath: string,
    e: React.MouseEvent
  ) => {
    const isFile = isFileNode(item);

    if (isFile) {
      if (!(e.ctrlKey || e.metaKey)) {
        shouldScrollToSelection.current = true; // Allow scrolling for user-initiated selection
        setSelectedFileId(item.id);
      }
    } else {
      // Clear file selection when folder is selected
      if (selectedFileId !== null) {
        setSelectedFileId(null);
        onSelectItem(null);
      }

      if (e.ctrlKey || e.metaKey) {
        setSelectedFolderPaths((prev) => {
          const isAlreadySelected = prev.some((f) => f === itemPath);
          // Multi-select folders
          const currentSelection = selectedFolderPaths;
          const newSelection = isAlreadySelected
            ? currentSelection.filter((f) => f !== itemPath)
            : [...currentSelection, itemPath];

          return newSelection;
        });

        if (onSelectFolder) {
          onSelectFolder(itemPath);
        }
      } else {
        // Set folder selection
        setSelectedFolderPath(itemPath);
        setSelectedFolderPaths([itemPath]);
        if (onSelectFolder) {
          onSelectFolder(itemPath);
        }
      }
    }
  };

  const handleItemRightClick = (
    item: FolderV2 | File,
    itemPath: string,
    e: React.MouseEvent
  ) => {
    e.preventDefault();

    const isFile = isFileNode(item);

    // Select the item if it's not already selected
    if (isFile) {
      if (selectedFileId !== item.id) {
        setSelectedFileId(item.id);
        // Clear folder selection
        if (selectedFolderPath) {
          setSelectedFolderPath(null);
          setSelectedFolderPaths([]);
          if (onSelectFolder) {
            onSelectFolder(null);
          }
        }
      }
    } else {
      if (!selectedFolderPaths.includes(itemPath)) {
        setSelectedFolderPath(itemPath);
        setSelectedFolderPaths([itemPath]);
        if (onSelectFolder) {
          onSelectFolder(itemPath);
        }
        // Clear file selection
        if (selectedFileId !== null) {
          setSelectedFileId(null);
          onSelectItem(null);
        }
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

  const startRenaming = (item: FolderV2 | File, itemPath: string) => {
    setRenamingItem({
      item,
      itemPath,
      newName: item.name,
    });
    closeContextMenu();
  };

  const handleRenameSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!renamingItem || !renamingItem.newName.trim()) {
      setRenamingItem(null);
      return;
    }

    const result = await folderTreeHook.renameItem(
      renamingItem.itemPath,
      renamingItem.newName.trim()
    );

    if (result.success) {
      console.log(`Successfully renamed to "${renamingItem.newName.trim()}"`);
    } else {
      console.error("Rename failed:", result.error);
      // You could show a toast/notification here
    }

    setRenamingItem(null);
  };

  const handleRenameKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleRenameSubmit(e as any);
    } else if (e.key === "Escape") {
      setRenamingItem(null);
    }
  };

  const getContextMenuItems = (item: FolderV2 | File, itemPath: string) => {
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
        }
      );
    } else {
      if (selectedFolderPaths.length === 1) {
        menuItems.push({
          text: "Rename folder",
          onClick: () => {
            startRenaming(item, itemPath);
          },
        });
      }
      if (selectedFolderPaths.length >= 2) {
        menuItems.push({
          text: "Merge Folders",
          onClick: () => {
            closeContextMenu();
            setTimeout(async () => {
              try {
                const result = await folderTreeHook.mergeItems(
                  selectedFolderPaths,
                  "Merged Folder"
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
      }

      if (selectedFolderPaths.length >= 2 && folderTree) {
        // Check if the selected folders can be flattened using the validation function
        const flattenCheck = canFlattenFolders(folderTree, selectedFolderPaths);

        if (flattenCheck.canFlatten) {
          menuItems.push({
            text: "Flatten folders",
            onClick: () => {
              closeContextMenu();
              // Use setTimeout to ensure the context menu closes before starting the async operation
              setTimeout(async () => {
                try {
                  const result =
                    await folderTreeHook.flattenItems(selectedFolderPaths);
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

    // Add standard folder options
    menuItems.push(
      {
        text: folderTreeHook.expandedFolders.has(itemPath)
          ? "Collapse Folder"
          : "Expand Folder",
        onClick: () => {
          folderTreeHook.toggleFolder(itemPath);
          closeContextMenu();
        },
      },
      {
        text: "Copy Folder Path",
        onClick: () => {
          navigator.clipboard.writeText((item as FolderV2).path || itemPath);
          closeContextMenu();
        },
      }
    );

    return menuItems;
  };

  const renderNode = (
    node: FolderV2 | File,
    level: number = 0,
    expandedFolders: Set<string>,
    parentPath: string = ""
  ): React.ReactNode => {
    const nodePath = buildNodePath(parentPath, node.name);
    const isFile = isFileNode(node);
    const isExpanded = !isFile && expandedFolders.has(nodePath);
    const hasChildren =
      !isFile &&
      (node as FolderV2).children &&
      (node as FolderV2).children!.length > 0;
    const isHighlighted =
      (isFile && selectedFileId === node.id) ||
      (!isFile && selectedFolderPaths.includes(nodePath));

    // Check if this folder is in the path to the selected file
    const isInSelectedPath =
      !isFile && selectedFileId && folderViewResponse
        ? (() => {
            const pathToFile = getFilePathInTree(folderTree!, selectedFileId);
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

    return (
      <div key={nodePath}>
        <FolderItem
          $level={level}
          $isFile={isFile}
          $isSelected={isHighlighted}
          $isInHighlightedPath={isInSelectedPath}
          data-file-id={isFile ? node.id : undefined}
          data-folder-path={!isFile ? nodePath : undefined}
          onClick={(e) => handleItemClick(node, nodePath, e)}
          onContextMenu={(e) => handleItemRightClick(node, nodePath, e)}
        >
          {!isFile && hasChildren ? (
            <ExpandIcon onClick={(e) => handleChevronClick(e, nodePath)}>
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
                }}
              >
                {node.name}
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

        {!isFile && isExpanded && hasChildren && (
          <div>
            {(node as FolderV2).children?.map((child) =>
              renderNode(child, level + 1, expandedFolders, nodePath)
            )}
          </div>
        )}
      </div>
    );
  };

  if (!folderViewResponse) {
    return <ErrorMessage>Failed to load folder structure</ErrorMessage>;
  }

  return (
    <ContentContainer>
      {folderViewResponse ? (
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
                {folderTree && renderNode(folderTree, 0, expandedFolders, "")}
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
            contextMenu.itemPath
          )}
        />
      )}
    </ContentContainer>
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
