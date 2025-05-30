import React, { useState, useEffect, useRef } from "react";
import styled from "styled-components";
import { FolderV2, FolderViewResponse } from "../types/types";
import { ChevronDown, ChevronRight, FileIcon, FolderIcon } from "lucide-react";

export enum FolderBrowserViewType {
  ORIGINAL = "ORIGINAL",
  NEW = "NEW",
}

// Helper function to determine if a node is a file (has id property) or folder
const isFileNode = (node: FolderV2 | File): node is File => {
  return "id" in node;
};

// Helper function to find a file in a tree structure
const findFileInTree = (tree: FolderV2 | File, fileId: number): File | null => {
  if (isFileNode(tree) && tree.id === fileId) {
    return tree;
  }
  if (!isFileNode(tree) && tree.children) {
    for (const child of tree.children) {
      const found = findFileInTree(child, fileId);
      if (found) return found;
    }
  }
  return null;
};

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
  viewType: FolderBrowserViewType;
  externalSelectedFile: number | null;
  shouldSync?: boolean;
  showConfidence?:boolean;
}

export const FolderBrowser: React.FC<FolderBrowserProps> = ({
  folderViewResponse,
  onSelectItem,
  viewType,
  externalSelectedFile,
  shouldSync = true,
  showConfidence = false,
}) => {
  const folderTree =
    folderViewResponse &&
    (viewType === FolderBrowserViewType.NEW
      ? folderViewResponse.new
      : folderViewResponse.original);

  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    new Set()
  );
  const [selectedFileId, setSelectedFileId] = useState<number | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

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
            (container.clientHeight / 2) + 
            (fileElement.clientHeight / 2);
          
          container.scrollTo({
            top: scrollTop,
            behavior: 'smooth'
          });
        }
      }
    }, 100); // Small delay to allow DOM updates
  };

  // Keep selectedFileId in sync with externalSelectedFile and expand parent folders
  useEffect(() => {
    if (externalSelectedFile !== selectedFileId) {
      setSelectedFileId(externalSelectedFile);

      // Expand all parent folders to the selected item
      if (externalSelectedFile && folderViewResponse && folderTree) {
        synchronizeFolders(folderTree, externalSelectedFile);
        // Scroll to the selected file if shouldSync is enabled
        scrollToSelectedFile(externalSelectedFile);
      }
    }
  }, [externalSelectedFile, folderViewResponse, folderTree, shouldSync]);

  // Keep externalSelectedFile in sync if selectedFileId changes internally
  useEffect(() => {
    if (selectedFileId !== externalSelectedFile) {
      onSelectItem(selectedFileId);

      // Expand all parent folders to the selected item
      if (selectedFileId && folderViewResponse && folderTree) {
        synchronizeFolders(folderTree, selectedFileId);
        // Scroll to the selected file if shouldSync is enabled
        scrollToSelectedFile(selectedFileId);
      }
    }
  }, [selectedFileId, shouldSync]);

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

  const handleFileClick = (fileNode: File) => {
    if (isFileNode(fileNode)) {
      setSelectedFileId(fileNode.id);
    }
  };

  const renderNode = (
    node: FolderV2 | File,
    level: number = 0,
    expandedFolders: Set<string>,
    parentPath: string = ""
  ): React.ReactNode => {
    const nodePath = parentPath ? `${parentPath}/${node.name}` : node.name;
    const isFile = isFileNode(node);
    const isExpanded = !isFile && expandedFolders.has(nodePath);
    const hasChildren =
      !isFile &&
      (node as FolderV2).children &&
      (node as FolderV2).children!.length > 0;
    const isHighlighted = isFile && selectedFileId === node.id;

    // Check if this folder is in the path to the selected file
    const isInSelectedPath =
      selectedFileId && folderViewResponse
        ? getFilePathInTree(folderTree, selectedFileId)?.some(
            (pathSegment) =>
              nodePath.endsWith(pathSegment) || nodePath === pathSegment
          )
        : false;

    return (
      <div key={nodePath}>
        <FolderItem
          $level={level}
          $isFile={isFile}
          $isSelected={isHighlighted}
          $isInHighlightedPath={isInSelectedPath}
          data-file-id={isFile ? node.id : undefined}
          onClick={() => {
            if (isFile) {
              handleFileClick(node as File);
            } else {
              if (hasChildren) toggleFolder(nodePath);
            }
          }}
        >
          {!isFile && hasChildren ? (
            <ExpandIcon>
              {isExpanded ? (
                <ChevronDown size={14} />
              ) : (
                <ChevronRight size={14} />
              )}
            </ExpandIcon>
          ) : (
            <ExpandIcon style={{ opacity: 0.7 }}>
              {isFile ? <FileIcon size={14} /> : <FolderIcon size={14} />}
            </ExpandIcon>
          )}
          <FolderName
            $isFile={isFile}
            $confidence={!isFile ? (node as FolderV2).confidence : undefined}
            style={{ display: "flex", alignItems: "center" }}
          >
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
            {showConfidence &&
              !isFile &&
              (node as FolderV2).confidence !== undefined && (
                <ConfidenceInline $confidence={(node as FolderV2).confidence}>
                  (Confidence: {Math.round((node as FolderV2).confidence*100)}%)
                </ConfidenceInline>
              )}
          </FolderName>
          {isFile && "fileType" in node && node.fileType && (
            <FileType>{node.fileType}</FileType>
          )}
          {isFile && "size" in node && node.size && (
            <FileSize>{node.size}</FileSize>
          )}
          {!isFile && "path" in node && node.path && nodePath !== "root" && (
            <FolderPath>{node.path}</FolderPath>
          )}
        </FolderItem>

        {!isFile && isExpanded && hasChildren && (
          <div>
            {(node as FolderV2).children?.map((child) =>
              renderNode(
                child,
                level + 1,
                expandedFolders,
                nodePath
              )
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
                gap: "2rem",
                height: "100%",
                width: "100%",
              }}
            >
              <div style={{ flex: 1 }}>
                {renderNode(folderTree, 0, expandedFolders, "")}
              </div>
            </div>
          </FolderTree>
        </>
      ) : (
        <ErrorMessage>Failed to load folder structure</ErrorMessage>
      )}
    </ContentContainer>
  );
};

const ContentContainer = styled.div`
  width: 100%;
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  padding: 1.5rem;
  box-sizing: border-box;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0; /* Critical for proper flexbox behavior with scrolling */
`;

const FolderTree = styled.div`
  margin-top: 0.5rem;
  flex: 1;
  overflow-y: auto;
  min-height: 0; /* Critical for proper flexbox behavior with scrolling */
  width: 100%;

  /* Modern scrollbar styling */
  &::-webkit-scrollbar {
    width: 8px;
  }

  &::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
  }

  &::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 4px;
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
  padding: 0.35rem 0.5rem;
  padding-left: ${(props) => props.$level * 0.9 + 0.6}rem;
  cursor: pointer;
  border-radius: 0.15rem;
  transition: all 0.15s ease;
  margin: 1px 0;
  font-size: 0.9rem;
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
  margin-right: 0.3rem;
  width: 16px;
  height: 16px;
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

const LoadingMessage = styled.div`
  text-align: center;
  padding: 2rem;
  color: #6b7280;
`;

const ErrorMessage = styled.div`
  text-align: center;
  padding: 2rem;
  color: #ef4444;
`;
