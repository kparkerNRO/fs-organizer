// src/pages/FolderStructurePage.tsx
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { fetchFolderStructureComparison } from "../api";
import { FolderV2, File, FolderViewResponse } from "../types/types";
import {
  Folder as FolderIcon,
} from "lucide-react";
import { ResetButton } from "../components/ResetButton";
import {
  FolderBrowserViewType,
  FolderBrowser,
} from "../components/FolderBrowser";

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

// Cache keys for localStorage
const CACHE_KEYS = {
  FOLDER_DATA: "fs_organizer_folderData",
  FOLDER_STATE: "fs_organizer_folderState",
};

interface FolderState {
  selectedFileId: number | null;
  expanded: string[];
}

export const FolderStructurePage: React.FC = () => {
  const [folderComparison, setFolderComparison] =
    useState<FolderViewResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const [selectedFileId, setSelectedFileId] = useState<number | null>(null);
  const [isStateLoaded, setIsStateLoaded] = useState(false);

  // Reset function to clear cache and reload data
  const handleReset = async () => {
    // Clear cache
    localStorage.removeItem(CACHE_KEYS.FOLDER_DATA);
    localStorage.removeItem(CACHE_KEYS.FOLDER_STATE);

    // Reset state
    setSelectedFileId(null);
    // setExpandedFoldersOriginal(new Set(["root"]));
    // setExpandedFoldersNew(new Set(["root"]));

    // Reload data
    try {
      setLoading(true);
      const data = await fetchFolderStructureComparison();
      setFolderComparison(data);

      // Save fresh data to cache
      localStorage.setItem(CACHE_KEYS.FOLDER_DATA, JSON.stringify(data));
    } catch (error) {
      console.error("Error reloading folder structure:", error);
    } finally {
      setLoading(false);
    }
  };

  // Load state from cache on component mount
    useEffect(() => {
      const loadInitialState = () => {
        // Load folder state only from localStorage cache
        const cachedState = localStorage.getItem(CACHE_KEYS.FOLDER_STATE);
        let folderState: number | null = null;
  
        if (cachedState) {
          try {
            folderState = JSON.parse(cachedState);
            console.log("Restoring folder state:", folderState);
  
            if (folderState) {
              // Restore state from cache
                setSelectedFileId(folderState);
            }
          } catch (e) {
            console.warn("Failed to parse cached folder state:", e);
          }
        } 
  
        setIsStateLoaded(true);
      };
  
      const loadFolderStructure = async () => {
        try {
          setLoading(true);
  
          // Load initial state first
          loadInitialState();
  
          // Try to load from cache first
          const cachedData = localStorage.getItem(CACHE_KEYS.FOLDER_DATA);
          if (cachedData) {
            const parsed = JSON.parse(cachedData);
            setFolderComparison(parsed);
            setLoading(false);
            return;
          }
  
          // Fetch fresh data if no cache
          const data = await fetchFolderStructureComparison();
          setFolderComparison(data);
  
          // Save to cache
          localStorage.setItem(CACHE_KEYS.FOLDER_DATA, JSON.stringify(data));
        } catch (error) {
          console.error("Error loading folder structure:", error);
        } finally {
          setLoading(false);
        }
      };
  
      loadFolderStructure();
    }, []);

    // Save folder state to localStorage whenever it changes (but only after initial load)
  useEffect(() => {
    if (!isStateLoaded) return; // Don't save during initial state loading


      if(selectedFileId){
      console.log("Saving folder state:", selectedFileId);
      localStorage.setItem(CACHE_KEYS.FOLDER_STATE, JSON.stringify(selectedFileId));
      }
  }, [selectedFileId, isStateLoaded]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Escape key: clear file selection
      if (e.key === "Escape") {
        setSelectedFileId(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  return (
    <PageContainer>
      <Header>
        <Title>Folder Structure</Title>
        <ResetButton onReset={handleReset} />
      </Header>

      <ContentContainer>
        <div
          style={{
            paddingBottom: "0.5rem",
            borderBottom: "1px solid #e5e7eb",
            marginBottom: "0.5rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center" }}>
            <FolderIcon
              size={16}
              style={{ marginRight: "0.5rem", opacity: 0.7 }}
            />
            <span style={{ fontWeight: 500 }}>Folders</span>
          </div>
          <InstructionsButton
            onClick={(e) => {
              e.stopPropagation();
              alert(
                "Instructions:\n\n• Click files to highlight and auto-expand in other view\n• Click folders to expand/collapse\n• Escape to clear file selection"
              );
            }}
          >
            ?
          </InstructionsButton>
        </div>
        <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
          <div
            style={{
              flex: 1,
              minWidth: 0,
              display: "flex",
              flexDirection: "column",
            }}
          >
            <span style={{ fontWeight: 500, marginBottom: "0.5rem" }}>
              Original
            </span>
            <FolderBrowser
              folderViewResponse={folderComparison}
              onSelectItem={setSelectedFileId}
              viewType={FolderBrowserViewType.ORIGINAL}
              externalSelectedFile={selectedFileId}
            />
          </div>
          <div
            style={{
              flex: 1,
              minWidth: 0,
              display: "flex",
              flexDirection: "column",
            }}
          >
            <span style={{ fontWeight: 500, marginBottom: "0.5rem" }}>New</span>
            <FolderBrowser
              folderViewResponse={folderComparison}
              onSelectItem={setSelectedFileId}
              viewType={FolderBrowserViewType.NEW}
              externalSelectedFile={selectedFileId}
            />
          </div>
        </div>
      </ContentContainer>
    </PageContainer>
  );
};

// Styled components
const PageContainer = styled.div`
  background-color: #f3f4f6;
  height: calc(100vh - 57px); /* Account for navbar height */
  overflow: hidden;
  display: flex;
  flex-direction: column;
  width: 100vw;
  box-sizing: border-box;
  padding: 1rem;
`;

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

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  flex-shrink: 0;
`;

const Title = styled.h1`
  font-size: 2rem;
  font-weight: 600;
  color: #1f2937;
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
      const confidence = Math.max(0, Math.min(100, props.$confidence));
      const normalizedConfidence = confidence / 100;
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

const InstructionsButton = styled.button`
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background-color: #e5e7eb;
  border: none;
  font-size: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.15s ease;

  &:hover {
    background-color: #d1d5db;
  }
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
