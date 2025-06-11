// src/pages/FolderStructurePage.tsx
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { fetchFolderStructureComparison } from "../api";
import { FolderViewResponse } from "../types/types";
import {
  Folder as FolderIcon,
} from "lucide-react";
import { ResetButton } from "../components/ResetButton";
import {
  FolderBrowserViewType,
  FolderBrowser,
} from "../components/FolderBrowser";



// Cache keys for localStorage
const CACHE_KEYS = {
  FOLDER_DATA: "fs_organizer_folderData",
  FOLDER_STATE: "fs_organizer_folderState",
};


export const FolderStructurePage: React.FC = () => {
  const [folderComparison, setFolderComparison] =
    useState<FolderViewResponse | null>(null);
  const [, setLoading] = useState(true);

  const [selectedFileId, setSelectedFileId] = useState<number | null>(null);
  const [isStateLoaded, setIsStateLoaded] = useState(false);
  const [shouldSync, setShouldSync] = useState(true);

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
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <SyncToggleContainer>
              <SyncToggleLabel>Sync scroll:</SyncToggleLabel>
              <SyncToggle
                $isEnabled={shouldSync}
                onClick={() => setShouldSync(!shouldSync)}
              >
                <SyncToggleThumb $isEnabled={shouldSync} />
              </SyncToggle>
            </SyncToggleContainer>
            <InstructionsButton
              onClick={(e) => {
                e.stopPropagation();
                alert(
                  "Instructions:\n\n• Click files to highlight and auto-expand in other view\n• Click folders to expand/collapse\n• Escape to clear file selection\n• Toggle 'Sync scroll' to automatically scroll to selected files"
                );
              }}
            >
              ?
            </InstructionsButton>
          </div>
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
              shouldSync={shouldSync}
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
              shouldSync={shouldSync}
              showConfidence={true}
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

const SyncToggleContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const SyncToggleLabel = styled.span`
  font-size: 0.75rem;
  color: #6b7280;
  font-weight: 500;
`;

const SyncToggle = styled.div<{ $isEnabled: boolean }>`
  width: 32px;
  height: 18px;
  border-radius: 9px;
  background-color: ${(props) => (props.$isEnabled ? "#3b82f6" : "#d1d5db")};
  cursor: pointer;
  transition: background-color 0.2s ease;
  position: relative;
  
  &:hover {
    background-color: ${(props) => (props.$isEnabled ? "#2563eb" : "#9ca3af")};
  }
`;

const SyncToggleThumb = styled.div<{ $isEnabled: boolean }>`
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background-color: white;
  position: absolute;
  top: 2px;
  left: ${(props) => (props.$isEnabled ? "16px" : "2px")};
  transition: left 0.2s ease;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
`;

