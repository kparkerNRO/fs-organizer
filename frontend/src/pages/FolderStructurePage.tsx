// src/pages/FolderStructurePage.tsx
import React, { useState, useEffect } from "react";
import { fetchFolderStructureComparison } from "../api";
import { FolderViewResponse } from "../types/types";
import { Folder as FolderIcon } from "lucide-react";
import { ResetButton } from "../components/ResetButton";
import { FolderBrowser } from "../components/FolderBrowser";

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

    if (selectedFileId) {
      console.log("Saving folder state:", selectedFileId);
      localStorage.setItem(
        CACHE_KEYS.FOLDER_STATE,
        JSON.stringify(selectedFileId)
      );
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
    <div
      className="bg-gray-100 overflow-hidden flex flex-col w-screen box-border p-4"
      style={{ height: "calc(100vh - 57px)" }}
    >
      <div className="flex justify-between items-center mb-4 shrink-0">
        <h1 className="text-3xl font-semibold text-gray-800">
          Folder Structure
        </h1>
        <ResetButton onReset={handleReset} />
      </div>

      <div className="w-full bg-white rounded-lg shadow-sm p-6 box-border overflow-hidden flex flex-col flex-1 min-h-0">
        <div className="pb-2 border-b border-gray-200 mb-2 flex items-center justify-between">
          <div className="flex items-center">
            <FolderIcon size={16} className="mr-2 opacity-70" />
            <span className="font-medium">Folders</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500 font-medium">
                Sync scroll:
              </span>
              <div
                className={`w-8 h-[18px] rounded-[9px] cursor-pointer transition-colors duration-200 relative ${
                  shouldSync
                    ? "bg-blue-500 hover:bg-blue-600"
                    : "bg-gray-300 hover:bg-gray-400"
                }`}
                onClick={() => setShouldSync(!shouldSync)}
              >
                <div
                  className={`w-[14px] h-[14px] rounded-full bg-white absolute top-0.5 transition-all duration-200 shadow-sm ${
                    shouldSync ? "left-4" : "left-0.5"
                  }`}
                />
              </div>
            </div>
            <button
              className="w-5 h-5 rounded-full bg-gray-200 border-none text-xs flex items-center justify-center cursor-pointer transition-all duration-150 ease-in-out hover:bg-gray-300"
              onClick={(e) => {
                e.stopPropagation();
                alert(
                  "Instructions:\n\n• Click files to highlight and auto-expand in other view\n• Click folders to expand/collapse\n• Escape to clear file selection\n• Toggle 'Sync scroll' to automatically scroll to selected files"
                );
              }}
            >
              ?
            </button>
          </div>
        </div>
        <div className="flex flex-1 min-h-0">
          <div className="flex-1 min-w-0 flex flex-col">
            <span className="font-medium mb-2">Original</span>
            <FolderBrowser
              folderTree={folderComparison?.original || null}
              onSelectItem={setSelectedFileId}
              externalSelectedFile={selectedFileId}
              shouldSync={shouldSync}
            />
          </div>
          <div className="flex-1 min-w-0 flex flex-col">
            <span className="font-medium mb-2">New</span>
            <FolderBrowser
              folderTree={folderComparison?.new || null}
              onSelectItem={setSelectedFileId}
              externalSelectedFile={selectedFileId}
              shouldSync={shouldSync}
              showConfidence={true}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
