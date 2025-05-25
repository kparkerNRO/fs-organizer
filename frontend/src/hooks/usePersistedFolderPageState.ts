import { useState, useEffect } from "react";

const STORAGE_KEY = "folderPageState";

interface FolderPageState {
  expandedOriginal: string[];
  expandedNew: string[];
  selectedFileId: string | null;
  highlightedPaths: {
    original: string[];
    new: string[];
  };
}

export const usePersistedFolderPageState = () => {
  // Get initial state from localStorage
  const getInitialState = (): FolderPageState => {
    try {
      const savedState = localStorage.getItem(STORAGE_KEY);
      if (savedState) {
        return JSON.parse(savedState);
      }
    } catch (error) {
      console.error("Error loading folder page state from localStorage:", error);
    }
    
    // Default state
    return {
      expandedOriginal: ["root"],
      expandedNew: ["root"],
      selectedFileId: null,
      highlightedPaths: {
        original: [],
        new: []
      }
    };
  };

  const initialState = getInitialState();

  // State hooks
  const [expandedFoldersOriginal, setExpandedFoldersOriginal] = useState<Set<string>>(
    new Set(initialState.expandedOriginal)
  );
  const [expandedFoldersNew, setExpandedFoldersNew] = useState<Set<string>>(
    new Set(initialState.expandedNew)
  );
  const [selectedFileId, setSelectedFileId] = useState<string | null>(
    initialState.selectedFileId
  );
  const [highlightedPaths, setHighlightedPaths] = useState<{original: string[], new: string[]}>(
    initialState.highlightedPaths || {original: [], new: []}
  );

  // Save to localStorage whenever any state changes
  useEffect(() => {
    try {
      const stateToSave: FolderPageState = {
        expandedOriginal: Array.from(expandedFoldersOriginal),
        expandedNew: Array.from(expandedFoldersNew),
        selectedFileId,
        highlightedPaths
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(stateToSave));
    } catch (error) {
      console.error("Error saving folder page state to localStorage:", error);
    }
  }, [expandedFoldersOriginal, expandedFoldersNew, selectedFileId, highlightedPaths]);

  // Function to reset to initial state
  const resetPageState = () => {
    setExpandedFoldersOriginal(new Set(["root"]));
    setExpandedFoldersNew(new Set(["root"]));
    setSelectedFileId(null);
    setHighlightedPaths({original: [], new: []});
  };

  return {
    expandedFoldersOriginal,
    setExpandedFoldersOriginal,
    expandedFoldersNew,
    setExpandedFoldersNew,
    selectedFileId,
    setSelectedFileId,
    highlightedPaths,
    setHighlightedPaths,
    resetPageState,
  };
};