/**
 * Utility functions for folder selection in Electron using native dialog API
 */

export interface FolderSelectionResult {
  success: boolean;
  path?: string;
  error?: string;
}

/**
 * Opens a folder selection dialog using Electron's native dialog
 * Returns the full file system path to the selected folder
 */
export const selectFolder = async (): Promise<FolderSelectionResult> => {
  return await window.electronAPI.selectFolder();
};

/**
 * Extended version for backward compatibility
 * In Electron, this behaves the same as selectFolder since we get full paths
 */
export const selectFolderWithContents =
  async (): Promise<FolderSelectionResult> => {
    return await selectFolder();
  };

// Type declarations for Electron API
declare global {
  interface Window {
    electronAPI: {
      platform: string;
      selectFolder: () => Promise<FolderSelectionResult>;
    };
  }
}
