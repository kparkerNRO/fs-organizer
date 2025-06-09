/**
 * Utility functions for folder selection using the File System Access API
 * with fallbacks for Firefox and other browsers
 */

export interface FolderSelectionResult {
  success: boolean;
  path?: string;
  error?: string;
}

/**
 * Checks if the File System Access API is supported in the current browser
 */
export const isFileSystemAccessSupported = (): boolean => {
  return 'showDirectoryPicker' in window && typeof window.showDirectoryPicker === 'function';
};

/**
 * Creates and triggers a hidden file input for folder selection (Firefox compatible)
 */
export const selectFolderWithInput = (): Promise<FolderSelectionResult> => {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.webkitdirectory = true;
    input.multiple = true;
    input.style.display = 'none';
    
    input.onchange = (event) => {
      const target = event.target as HTMLInputElement;
      const files = target.files;
      
      if (!files || files.length === 0) {
        resolve({
          success: false,
          error: 'No folder selected'
        });
        return;
      }
      
      // Get the folder name from the first file
      const firstFile = files[0];
      const pathParts = firstFile.webkitRelativePath.split('/');
      const folderName = pathParts[0];
      
      resolve({
        success: true,
        path: folderName
      });
      
      // Clean up
      document.body.removeChild(input);
    };
    
    input.oncancel = () => {
      resolve({
        success: false,
        error: 'Folder selection was cancelled'
      });
      document.body.removeChild(input);
    };
    
    // Add to DOM and trigger click
    document.body.appendChild(input);
    input.click();
  });
};

/**
 * Opens a folder selection dialog using the best available method
 * - File System Access API for Chrome/Edge
 * - webkitdirectory input for Firefox and other browsers
 */
export const selectFolder = async (): Promise<FolderSelectionResult> => {
  try {
    if (isFileSystemAccessSupported()) {
      // Use File System Access API for supported browsers
      const directoryHandle = await window.showDirectoryPicker({
        mode: 'read'
      });

      return {
        success: true,
        path: directoryHandle.name
      };
    } else {
      // Use webkitdirectory input for Firefox and other browsers
      return await selectFolderWithInput();
    }
  } catch (error) {
    // Handle user cancellation
    if (error instanceof Error && error.name === 'AbortError') {
      return {
        success: false,
        error: 'Folder selection was cancelled'
      };
    }

    // Handle other errors
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to select folder'
    };
  }
};

/**
 * Extended version that also handles reading folder contents
 * (for future use when we want to actually process the selected folder)
 */
export const selectFolderWithContents = async (): Promise<FolderSelectionResult & { directoryHandle?: FileSystemDirectoryHandle; files?: FileList }> => {
  try {
    if (isFileSystemAccessSupported()) {
      const directoryHandle = await window.showDirectoryPicker({
        mode: 'read'
      });

      return {
        success: true,
        path: directoryHandle.name,
        directoryHandle
      };
    } else {
      // Use webkitdirectory input and return files
      return new Promise((resolve) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.webkitdirectory = true;
        input.multiple = true;
        input.style.display = 'none';
        
        input.onchange = (event) => {
          const target = event.target as HTMLInputElement;
          const files = target.files;
          
          if (!files || files.length === 0) {
            resolve({
              success: false,
              error: 'No folder selected'
            });
            return;
          }
          
          const firstFile = files[0];
          const pathParts = firstFile.webkitRelativePath.split('/');
          const folderName = pathParts[0];
          
          resolve({
            success: true,
            path: folderName,
            files
          });
          
          document.body.removeChild(input);
        };
        
        input.oncancel = () => {
          resolve({
            success: false,
            error: 'Folder selection was cancelled'
          });
          document.body.removeChild(input);
        };
        
        document.body.appendChild(input);
        input.click();
      });
    }
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      return {
        success: false,
        error: 'Folder selection was cancelled'
      };
    }

    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to select folder'
    };
  }
};

// Type declarations
declare global {
  interface Window {
    showDirectoryPicker(options?: {
      mode?: 'read' | 'readwrite';
      startIn?: FileSystemHandle | string;
    }): Promise<FileSystemDirectoryHandle>;
  }
  
  interface HTMLInputElement {
    webkitdirectory: boolean;
  }
}