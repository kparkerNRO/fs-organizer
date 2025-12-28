// src/api.ts

import {
  fetchMockCategoryData,
  fetchMockFolderStructure,
  fetchMockFolderStructureComparison,
} from "./mock_data/mockApi";
import {
  Folder,
  FolderV2,
  FolderViewResponse,
  AsyncTaskResponse,
  TaskInfo,
} from "./types/types";
import { env } from "./config/env";

export interface FetchCategoriesParams {
  page_size: number;
  page: number;
  sortField?: string;
  sortOrder?: string;
}

export interface FetchCategoriesResponse {
  data: Folder[];
  totalItems: number;
  totalPages: number;
  currentPage: number;
}

const isMockMode = false;

// Pipeline API functions
export const gatherFiles = async (
  basePath: string,
  onProgress?: (progress: number) => void,
  abortSignal?: AbortSignal,
): Promise<TaskInfo> => {
  // Start the gather task
  const response = await fetch(`${env.apiUrl}/api/gather`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ base_path: basePath }),
    signal: abortSignal,
  });

  if (!response.ok) {
    throw new Error(`Failed to start gather task: ${response.statusText}`);
  }

  const taskResponse: AsyncTaskResponse = await response.json();

  // Poll for completion
  return await pollTaskToCompletion(
    taskResponse.task_id,
    onProgress,
    abortSignal,
  );
};

export const groupFolders = async (
  onProgress?: (progress: number) => void,
  abortSignal?: AbortSignal,
): Promise<TaskInfo> => {
  // Start the group task
  const response = await fetch(`${env.apiUrl}/api/group`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    signal: abortSignal,
  });

  if (!response.ok) {
    throw new Error(`Failed to start group task: ${response.statusText}`);
  }

  const taskResponse: AsyncTaskResponse = await response.json();

  // Poll for completion
  return await pollTaskToCompletion(
    taskResponse.task_id,
    onProgress,
    abortSignal,
  );
};

export const generateFolders = async (
  onProgress?: (progress: number) => void,
  abortSignal?: AbortSignal,
): Promise<TaskInfo> => {
  // Start the folders task
  const response = await fetch(`${env.apiUrl}/api/folders`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    signal: abortSignal,
  });

  if (!response.ok) {
    throw new Error(`Failed to start folders task: ${response.statusText}`);
  }

  const taskResponse: AsyncTaskResponse = await response.json();

  // Poll for completion
  return await pollTaskToCompletion(
    taskResponse.task_id,
    onProgress,
    abortSignal,
  );
};

// Stage structure GET endpoints
export const getGatherStructure = async (): Promise<FolderV2 | null> => {
  if (isMockMode) {
    return null; // Mock mode doesn't support existing structures
  }

  try {
    const response = await fetch(`${env.apiUrl}/api/gather/structure`);
    if (!response.ok) {
      if (response.status === 404) {
        return null; // No structure found
      }
      throw new Error(`Failed to get gather structure: ${response.statusText}`);
    }
    const data = await response.json();
    return data.folder_structure;
  } catch (error) {
    console.error("Error fetching gather structure:", error);
    return null;
  }
};

export const getGroupStructure = async (): Promise<FolderV2 | null> => {
  if (isMockMode) {
    return null; // Mock mode doesn't support existing structures
  }

  try {
    const response = await fetch(`${env.apiUrl}/api/group/structure`);
    if (!response.ok) {
      if (response.status === 404) {
        return null; // No structure found
      }
      throw new Error(`Failed to get group structure: ${response.statusText}`);
    }
    const data = await response.json();
    return data.folder_structure;
  } catch (error) {
    console.error("Error fetching group structure:", error);
    return null;
  }
};

export const getFoldersStructure = async (): Promise<FolderV2 | null> => {
  // TODO: Implement fetching folders structure from backend
  return null;
};

// Task polling utilities
export const getTaskStatus = async (taskId: string): Promise<TaskInfo> => {
  const response = await fetch(`${env.apiUrl}/api/tasks/${taskId}`);

  if (!response.ok) {
    throw new Error(`Failed to get task status: ${response.statusText}`);
  }

  return await response.json();
};

export const pollTaskToCompletion = async (
  taskId: string,
  onProgress?: (progress: number) => void,
  abortSignal?: AbortSignal,
  pollInterval: number = 1000,
): Promise<TaskInfo> => {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        if (abortSignal?.aborted) {
          reject(new Error("Task was aborted"));
          return;
        }

        const taskInfo = await getTaskStatus(taskId);

        if (onProgress && typeof taskInfo.progress === "number") {
          onProgress(taskInfo.progress);
        }

        if (taskInfo.status === "completed") {
          resolve(taskInfo);
          return;
        }

        if (taskInfo.status === "failed") {
          reject(new Error(taskInfo.error || "Task failed"));
          return;
        }

        // Continue polling if task is pending or running
        setTimeout(poll, pollInterval);
      } catch (error) {
        reject(error);
      }
    };

    poll();
  });
};

export const fetchCategories = async (
  params: FetchCategoriesParams,
): Promise<FetchCategoriesResponse> => {
  if (isMockMode) {
    return await fetchMockCategoryData(params);
  }

  const response = await fetch(
    `${env.apiUrl}/groups?` +
      `page=${params.page}&` +
      `pageSize=${params.page_size}` +
      `${params.sortField ? `&sort_column=${params.sortField}` : ""}` +
      `${params.sortOrder ? `&sort_order=${params.sortOrder}` : ""}`,
  );
  const data = await response.json();
  console.log(data);
  return data;
};

// Types are now imported from types.ts to match backend API exactly

export const fetchFolderStructure = async (): Promise<FolderV2> => {
  try {
    if (isMockMode) {
      return await fetchMockFolderStructure();
    }

    const response = await fetch(`${env.apiUrl}/folders`);
    const data = await response.json();
    return data.new; // Return just the new structure from the comparison
  } catch (error) {
    console.error("Error fetching folder structure:", error);
    // Return a simple error structure in case of failure
    return {
      name: "Error loading folders",
      count: 0,
      confidence: 0,
      children: [],
    };
  }
};

export const fetchFolderStructureComparison =
  async (): Promise<FolderViewResponse> => {
    try {
      if (isMockMode) {
        return await fetchMockFolderStructureComparison();
      }

      const response = await fetch(`${env.apiUrl}/folders`);
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Error fetching folder structure comparison:", error);
      // Return a simple error structure in case of failure
      return {
        original: {
          name: "Error loading original folders",
          count: 0,
          confidence: 0,
          children: [],
        },
        new: {
          name: "Error loading new folders",
          count: 0,
          confidence: 0,
          children: [],
        },
      };
    }
  };

export const saveGraph = async (folderStructure: FolderV2): Promise<void> => {
  const response = await fetch(`${env.apiUrl}/api/save-graph`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ structure: folderStructure }),
  });

  if (!response.ok) {
    throw new Error(`Failed to save graph: ${response.statusText}`);
  }
};
