// src/api.ts

import { fetchMockCategoryData, fetchMockFolderStructure, fetchMockFolderStructureComparison } from "./mock_data/mockApi";
import { useMockMode } from "./mock_data/MockModeContext";
import { Category } from "./types/types";

export interface FetchCategoriesParams {
  page_size: number;
  page: number;
  sortField?: string;
  sortOrder?: string;
}

export interface FetchCategoriesResponse {
  data: Category[];
  totalItems: number;
  totalPages: number;
  currentPage: number;
}

export const fetchCategories = async (
  params: FetchCategoriesParams
): Promise<FetchCategoriesResponse> => {
  // Use mock data if in mock mode
  const isMockMode = true; // Hardcoded for now, would use useMockMode() in a component
  
  if (isMockMode) {
    return await fetchMockCategoryData(params);
  }

  const response = await fetch(
    `http://0.0.0.0:8000/groups?` +
    `page=${params.page}&` +
    `pageSize=${params.page_size}` +
    `${params.sortField ? `&sort_column=${params.sortField}` : ''}` +
    `${params.sortOrder ? `&sort_order=${params.sortOrder}` : ''}`
  );
  const data = await response.json();
  console.log(data);
  return data;
};

export interface FileNode {
  id: string;
  name: string;
  path?: string;
  fileType?: string;
  size?: string;
  categories?: string[];
  confidence?: number;
  originalPath?: string;
  children: (FolderNode | FileNode)[];
}

export interface FolderNode {
  id: string;
  name: string;
  children?: (FolderNode | FileNode)[];
  path?: string;
  confidence?: number; // Confidence level 0-100 for folder categorization
}

export interface FolderStructureComparison {
  original: FolderNode;
  new: FolderNode;
}

export const fetchFolderStructure = async (): Promise<FolderNode> => {
  try {
    // Use mock data if in mock mode
    const isMockMode = true; // Hardcoded for now, would use useMockMode() in a component
    
    if (isMockMode) {
      return await fetchMockFolderStructure();
    }
    
    const response = await fetch('http://0.0.0.0:8000/folders');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching folder structure:", error);
    // Return a simple error structure in case of failure
    return {
      id: "error",
      name: "Error loading folders"
    };
  }
};

export const fetchFolderStructureComparison = async (): Promise<FolderStructureComparison> => {
  try {
    // Use mock data if in mock mode
    const isMockMode = true; // Hardcoded for now, would use useMockMode() in a component
    
    if (isMockMode) {
      return await fetchMockFolderStructureComparison();
    }
    
    const response = await fetch('http://0.0.0.0:8000/folders/comparison');
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching folder structure comparison:", error);
    // Return a simple error structure in case of failure
    return {
      original: {
        id: "error",
        name: "Error loading original folders"
      },
      new: {
        id: "error", 
        name: "Error loading new folders"
      }
    };
  }
};