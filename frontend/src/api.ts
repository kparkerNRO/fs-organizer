// src/api.ts

import { fetchMockCategoryData, fetchMockFolderStructure } from "./mock_data/mockApi";
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

export interface FolderNode {
  id: string;
  name: string;
  children?: FolderNode[];
  path?: string;
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