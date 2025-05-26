// src/api.ts

import { fetchMockCategoryData, fetchMockFolderStructure, fetchMockFolderStructureComparison } from "./mock_data/mockApi";
import { Folder, FolderV2, FolderViewResponse } from "./types/types";
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

export const fetchCategories = async (
  params: FetchCategoriesParams
): Promise<FetchCategoriesResponse> => {
  // Use mock data if in mock mode
  const isMockMode = false; // Hardcoded for now, would use useMockMode() in a component
  
  if (isMockMode) {
    return await fetchMockCategoryData(params);
  }

  const response = await fetch(
    `${env.apiUrl}/groups?` +
    `page=${params.page}&` +
    `pageSize=${params.page_size}` +
    `${params.sortField ? `&sort_column=${params.sortField}` : ''}` +
    `${params.sortOrder ? `&sort_order=${params.sortOrder}` : ''}`
  );
  const data = await response.json();
  console.log(data);
  return data;
};

// Types are now imported from types.ts to match backend API exactly

export const fetchFolderStructure = async (): Promise<FolderV2> => {
  try {
    // Use mock data if in mock mode
    const isMockMode = false; // Hardcoded for now, would use useMockMode() in a component
    
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
      children: []
    };
  }
};

export const fetchFolderStructureComparison = async (): Promise<FolderViewResponse> => {
  try {
    // Use mock data if in mock mode
    const isMockMode = false; // Hardcoded for now, would use useMockMode() in a component
    
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
        children: []
      },
      new: {
        name: "Error loading new folders",
        count: 0,
        confidence: 0,
        children: []
      }
    };
  }
};