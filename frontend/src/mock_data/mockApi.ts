// src/mockApi.ts

import {  mockCategoryData, mockFolderStructure, mockOriginalFolderStructure } from "./mockData";
import { FetchCategoriesParams, FetchCategoriesResponse, FolderNode, FolderStructureComparison } from "../api";


export const fetchMockCategoryData = (
  params: FetchCategoriesParams
): Promise<FetchCategoriesResponse> => {
  const mockData = {
    data: mockCategoryData,
    totalItems: mockCategoryData.length,
    totalPages: 1,
  };

  return new Promise((resolve) => {
    setTimeout(() => resolve(mockData), 500); // Simulate API delay
  });
};

export const fetchMockFolderStructure = (): Promise<FolderNode> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve(mockFolderStructure), 500); // Simulate API delay
  });
};

export const fetchMockFolderStructureComparison = (): Promise<FolderStructureComparison> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve({
      original: mockOriginalFolderStructure,
      new: mockFolderStructure
    }), 500); // Simulate API delay
  });
};
