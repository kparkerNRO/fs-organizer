// src/mockApi.ts

import {  mockCategoryData, mockFolderStructure, mockOriginalFolderStructure } from "./mockData";
import { FetchCategoriesParams, FetchCategoriesResponse } from "../api";
import { FolderV2, FolderViewResponse } from "../types/types";


export const fetchMockCategoryData = (
  _params: FetchCategoriesParams
): Promise<FetchCategoriesResponse> => {
  const mockData = {
    data: mockCategoryData,
    totalItems: mockCategoryData.length,
    totalPages: 1,
    currentPage: 1,
  };

  return new Promise((resolve) => {
    setTimeout(() => resolve(mockData), 500); // Simulate API delay
  });
};

export const fetchMockFolderStructure = (): Promise<FolderV2> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve(mockFolderStructure), 500); // Simulate API delay
  });
};

export const fetchMockFolderStructureComparison = (): Promise<FolderViewResponse> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve({
      original: mockOriginalFolderStructure,
      new: mockFolderStructure
    }), 500); // Simulate API delay
  });
};
