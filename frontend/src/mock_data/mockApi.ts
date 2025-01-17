// src/mockApi.ts

import { mockFolders, mockCategoryData } from "./mockData";
import { Category, Folder } from "../types/types";
import { FetchCategoriesParams, FetchCategoriesResponse } from "../api";

export const fetchMockFolders = (): Promise<Folder[]> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve(mockFolders), 500); // Simulate API delay
  });
};

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
