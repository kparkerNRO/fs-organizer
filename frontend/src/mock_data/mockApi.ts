// src/mockApi.ts

import { mockFolders, mockCategoryData } from "./mockData";
import { Category, Folder } from "../types";

export const fetchMockFolders = (): Promise<Folder[]> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve(mockFolders), 500); // Simulate API delay
  });
};

export const fetchMockCategoryData = (): Promise<Category[]> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve(mockCategoryData)); // Simulate API delay
  });
};
