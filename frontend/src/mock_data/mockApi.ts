// src/mockApi.ts

import { mockFolders } from "./mockData";
import { Folder } from "../types";

export const fetchMockFolders = (): Promise<Folder[]> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve(mockFolders), 500); // Simulate API delay
  });
};
