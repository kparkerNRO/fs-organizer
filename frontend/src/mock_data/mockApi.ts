// src/mockApi.ts

import {
  mockCategoryData,
  mockFolderStructure,
  mockOriginalFolderStructure,
} from "./mockData";
import { FetchCategoriesParams, FetchCategoriesResponse } from "../api";
import { FolderV2, FolderViewResponse } from "../types/types";

export const fetchMockCategoryData = (
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _params: FetchCategoriesParams,
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

export const fetchMockFolderStructureComparison =
  (): Promise<FolderViewResponse> => {
    return new Promise((resolve) => {
      setTimeout(
        () =>
          resolve({
            original: mockOriginalFolderStructure,
            new: mockFolderStructure,
          }),
        500,
      ); // Simulate API delay
    });
  };

// Import Wizard Mock APIs
export const importFolder = (sourcePath: string): Promise<FolderV2> => {
  const mockImportedStructure: FolderV2 = {
    name: sourcePath.split("/").pop() || "Imported Folder",
    count: 156,
    confidence: 0.85,
    children: [
      {
        name: "Photos",
        count: 45,
        confidence: 0.9,
        children: [
          {
            name: "IMG_001.jpg",
            count: 1,
            confidence: 0.95,
            children: [],
          },
          {
            name: "IMG_002.jpg",
            count: 1,
            confidence: 0.95,
            children: [],
          },
        ],
      },
      {
        name: "Documents",
        count: 23,
        confidence: 0.8,
        children: [
          {
            name: "Report.pdf",
            count: 1,
            confidence: 0.9,
            children: [],
          },
        ],
      },
      {
        name: "Videos",
        count: 12,
        confidence: 0.95,
        children: [],
      },
    ],
  };

  return new Promise((resolve) => {
    setTimeout(() => resolve(mockImportedStructure), 2000);
  });
};

export const groupFolders = (
  originalStructure: FolderV2,
): Promise<FolderV2> => {
  const mockGroupedStructure: FolderV2 = {
    name: "Grouped Structure",
    count: originalStructure.count,
    confidence: 0.88,
    children: [
      {
        name: "Media Files",
        count: 57,
        confidence: 0.92,
        children: [
          {
            name: "Photos (45 items)",
            count: 45,
            confidence: 0.9,
            children: [],
          },
          {
            name: "Videos (12 items)",
            count: 12,
            confidence: 0.95,
            children: [],
          },
        ],
      },
      {
        name: "Documents",
        count: 23,
        confidence: 0.8,
        children: [],
      },
    ],
  };

  return new Promise((resolve) => {
    setTimeout(() => resolve(mockGroupedStructure), 1500);
  });
};

export const organizeFolders = (
  groupedStructure: FolderV2,
): Promise<FolderV2> => {
  const mockOrganizedStructure: FolderV2 = {
    name: "Organized Structure",
    count: groupedStructure.count,
    confidence: 0.93,
    children: [
      {
        name: "01_Media",
        count: 57,
        confidence: 0.95,
        children: [
          {
            name: "01_Photos",
            count: 45,
            confidence: 0.9,
            children: [],
          },
          {
            name: "02_Videos",
            count: 12,
            confidence: 0.95,
            children: [],
          },
        ],
      },
      {
        name: "02_Documents",
        count: 23,
        confidence: 0.88,
        children: [],
      },
    ],
  };

  return new Promise((resolve) => {
    setTimeout(() => resolve(mockOrganizedStructure), 1500);
  });
};

export const applyOrganization = (
  organizedStructure: FolderV2,
  targetPath: string,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  duplicateHandling: string,
): Promise<{ success: boolean; message: string }> => {
  return new Promise((resolve) => {
    setTimeout(
      () =>
        resolve({
          success: true,
          message: `Successfully organized ${organizedStructure.count} items to ${targetPath}`,
        }),
      3000,
    );
  });
};
