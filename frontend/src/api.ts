// src/api.ts

import { fetchMockFolders, fetchMockCategoryData } from "./mock_data/mockApi";
import { useMockMode } from "./mock_data/MockModeContext";
import { Category } from "./types/types";

export interface FetchCategoriesParams {
  limit: number;
  offset: number;
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

  // if (useMockData) {
  // return fetchMockCategoryData(params);
  // }

  const response = await fetch(`http://0.0.0.0:8000/groups`); // ?page=${params.page}&pageSize=${params.pageSize}
  const data = await response.json();
  return {
    data: data['categories'],
    totalItems: data.length,
    totalPages: 1,
    currentPage: 1
  };
};
