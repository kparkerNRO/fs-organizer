// src/api.ts

import { fetchMockFolders, fetchMockCategoryData } from "./mock_data/mockApi";
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

  // if (useMockData) {
  // return fetchMockCategoryData(params);
  // }

  const response = await fetch(`http://0.0.0.0:8000/groups?page=${params.page}&pageSize=${params.page_size}`); // 
  const data = await response.json();
  console.log(data)
  return data
};
