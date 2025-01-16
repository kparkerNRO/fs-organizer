// src/api.ts

import { fetchMockFolders, fetchMockCategoryData } from "./mock_data/mockApi";
import { useMockMode } from "./mock_data/MockModeContext";
import { Category } from "./types";

export const fetchFolders = async () => {
  //   const useMockData = useMockMode();

  //   if (useMockData) {
  return fetchMockFolders();
  //   }

  //   const response = await fetch(`/api/folders`);
  //   return response.json();
};

export interface FetchCategoriesParams {
  page: number;
  pageSize: number;
}

export interface FetchCategoriesResponse {
  data: Category[];
  totalItems: number;
  totalPages: number;
}

export const fetchCategories = async (
  params: FetchCategoriesParams
): Promise<FetchCategoriesResponse> => {
  // const useMockData = useMockMode();

  // if (useMockData) {
  return fetchMockCategoryData(params);
  // }

  // const response = await fetch(`/api/categories?page=${params.page}&pageSize=${params.pageSize}`);
  // const data = await response.json();
  // return {
  //   data: data.categories,
  //   totalItems: data.totalItems,
  //   totalPages: data.totalPages,
  // };
};
