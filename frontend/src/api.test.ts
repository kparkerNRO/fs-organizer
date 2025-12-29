import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  fetchCategories,
  fetchFolderStructure,
  fetchFolderStructureComparison,
} from "./api";
import {
  mockFetchSuccess,
  mockFetchError,
  resetMocks,
  mockCategoriesResponse,
  mockFolderViewResponse,
} from "./test/mocks";

// Mock the env module
vi.mock("./config/env", () => ({
  env: {
    apiUrl: "http://localhost:8000",
  },
}));

describe("API Functions", () => {
  beforeEach(() => {
    resetMocks();
  });

  describe("fetchCategories", () => {
    it("should fetch categories successfully", async () => {
      mockFetchSuccess(mockCategoriesResponse);

      const params = {
        page_size: 10,
        page: 1,
        sortField: "name",
        sortOrder: "asc",
      };

      const result = await fetchCategories(params);

      expect(fetch).toHaveBeenCalledWith(
        "http://localhost:8000/groups?page=1&pageSize=10&sort_column=name&sort_order=asc",
      );
      expect(result).toEqual(mockCategoriesResponse);
    });

    it("should handle API errors", async () => {
      mockFetchError("Network error");

      const params = {
        page_size: 10,
        page: 1,
      };

      await expect(fetchCategories(params)).rejects.toThrow("Network error");
    });

    it("should build URL correctly without optional params", async () => {
      mockFetchSuccess(mockCategoriesResponse);

      const params = {
        page_size: 5,
        page: 2,
      };

      await fetchCategories(params);

      expect(fetch).toHaveBeenCalledWith(
        "http://localhost:8000/groups?page=2&pageSize=5",
      );
    });
  });

  describe("fetchFolderStructure", () => {
    it("should fetch folder structure successfully", async () => {
      mockFetchSuccess(mockFolderViewResponse);

      const result = await fetchFolderStructure();

      expect(fetch).toHaveBeenCalledWith("http://localhost:8000/folders");
      expect(result).toEqual(mockFolderViewResponse.new);
    });

    it("should return error structure on API failure", async () => {
      mockFetchError("Server error");

      const result = await fetchFolderStructure();

      expect(result).toEqual({
        name: "Error loading folders",
        count: 0,
        confidence: 0,
        children: [],
      });
    });
  });

  describe("fetchFolderStructureComparison", () => {
    it("should fetch folder structure comparison successfully", async () => {
      mockFetchSuccess(mockFolderViewResponse);

      const result = await fetchFolderStructureComparison();

      expect(fetch).toHaveBeenCalledWith("http://localhost:8000/folders");
      expect(result).toEqual(mockFolderViewResponse);
    });

    it("should return error structure on API failure", async () => {
      mockFetchError("Server error");

      const result = await fetchFolderStructureComparison();

      expect(result).toEqual({
        original: {
          name: "Error loading original folders",
          count: 0,
          confidence: 0,
          children: [],
        },
        new: {
          name: "Error loading new folders",
          count: 0,
          confidence: 0,
          children: [],
        },
      });
    });
  });
});
