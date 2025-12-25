// CategoriesPage.tsx
import React, { useState, useEffect } from "react";
import { Folder, SortConfig } from "../types/types";
import { CategoryTable } from "../components/CategoryTable";
import { CategoryDetails } from "../components/CategoryDetails";
import { fetchCategories } from "../api";
import { usePersistedCategories } from "../hooks/usePersistedCategories";
import { ResetButton } from "../components/ResetButton";
import { CategoryDetailsProps } from "../types/types";

interface PaginationState {
  currentPage: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
}

export const CategoriesPage: React.FC = () => {
  const [data] = useState<Folder[]>([]);
  const [selectedItem, setSelectedItem] = useState<CategoryDetailsProps>({
    category: null,
    folder: null,
  });

  const [pagination, setPagination] = useState<PaginationState>({
    currentPage: 1,
    pageSize: 10,
    totalItems: 0,
    totalPages: 1,
  });

  const { categories, setCategories, resetToInitial } =
    usePersistedCategories(data);

  const handleUpdateCategories = (updatedCategories: Folder[]) => {
    setCategories(updatedCategories);
  };

  const handleReset = () => {
    // Reset pagination state
    setPagination((prev) => ({
      ...prev,
      currentPage: 1,
    }));

    // Reset local selection state
    setSelectedItem({ category: null, folder: null });

    // Fetch first page of data
    fetchCategoryData(1, pagination.pageSize);
  };

  useEffect(() => {
    fetchCategoryData(pagination.currentPage, pagination.pageSize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handlePageChange = async (page: number) => {
    setPagination((prev) => ({ ...prev, currentPage: page }));
    await fetchCategoryData(page, pagination.pageSize);
  };

  const handlePageSizeChange = async (size: number) => {
    setPagination((prev) => ({
      ...prev,
      pageSize: size,
      currentPage: 1, // Reset to first page when changing page size
    }));
    await fetchCategoryData(1, size);
  };

  const handleSortChange = async (sort_config: SortConfig) => {
    await fetchCategoryData(
      pagination.currentPage,
      pagination.pageSize,
      sort_config
    );
  };

  const fetchCategoryData = async (
    page: number,
    page_size: number,
    sort_config?: SortConfig
  ) => {
    try {
      const response = await fetchCategories({
        page_size: page_size,
        page: page,
        sortField: sort_config?.field,
        sortOrder: sort_config?.direction,
      });
      setCategories(response.data);
      setPagination((prev) => ({
        ...prev,
        totalItems: response.totalItems,
        totalPages: response.totalPages,
      }));
    } catch (error) {
      console.error("Error fetching categories:", error);
    }
  };

  return (
    <div className="bg-gray-100 p-8 min-h-[calc(100vh-57px)]">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-semibold text-gray-800">Categories</h1>
        <ResetButton onReset={() => resetToInitial(handleReset)} />
      </div>

      <div className="max-w-[1200px] mx-auto flex flex-col gap-8 overflow-hidden">
        <CategoryTable
          categories={categories}
          onSelectItem={setSelectedItem}
          onUpdateCategories={handleUpdateCategories}
          currentPage={pagination.currentPage}
          totalPages={pagination.totalPages}
          pageSize={pagination.pageSize}
          totalItems={pagination.totalItems}
          onPageChange={handlePageChange}
          onPageSizeChange={handlePageSizeChange}
          onSortChange={handleSortChange}
        />
        <CategoryDetails
          category={selectedItem?.category ?? undefined}
          folder={selectedItem?.folder ?? undefined}
        />
      </div>
    </div>
  );
};
