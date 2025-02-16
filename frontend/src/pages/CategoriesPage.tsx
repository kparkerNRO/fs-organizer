// CategoriesPage.tsx
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { Category, Folder } from "../types/types";
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
  const [data, setData] = useState<any[]>([]);
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

  const handleUpdateCategories = (updatedCategories: Category[]) => {
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

  const fetchCategoryData = async ( page: number, page_size: number,) => {
    try {
      const response = await fetchCategories({
        page_size: page_size,
        page: page,
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
    <PageContainer>
      <Header>
        <Title>Categories</Title>
        <ResetButton onReset={() => resetToInitial(handleReset)} />
      </Header>

      <ContentContainer>
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
        />
        <CategoryDetails
          category={selectedItem?.category ?? undefined}
          folder={selectedItem?.folder ?? undefined}
        />
      </ContentContainer>
    </PageContainer>
  );
};

const PageContainer = styled.div`
  min-height: 100vh;
  background-color: #f3f4f6;
  padding: 2rem;
`;

const ContentContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
`;

const Title = styled.h1`
  font-size: 2rem;
  font-weight: 600;
  color: #1f2937;
`;
