// CategoriesPage.tsx
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { Category, Folder } from "../types";
import { CategoryTable } from "../components/CategoryTable";
import { CategoryDetails } from "../components/CategoryDetails";
import { fetchCategories } from "../api";
import { usePersistedCategories } from '../hooks/usePersistedCategories';
import { ResetButton } from '../components/ResetButton';


export const CategoriesPage: React.FC = () => {
  const [data, setData] = useState<any[]>([]);
  const [selectedItem, setSelectedItem] = useState<Category | Folder  | null>(null);
  // const [categories, setCategories] = useState<Category[]>([]);

  const {
    categories,
    setCategories,
    resetToInitial
  } = usePersistedCategories(data);

  const handleUpdateCategories = (updatedCategories: Category[]) => {
    setCategories(updatedCategories);
  };
  
  useEffect(() => {
    const fetchData = async () => {
      // Fetch data from FastAPI
      setData(await fetchCategories());
    };
    fetchData();
  }, []);

  return (
    <PageContainer>
      <Header>
        <Title>Categories</Title>
        <ResetButton onReset={resetToInitial} />
      </Header>

      <ContentContainer>
        <CategoryTable
          categories={categories}
          onSelectItem={setSelectedItem}
          onUpdateCategories={handleUpdateCategories}
        />
        <CategoryDetails item={selectedItem} />
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