// CategoriesPage.tsx
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { Category, Folder } from "../types";
import { CategoryTable } from "../components/CategoryTable";
import { CategoryDetails } from "../components/CategoryDetails";
import { fetchCategories } from "../api";

export const CategoriesPage: React.FC = () => {
  const [data, setData] = useState<any[]>([]);
  const [selectedItem, setSelectedItem] = useState<Category | Folder | null>(null);

  
  useEffect(() => {
    const fetchData = async () => {
      // Fetch data from FastAPI
      setData(await fetchCategories());
    };
    fetchData();
  }, []);

  return (
    <PageContainer>
      <ContentContainer>
        <CategoryTable
          categories={data}
          onSelectItem={setSelectedItem}
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