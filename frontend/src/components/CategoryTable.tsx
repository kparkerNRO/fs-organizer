import React from "react";
import styled from "styled-components";
import { Category } from "../types";
import { ChevronDown } from "lucide-react"; 

interface CategoryTableProps {
  categories: Category[];
  onSelectCategory: (cat: Category) => void;
  selectedCategory: Category | null;  
}

export const CategoryTable: React.FC<CategoryTableProps> = ({
  categories,
  onSelectCategory,
  selectedCategory,
}) => {
  return (
    <TableContainer>
      <HeaderContainer>
        <Title>Categories</Title>
        <SearchInput placeholder="Search" />
      </HeaderContainer>

      <TableGrid>
        <HeaderGrid>
          <HeaderCell>
            Title <ChevronDown size={16} />
          </HeaderCell>
          <HeaderCell>
            Classification <ChevronDown size={16} />
          </HeaderCell>
          <HeaderCell>
            Count <ChevronDown size={16} />
          </HeaderCell>
          <HeaderCell>
            Possible classifications <ChevronDown size={16} />
          </HeaderCell>
          <HeaderCell>
            Confidence <ChevronDown size={16} />
          </HeaderCell>
        </HeaderGrid>

        <RowsContainer>
          {categories.map((cat, index) => (
            <TableRow
              key={cat.id}
              onClick={() => onSelectCategory(cat)}
              $isEven={index % 2 === 0}
              $isSelected={selectedCategory?.id === cat.id}
            >
              <RowCell>
                <span>›</span>
                {cat.name}
              </RowCell>
              <RowCell>{cat.classification}</RowCell>
              <RowCell>{cat.count}</RowCell>
              <RowCell>[Subject: 10, Category: 5, ...]</RowCell>
              <RowCell>{cat.confidence}%</RowCell>
            </TableRow>
          ))}
        </RowsContainer>
      </TableGrid>
    </TableContainer>
  );
};

const TableContainer = styled.div`
  padding: 1.5rem;
  background: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
`;

const HeaderContainer = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
`;

const Title = styled.h1`
  font-size: 2.25rem;
  font-weight: 600;
`;

const SearchInput = styled.input`
  padding: 0.5rem 1rem;
  border: 1px solid #e2e8f0;
  border-radius: 0.5rem;
  outline: none;
  
  &:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
  }
`;

const TableGrid = styled.div`
  width: 100%;
`;

const HeaderGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1rem;
  padding: 0 1rem;
  margin-bottom: 0.5rem;
`;

const HeaderCell = styled.div`
  display: flex;
  align-items: center;
  font-size: 0.875rem;
  color: #4b5563;
  cursor: pointer;
  
  svg {
    margin-left: 0.25rem;
  }
`;

const RowsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const TableRow = styled.div<{ $isEven: boolean; $isSelected: boolean }>`
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1rem;
  padding: 0.75rem 1rem;
  border-radius: 0.5rem;
  cursor: pointer;
  background-color: ${props => 
    props.$isSelected 
      ? '#e0f2fe'  // Light blue highlight for selected row
      : props.$isEven 
        ? '#f3f4f6' 
        : '#eff6ff'
  };
  border: ${props => props.$isSelected ? '2px solid #60a5fa' : 'none'};

  &:hover {
    background-color: ${props => 
      props.$isSelected 
        ? '#dbeafe'  // Slightly darker on hover even when selected
        : props.$isEven 
          ? '#e5e7eb' 
          : '#dbeafe'
    };
  }
`;

const RowCell = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;