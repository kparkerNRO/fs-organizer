import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { Category, Folder } from "../types";
import { ChevronDown, ChevronRight } from "lucide-react";

interface CategoryTableProps {
  categories: Category[];
  onSelectItem: (item: Category | Folder | Folder[] | null) => void;
}

export const CategoryTable: React.FC<CategoryTableProps> = ({
  categories,
  onSelectItem,
}) => {
  const [expandedCategories, setExpandedCategories] = useState<number[]>([]);
  const [activeCategory, setActiveCategory] = useState<Category | null>(null);

  const [selectedFolders, setSelectedFolders] = useState<Folder[]>([]);

  // Handle updates to selection
  useEffect(() => {
    if (selectedFolders.length > 1) {
      onSelectItem(null);
    } else if (selectedFolders.length === 1) {
      onSelectItem(selectedFolders[0]);
    }
  }, [selectedFolders, onSelectItem]);

  // Handle category selection updates
  useEffect(() => {
    if (activeCategory && selectedFolders.length === 0) {
      onSelectItem(activeCategory);
    }
  }, [activeCategory, selectedFolders.length, onSelectItem]);

  const toggleExpand = (categoryId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedCategories((prev) =>
      prev.includes(categoryId)
        ? prev.filter((id) => id !== categoryId)
        : [...prev, categoryId]
    );
  };

  const handleFolderSelection = (
    folder: Folder,
    parentCategory: Category,
    e: React.MouseEvent
  ) => {
    e.stopPropagation();

    if (e.ctrlKey || e.metaKey) {
      // if (activeCategory && !activeCategory.children?.includes(folder)) {
      if (activeCategory?.id === parentCategory.id) {
        setSelectedFolders((prev) => {
          const isAlreadySelected = prev.some((f) => f.id === folder.id);
          const newSelection = isAlreadySelected
            ? prev.filter((f) => f.id !== folder.id)
            : [...prev, folder];

          return newSelection;
        });
      }
    } else {
      // Handle normal click
      setSelectedFolders([folder]);
      setActiveCategory(parentCategory);
    }
  };

  const handleCategorySelection = (category: Category) => {
    // setSelectedFolders([]);
    // onSelectItem(category);
    if (
      selectedFolders.length > 0 &&
      !category.children?.some((folder) => folder.id === selectedFolders[0].id)
    ) {
      setSelectedFolders([]);
    }
    setActiveCategory(category);
  };

  const renderCategory = (category: Category, index: number) => (
    <React.Fragment key={category.id}>
      <TableRow
        $isEven={index % 2 === 0}
        $isSelected={category.id === activeCategory?.id}
        onClick={() => handleCategorySelection(category)}
      >
        <RowCell>
          {category.children ? (
            <ExpandButton onClick={(e) => toggleExpand(category.id, e)}>
              {expandedCategories.includes(category.id) ? (
                <ChevronDown size={16} />
              ) : (
                <ChevronRight size={16} />
              )}
            </ExpandButton>
          ) : (
            <ExpandButton></ExpandButton>
          )}
          {category.name}
        </RowCell>
        <RowCell>{category.classification}</RowCell>
        <RowCell>{category.count}</RowCell>
        <RowCell>{category.possibleClassifications?.join(", ") || "-"}</RowCell>
        <RowCell>{category.confidence}%</RowCell>
      </TableRow>
      {expandedCategories.includes(category.id) &&
        category.children?.map((folder) => (
          <TableRow
            key={folder.id}
            $isEven={index % 2 === 0}
            $isSelected={selectedFolders.some((f) => f.id === folder.id)}
            $isChild
            onClick={(e) => handleFolderSelection(folder, category, e)}
          >
            <RowCell>
              <IndentSpace />
              {folder.name}
            </RowCell>
            <RowCell>{folder.classification}</RowCell>
            <RowCell>-</RowCell>
            <RowCell>{folder.original_filename}</RowCell>
            <RowCell>{folder.confidence}%</RowCell>
          </TableRow>
        ))}
    </React.Fragment>
  );

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
          {categories.map((category, index) => renderCategory(category, index))}
        </RowsContainer>
      </TableGrid>
    </TableContainer>
  );
};

const ExpandButton = styled.span`
  margin-right: 0.5rem;
  display: inline-flex;
  align-items: center;
`;

const IndentSpace = styled.span`
  display: inline-block;
  width: 1.5rem;
`;

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

const TableRow = styled.div<{
  $isEven: boolean;
  $isSelected: boolean;
  $isChild?: boolean;
}>`
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1rem;
  padding: 0.75rem 1rem;
  border-radius: 0.5rem;
  cursor: pointer;
  background-color: ${(props) =>
    props.$isSelected ? "#e0f2fe" : props.$isEven ? "#f3f4f6" : "#eff6ff"};
  border: ${(props) => (props.$isSelected ? "2px solid #60a5fa" : "none")};
  margin-left: ${(props) => (props.$isChild ? "1.5rem" : "0")};

  &:hover {
    background-color: ${(props) =>
      props.$isSelected ? "#dbeafe" : props.$isEven ? "#e5e7eb" : "#dbeafe"};
  }
`;
const RowCell = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;
