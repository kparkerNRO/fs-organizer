import React, { useState, useEffect } from "react";
import styled from "styled-components";
import {
  Folder,
  LegacyFolder,
  SortConfig,
  CategoryDetailsProps,
} from "../types/types";
import { ChevronDown, ChevronRight, ChevronUp, X } from "lucide-react";
import { ContextMenu } from "./ContextMenu";
import { Pagination } from "./Pagination";
import { usePageState } from "../hooks/usePageState";
import { SORT_FIELD, SORT_ORDER } from "../types/enums";

interface CategoryTableProps {
  categories: Folder[];
  onSelectItem: (category_info: CategoryDetailsProps) => void;
  onUpdateCategories: (updatedCategories: Folder[]) => void;
  currentPage: number;
  totalPages: number;
  pageSize: number;
  totalItems: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
  onSortChange: (sortConfig: SortConfig) => void;
}

interface ContextMenuState {
  show: boolean;
  x: number;
  y: number;
}

export const CategoryTable: React.FC<CategoryTableProps> = ({
  categories,
  onSelectItem,
  onUpdateCategories,
  currentPage,
  totalPages,
  pageSize,
  totalItems,
  onPageChange,
  onPageSizeChange,
  onSortChange,
}) => {
  const [activeCategory, setActiveCategory] = useState<Folder | null>(null);
  const [selectedFolders, setSelectedFolders] = useState<LegacyFolder[]>([]);
  const [draggedOverCategoryId, setDraggedOverCategoryId] = useState<
    number | null
  >(null);
  const [contextMenu, setContextMenu] = useState<ContextMenuState>({
    show: false,
    x: 0,
    y: 0,
  });

  const {
    pageState,
    updateSortConfig,
    updateExpandedCategories,
    updateSelectedItem,
  } = usePageState();

  // Initialize state from pageState
  const [expandedCategories, setExpandedCategories] = useState<number[]>(
    pageState.expandedCategories,
  );

  // Handle sort column click
  const handleHeaderClick = (field: SORT_FIELD) => {
    const currentSort = pageState.sortConfig;
    let newSort: SortConfig;

    if (currentSort.field === field) {
      if (currentSort.direction === SORT_ORDER.ASC) {
        newSort = { field, direction: SORT_ORDER.DESC };
      } else if (currentSort.direction === SORT_ORDER.DESC) {
        newSort = { field: SORT_FIELD.NAME, direction: SORT_ORDER.ASC };
      } else {
        newSort = { field, direction: SORT_ORDER.ASC };
      }
    } else {
      newSort = { field, direction: SORT_ORDER.ASC };
    }

    updateSortConfig(newSort);
    onSortChange(newSort);
  };

  // Render sort icon based on current sort state
  const SortIcon = ({ field }: { field: SORT_FIELD }) => {
    const { sortConfig } = pageState;

    if (sortConfig.field !== field) {
      return (
        <div className="w-4 h-4 opacity-0 group-hover:opacity-50">
          <ChevronUp size={16} />
        </div>
      );
    }

    if (sortConfig.direction === SORT_ORDER.NONE) {
      return (
        <div className="w-4 h-4 text-gray-400 hover:text-gray-600">
          <X size={16} />
        </div>
      );
    }

    return (
      <div className="w-4 h-4 text-blue-500">
        {sortConfig.direction === "asc" ? (
          <ChevronUp size={16} />
        ) : (
          <ChevronDown size={16} />
        )}
      </div>
    );
  };

  // Update persisted state when expandedCategories changes
  useEffect(() => {
    updateExpandedCategories(expandedCategories);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expandedCategories]);

  // Handle updates to selection
  useEffect(() => {
    if (selectedFolders.length > 1) {
      onSelectItem({ category: null, folder: null });
      updateSelectedItem(null);
    } else if (selectedFolders.length === 1) {
      onSelectItem({ category: activeCategory, folder: selectedFolders[0] });
      updateSelectedItem(selectedFolders[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFolders, onSelectItem]);

  useEffect(() => {
    if (activeCategory && selectedFolders.length === 0) {
      onSelectItem({ category: activeCategory, folder: null });
      updateSelectedItem(activeCategory.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCategory, selectedFolders.length, onSelectItem]);

  const handleContextMenu = (
    e: React.MouseEvent,
    folder: LegacyFolder,
    parentCategory: Folder,
  ) => {
    e.preventDefault();

    // If the folder isn't already selected, select it
    if (!selectedFolders.some((f) => f.id === folder.id)) {
      handleFolderSelection(folder, parentCategory, e);
    }

    setContextMenu({
      show: true,
      x: e.clientX,
      y: e.clientY,
    });
  };

  const closeContextMenu = () => {
    setContextMenu({ show: false, x: 0, y: 0 });
  };

  const createNewGroup = (name?: string) => {
    if (selectedFolders.length === 0) return;

    const categoryName = name == null ? selectedFolders[0].name : name;

    // Create new category based on first selected folder
    const newCategory: Folder = {
      id: Math.max(...categories.map((c) => c.id)) + 1,
      name: categoryName,
      classification: selectedFolders[0].classification,
      confidence: selectedFolders[0].confidence,
      children: selectedFolders,
      count: selectedFolders.length,
    };

    // Find and remove selected folders from their current categories
    const updatedCategories = categories.map((category) => ({
      ...category,
      children:
        category.children?.filter(
          (child) => !selectedFolders.some((f) => f.id === child.id),
        ) || [],
    }));

    // Add the new category
    onUpdateCategories([...updatedCategories, newCategory]);

    // Update local state
    setSelectedFolders([]);
    setActiveCategory(newCategory);
    setExpandedCategories((prev) => [...prev, newCategory.id]);
    closeContextMenu();
  };

  const createGroupWithCommonPrefix = (folders: LegacyFolder[]) => {
    if (selectedFolders.length === 0) return;

    //Find the max common prefix (in whole words) of the folder names
    const names = folders.map((folder) => folder.name);
    if (names.length === 0) return "";

    const splitNames = names.map((name) => name.split(" "));
    const minLength = Math.min(...splitNames.map((parts) => parts.length));

    let commonPrefix = "";
    for (let i = 0; i < minLength; i++) {
      const wordSet = new Set(splitNames.map((parts) => parts[i]));
      if (wordSet.size === 1) {
        commonPrefix += `${splitNames[0][i]} `;
      } else {
        break;
      }
    }

    const prefix = commonPrefix.trim();

    // Create new group with the common prefix and move the folders to it
    const newCategory: Folder = {
      id: Math.max(...categories.map((c) => c.id)) + 1,
      name: prefix,
      classification: selectedFolders[0].classification,
      confidence: selectedFolders[0].confidence,
      children: selectedFolders.map((child) =>
        folders.some((f) => f.id === child.id)
          ? { ...child, name: child.name.replace(prefix, "").trim() }
          : child,
      ),
      count: selectedFolders.length,
    };

    // Find and remove selected folders from their current categories
    const updatedCategories = categories.map((category) => ({
      ...category,
      children:
        category.children?.filter(
          (child) => !selectedFolders.some((f) => f.id === child.id),
        ) || [],
    }));

    // Update the categories
    onUpdateCategories([...updatedCategories, newCategory]);
    setSelectedFolders([]);
    setActiveCategory(newCategory);
    setExpandedCategories((prev) => [...prev, newCategory.id]);
    closeContextMenu();
  };

  const handleDragStart = (
    folder: LegacyFolder,
    category: Folder,
    e: React.DragEvent,
  ) => {
    const working_folders = selectedFolders.some((f) => f.id === folder.id)
      ? selectedFolders
      : [folder];
    setSelectedFolders(working_folders);

    // Store the source category ID and selected folders in the drag data
    e.dataTransfer.setData(
      "application/json",
      JSON.stringify({
        sourceCategoryId: category.id,
        folders: working_folders,
      }),
    );
  };

  const handleDragOver = (e: React.DragEvent, categoryId: number) => {
    e.preventDefault();
    setDraggedOverCategoryId(categoryId);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    const relatedTarget = e.relatedTarget as HTMLElement;
    if (
      !relatedTarget?.closest(`[data-category-id="${draggedOverCategoryId}"]`)
    ) {
      setDraggedOverCategoryId(null);
    }
  };

  const handleDrop = async (targetCategory: Folder, e: React.DragEvent) => {
    e.preventDefault();
    setDraggedOverCategoryId(null);

    try {
      const data = JSON.parse(e.dataTransfer.getData("application/json"));
      const { sourceCategoryId, folders } = data;

      if (sourceCategoryId === targetCategory.id) return;

      const foldersToMove = folders.map((folder: LegacyFolder) => ({
        ...folder,
      }));

      const updatedCategories = categories.map((category) => {
        // source case
        if (category.id === sourceCategoryId) {
          const new_children = (category.children || []).filter(
            (child) =>
              !foldersToMove.some((f: { id: number }) => f.id === child.id),
          );
          return {
            ...category,
            children: new_children,
          };
        }

        // target case
        if (category.id === targetCategory.id) {
          const new_children = [...(category.children || []), ...foldersToMove];
          return {
            ...category,
            children: new_children,
          };
        }

        // default case
        return { ...category };
      });

      // Update state
      onUpdateCategories(updatedCategories);

      const updatedTargetCategory = updatedCategories.find(
        (c) => c.id === targetCategory.id,
      );
      setActiveCategory(updatedTargetCategory || null);

      setSelectedFolders([]);

      if (!expandedCategories.includes(targetCategory.id)) {
        setExpandedCategories((prev) => [...prev, targetCategory.id]);
      }
    } catch (error) {
      console.error("Error processing drop:", error);
    }
  };

  const toggleExpand = (categoryId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedCategories((prev) =>
      prev.includes(categoryId)
        ? prev.filter((id) => id !== categoryId)
        : [...prev, categoryId],
    );
  };

  const handleFolderSelection = (
    folder: LegacyFolder,
    parentCategory: Folder,
    e: React.MouseEvent,
  ) => {
    e.stopPropagation();

    if (e.ctrlKey || e.metaKey) {
      if (activeCategory?.id === parentCategory.id) {
        setSelectedFolders((prev) => {
          const isAlreadySelected = prev.some((f) => f.id === folder.id);
          const newSelection = isAlreadySelected
            ? prev.filter((f) => f.id !== folder.id)
            : [...prev, folder];

          return newSelection;
        });
      } else {
        setActiveCategory(parentCategory);
        setSelectedFolders([folder]);
      }
    } else {
      // Handle normal click
      setSelectedFolders([folder]);
      setActiveCategory(parentCategory);
    }
  };

  const handleCategorySelection = (category: Folder) => {
    setSelectedFolders([]);
    setActiveCategory(category);
  };

  const renderCategory = (category: Folder, index: number) => (
    <CategoryGroup
      key={category.id}
      $isDraggedOver={category.id === draggedOverCategoryId}
      onDragOver={(e) => handleDragOver(e, category.id)}
      onDragLeave={handleDragLeave}
      onDrop={(e) => handleDrop(category, e)}
    >
      <TableRow
        $isEven={index % 2 === 0}
        $isSelected={category.id === activeCategory?.id}
        onClick={() => handleCategorySelection(category)}
      >
        <RowCell>
          {category.children && category.children.length > 0 ? (
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
        <RowCell>{category.confidence * 100}%</RowCell>
      </TableRow>
      {expandedCategories.includes(category.id) &&
        category.children?.map((folder) => (
          <TableRow
            key={folder.id}
            $isEven={index % 2 === 0}
            $isSelected={selectedFolders.some((f) => f.id === folder.id)}
            $isChild
            draggable
            onClick={(e) => handleFolderSelection(folder, category, e)}
            onContextMenu={(e) => handleContextMenu(e, folder, category)}
            onDragStart={(e) => handleDragStart(folder, category, e)}
          >
            <RowCell>
              <IndentSpace />
              {folder.name}
            </RowCell>
            <RowCell>{folder.classification}</RowCell>
            <RowCell>-</RowCell>
            {/* <RowCell>{folder.original_filename}</RowCell> */}
            <RowCell>{folder.processed_names?.join(", ")}</RowCell>
            <RowCell>{folder.confidence}%</RowCell>
          </TableRow>
        ))}
    </CategoryGroup>
  );

  return (
    <TableContainer>
      <HeaderContainer>
        <SearchInput placeholder="Search" />
      </HeaderContainer>

      <TableGrid>
        <HeaderGrid>
          <HeaderCell
            $active={pageState.sortConfig.field === SORT_FIELD.NAME}
            onClick={() => handleHeaderClick(SORT_FIELD.NAME)}
          >
            Name <SortIcon field={SORT_FIELD.NAME} />
          </HeaderCell>
          <HeaderCell
            $active={pageState.sortConfig.field === SORT_FIELD.CLASSIFICATION}
            onClick={() => handleHeaderClick(SORT_FIELD.CLASSIFICATION)}
          >
            Classification <SortIcon field={SORT_FIELD.CLASSIFICATION} />
          </HeaderCell>
          <HeaderCell
            $active={pageState.sortConfig.field === SORT_FIELD.COUNT}
            onClick={() => handleHeaderClick(SORT_FIELD.COUNT)}
          >
            Count <SortIcon field={SORT_FIELD.COUNT} />
          </HeaderCell>
          <HeaderCell>Possible classifications</HeaderCell>
          <HeaderCell
            $active={pageState.sortConfig.field === SORT_FIELD.CONFIDENCE}
            onClick={() => handleHeaderClick(SORT_FIELD.CONFIDENCE)}
          >
            Confidence
            <SortIcon field={SORT_FIELD.CONFIDENCE} />
          </HeaderCell>
        </HeaderGrid>

        <RowsContainer>
          {categories.map((category, index) => renderCategory(category, index))}
        </RowsContainer>
      </TableGrid>
      <Pagination
        currentPage={currentPage}
        totalPages={totalPages}
        pageSize={pageSize}
        totalItems={totalItems}
        onPageChange={onPageChange}
        onPageSizeChange={onPageSizeChange}
      />
      {contextMenu.show && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          onClose={closeContextMenu}
          menu_items={[
            {
              text: "Create new group",
              onClick: () => createNewGroup(),
            },
            {
              text: "Extract common prefix",
              onClick: () => createGroupWithCommonPrefix(selectedFolders),
            },
          ]}
        />
      )}
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
  width: 100%;
  box-sizing: border-box;
  overflow-x: auto;
`;

const HeaderContainer = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
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

const HeaderCell = styled.div<{
  $active?: boolean;
}>`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  font-weight: 500;
  color: ${(props) => (props.$active ? "#2563eb" : "#4b5563")};
  cursor: pointer;
  transition: all 0.2s;

  &:hover {
    background-color: #f3f4f6;
    border-radius: 0.375rem;
  }

  /* Sort icon container */
  > div {
    display: flex;
    align-items: center;
    transition: all 0.2s;
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
  padding: 0.6rem 0.75rem; // Reduced from 0.75rem 1rem
  border-radius: 0.375rem; // Reduced from 0.5rem
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

const CategoryGroup = styled.div<{ $isDraggedOver: boolean }>`
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 0.25rem; // Reduced from 0.5rem
  margin-bottom: 0.1rem; // Reduced from 0.5rem

  ${(props) =>
    props.$isDraggedOver &&
    `
    &::after {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: rgba(147, 197, 253, 0.2);
      border: 2px dashed #2563eb;
      border-radius: 0.375rem;
      pointer-events: none;
    }
  `}

  &:last-child {
    margin-bottom: 0;
  }
`;
