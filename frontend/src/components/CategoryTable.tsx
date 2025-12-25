import React, { useState, useEffect } from "react";
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
    pageState.expandedCategories
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
    parentCategory: Folder
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
          (child) => !selectedFolders.some((f) => f.id === child.id)
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
          : child
      ),
      count: selectedFolders.length,
    };

    // Find and remove selected folders from their current categories
    const updatedCategories = categories.map((category) => ({
      ...category,
      children:
        category.children?.filter(
          (child) => !selectedFolders.some((f) => f.id === child.id)
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
    e: React.DragEvent
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
      })
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

      const foldersToMove = folders.map((folder: LegacyFolder) => ({ ...folder }));

      const updatedCategories = categories.map((category) => {
        // source case
        if (category.id === sourceCategoryId) {
          const new_children = (category.children || []).filter(
            (child) =>
              !foldersToMove.some((f: { id: number }) => f.id === child.id)
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
        (c) => c.id === targetCategory.id
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
        : [...prev, categoryId]
    );
  };

  const handleFolderSelection = (
    folder: LegacyFolder,
    parentCategory: Folder,
    e: React.MouseEvent
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

  const renderCategory = (category: Folder, index: number) => {
    const isDraggedOver = category.id === draggedOverCategoryId;
    const isEven = index % 2 === 0;
    const isCategorySelected = category.id === activeCategory?.id;

    return (
      <div
        key={category.id}
        className="relative flex flex-col gap-1 mb-0.5 last:mb-0"
        data-category-id={category.id}
        onDragOver={(e) => handleDragOver(e, category.id)}
        onDragLeave={handleDragLeave}
        onDrop={(e) => handleDrop(category, e)}
      >
        {/* Drag overlay */}
        {isDraggedOver && (
          <div className="absolute inset-0 bg-blue-300 bg-opacity-20 border-2 border-dashed border-blue-600 rounded-md pointer-events-none" />
        )}

        {/* Category row */}
        <div
          className={`grid grid-cols-5 gap-4 px-3 py-2.5 rounded-md cursor-pointer ${
            isCategorySelected
              ? "bg-sky-200 border-2 border-blue-400"
              : isEven
              ? "bg-gray-100"
              : "bg-blue-50"
          } ${
            isCategorySelected
              ? "hover:bg-sky-100"
              : isEven
              ? "hover:bg-gray-200"
              : "hover:bg-sky-100"
          }`}
          onClick={() => handleCategorySelection(category)}
        >
          <div className="flex items-center gap-2">
            {category.children && category.children.length > 0 ? (
              <span
                className="mr-2 inline-flex items-center"
                onClick={(e) => toggleExpand(category.id, e)}
              >
                {expandedCategories.includes(category.id) ? (
                  <ChevronDown size={16} />
                ) : (
                  <ChevronRight size={16} />
                )}
              </span>
            ) : (
              <span className="mr-2 inline-flex items-center"></span>
            )}
            {category.name}
          </div>
          <div className="flex items-center gap-2">{category.classification}</div>
          <div className="flex items-center gap-2">{category.count}</div>
          <div className="flex items-center gap-2">
            {category.possibleClassifications?.join(", ") || "-"}
          </div>
          <div className="flex items-center gap-2">{category.confidence * 100}%</div>
        </div>

        {/* Child folders */}
        {expandedCategories.includes(category.id) &&
          category.children?.map((folder) => {
            const isFolderSelected = selectedFolders.some((f) => f.id === folder.id);

            return (
              <div
                key={folder.id}
                className={`grid grid-cols-5 gap-4 px-3 py-2.5 rounded-md cursor-pointer ml-6 ${
                  isFolderSelected
                    ? "bg-sky-200 border-2 border-blue-400"
                    : isEven
                    ? "bg-gray-100"
                    : "bg-blue-50"
                } ${
                  isFolderSelected
                    ? "hover:bg-sky-100"
                    : isEven
                    ? "hover:bg-gray-200"
                    : "hover:bg-sky-100"
                }`}
                draggable
                onClick={(e) => handleFolderSelection(folder, category, e)}
                onContextMenu={(e) => handleContextMenu(e, folder, category)}
                onDragStart={(e) => handleDragStart(folder, category, e)}
              >
                <div className="flex items-center gap-2">
                  <span className="inline-block w-6" />
                  {folder.name}
                </div>
                <div className="flex items-center gap-2">{folder.classification}</div>
                <div className="flex items-center gap-2">-</div>
                <div className="flex items-center gap-2">
                  {folder.processed_names?.join(", ")}
                </div>
                <div className="flex items-center gap-2">{folder.confidence}%</div>
              </div>
            );
          })}
      </div>
    );
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-sm w-full box-border overflow-x-auto">
      <div className="flex justify-between items-center mb-8">
        <input
          className="px-4 py-2 border border-gray-300 rounded-lg outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          placeholder="Search"
        />
      </div>

      <div className="w-full">
        <div className="grid grid-cols-5 gap-4 px-4 mb-2">
          <div
            className={`group flex items-center gap-2 p-3 font-medium cursor-pointer transition-all hover:bg-gray-100 hover:rounded-md ${
              pageState.sortConfig.field === SORT_FIELD.NAME
                ? "text-blue-600"
                : "text-gray-600"
            }`}
            onClick={() => handleHeaderClick(SORT_FIELD.NAME)}
          >
            Name <SortIcon field={SORT_FIELD.NAME} />
          </div>
          <div
            className={`group flex items-center gap-2 p-3 font-medium cursor-pointer transition-all hover:bg-gray-100 hover:rounded-md ${
              pageState.sortConfig.field === SORT_FIELD.CLASSIFICATION
                ? "text-blue-600"
                : "text-gray-600"
            }`}
            onClick={() => handleHeaderClick(SORT_FIELD.CLASSIFICATION)}
          >
            Classification <SortIcon field={SORT_FIELD.CLASSIFICATION} />
          </div>
          <div
            className={`group flex items-center gap-2 p-3 font-medium cursor-pointer transition-all hover:bg-gray-100 hover:rounded-md ${
              pageState.sortConfig.field === SORT_FIELD.COUNT
                ? "text-blue-600"
                : "text-gray-600"
            }`}
            onClick={() => handleHeaderClick(SORT_FIELD.COUNT)}
          >
            Count <SortIcon field={SORT_FIELD.COUNT} />
          </div>
          <div className="flex items-center gap-2 p-3 font-medium text-gray-600">
            Possible classifications
          </div>
          <div
            className={`group flex items-center gap-2 p-3 font-medium cursor-pointer transition-all hover:bg-gray-100 hover:rounded-md ${
              pageState.sortConfig.field === SORT_FIELD.CONFIDENCE
                ? "text-blue-600"
                : "text-gray-600"
            }`}
            onClick={() => handleHeaderClick(SORT_FIELD.CONFIDENCE)}
          >
            Confidence
            <SortIcon field={SORT_FIELD.CONFIDENCE} />
          </div>
        </div>

        <div className="flex flex-col gap-2">
          {categories.map((category, index) => renderCategory(category, index))}
        </div>
      </div>
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
    </div>
  );
};
