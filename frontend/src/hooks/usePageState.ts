// src/hooks/usePageState.ts

import { useState, useEffect } from "react";
import { PageState, SortConfig } from "../types/types";
import { SORT_FIELD, SORT_ORDER } from "../types/enums";

const PAGE_STATE_KEY = "categoryPageState";

const defaultPageState: PageState = {
  sortConfig: {
    field: SORT_FIELD.NAME,
    direction: SORT_ORDER.ASC,
  },
  expandedCategories: [],
  selectedItem: null,
};

export const usePageState = () => {
  // Initialize state from localStorage or use defaults
  const [pageState, setPageState] = useState<PageState>(() => {
    try {
      const savedState = localStorage.getItem(PAGE_STATE_KEY);
      return savedState ? JSON.parse(savedState) : defaultPageState;
    } catch (error) {
      console.error("Error loading page state from localStorage:", error);
      return defaultPageState;
    }
  });

  // Save to localStorage whenever state changes
  useEffect(() => {
    try {
      localStorage.setItem(PAGE_STATE_KEY, JSON.stringify(pageState));
    } catch (error) {
      console.error("Error saving page state to localStorage:", error);
    }
  }, [pageState]);

  const updateSortConfig = (newConfig: SortConfig) => {
    setPageState((prev) => ({
      ...prev,
      sortConfig: newConfig,
    }));
  };

  const updateExpandedCategories = (categoryIds: number[]) => {
    setPageState((prev) => ({
      ...prev,
      expandedCategories: categoryIds,
    }));
  };

  const updateSelectedItem = (itemId: number | null) => {
    setPageState((prev) => ({
      ...prev,
      selectedItem: itemId,
    }));
  };

  const resetPageState = () => {
    setPageState(defaultPageState);
  };

  return {
    pageState,
    updateSortConfig,
    updateExpandedCategories,
    updateSelectedItem,
    resetPageState,
  };
};
