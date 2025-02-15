import { useState, useEffect } from "react";
import { Category } from "../types/types";

const STORAGE_KEY = "categoriesData";

export const usePersistedCategories = (initialCategories: Category[]) => {
  // Initialize state from localStorage or use initial categories
  const [categories, setCategories] = useState<Category[]>(() => {
    try {
      const savedCategories = localStorage.getItem(STORAGE_KEY);
      const result = savedCategories ? JSON.parse(savedCategories) : initialCategories;
      console.log("result", result);
      return result;
    } catch (error) {
      console.error("Error loading categories from localStorage:", error);
      return initialCategories;
    }
  });

  // 

  // Save to localStorage whenever categories change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(categories));
    } catch (error) {
      console.error("Error saving categories to localStorage:", error);
    }
  }, [categories]);

  // Function to reset to initial data
  const resetToInitial = (callback?: () => void) => {
    setCategories(initialCategories);
    if (callback) {
      callback();
    }
  };

  return {
    categories,
    setCategories,
    resetToInitial,
  };
};
