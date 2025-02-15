// src/types.ts
import { SORT_FIELD, SORT_ORDER } from "./enums";
import { Category, Folder } from "./types";

export interface SortConfig {
  field: SORT_FIELD;
  direction: SORT_ORDER;
}

export interface PageState {
  sortConfig: SortConfig;
  expandedCategories: number[];
  selectedItem: number | null;
}

export interface Category {
  id: number;
  name: string;
  classification: string;
  count: number;
  confidence: number;
  possibleClassifications?: string[];
  children?: Folder[];
  isExpanded?: boolean;
  isSelected?: boolean;
}

export interface Folder {
  id: number;
  name: string;
  classification: string;
  original_filename: string;
  cleaned_name: string;
  confidence: number;
  original_path: string;
  processed_names?: string[];
  isSelected?: boolean;
}export interface CategoryDetailsProps {
  category?: Category | null ;
  folder?: Folder| null ;
}

