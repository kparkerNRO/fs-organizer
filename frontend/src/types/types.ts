// src/types.ts
import { SORT_FIELD, SORT_ORDER } from "./enums";

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
  children?: LegacyFolder[];
  isExpanded?: boolean;
  isSelected?: boolean;
}

export interface LegacyFolder {
  id: number;
  name: string;
  classification: string;
  original_filename: string;
  cleaned_name: string;
  confidence: number;
  original_path: string;
  processed_names?: string[];
  isSelected?: boolean;
}

export interface LegacyFileItem {
  id: number;
  name: string;
  fileType: string;
  size?: string;
  original_path: string;
  confidence: number;
  categories?: string[];
}

export interface CategoryDetailsProps {
  category?: Category | null;
  folder?: LegacyFolder | null;
  file?: LegacyFileItem | null;
}

// New data structures for folder view
export interface Selectable {
  isSelected?: boolean;
}

export interface Expandable {
  isExpanded?: boolean;

}

export interface Folder extends Selectable, Expandable {
  name: string;
  count: number;
  confidence: number;
  children?: (Folder | File)[];
}

export interface File extends Selectable {
  id: number;
  name: string;
  confidence: number;
  possibleClassifications?: string[];
  originalPath: string;
  newPath: string;
}