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

export interface File extends Selectable {
  type?: "file";
  id: number;
  name: string;
  confidence: number;
  possibleClassifications: string[];
  originalPath: string;
  newPath: string | null;
  fileType?: string;
  size?: string;
}

export interface FolderV2 extends Selectable, Expandable {
  type?: "folder";
  id?: number;
  name: string;
  count: number;
  confidence: number;
  possibleClassifications?: string[];
  originalPath?: string;
  children: (File | FolderV2)[];
  path?: string;
}

export interface FolderViewResponse {
  original: FolderV2;
  new: FolderV2;
}

// Legacy interface for backward compatibility
export interface Folder extends FolderV2 {
  // Legacy properties for backward compatibility
  legacy?: boolean;
}

// Async task management types
export type TaskStatus = "pending" | "running" | "completed" | "failed";

export interface AsyncTaskResponse {
  task_id: string;
  message: string;
  status: TaskStatus;
}

export interface TaskInfo {
  task_id: string;
  status: TaskStatus;
  message: string;
  progress: number;
  result?: Record<string, unknown> | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
}
