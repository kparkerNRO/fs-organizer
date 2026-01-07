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

// Dual Representation Types (V2 API)

/**
 * Represents either a file/directory from the filesystem (Node) or a semantic category.
 * Files are always leaf nodes. ZIP files are treated as folders.
 */
export interface HierarchyItem {
  id: string; // e.g., "node-123", "category-abc"
  name: string;
  type: 'node' | 'category';
  originalPath?: string; // For nodes
}

/**
 * Stores all items in a flattened structure for quick lookup.
 */
export type ItemStore = Record<string, HierarchyItem>;

/**
 * Represents the parent-child relationships using IDs.
 * The key is the parent ID, and the value is an array of child IDs.
 */
export type Hierarchy = Record<string, string[]>;

/**
 * The complete data structure sent from the backend.
 */
export interface DualRepresentation {
  items: ItemStore;
  node_hierarchy: Hierarchy;
  category_hierarchy: Hierarchy;
}

/**
 * Represents changes made by the user on the frontend (moving nodes between categories).
 * Reordering of children within a category is not supported.
 */
export interface HierarchyDiff {
  // Key: Parent ID. Value: An array of child IDs that were added.
  added: Record<string, string[]>;
  // Key: Parent ID. Value: An array of child IDs that were removed.
  deleted: Record<string, string[]>;
}
