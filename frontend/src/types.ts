// src/types.ts

type SelectableItem = {
  item: Category | Folder;
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
  cleanedName: string;
  confidence: number;
  original_path: string;
  processed_names?: string[];
  isSelected?: boolean;
}
  