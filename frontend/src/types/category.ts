// src/types/category.ts

export interface Category {
    id: number;
    title: string;
    classification: string;
    count: number;
    confidence: number;
    path?: string;
    members?: Category[];
  }
  
  export interface CategoryUpdate {
    title: string;
  }
  
  export interface PaginatedResponse<T> {
    items: T[];
    total: number;
    page: number;
    page_size: number;
  }
  
  export interface SortConfig {
    key: keyof Category | null;
    direction: 'asc' | 'desc';
  }
  
  export interface FilterConfig {
    title: string;
    classification: string;
    count: string;
    confidence: string;
  }