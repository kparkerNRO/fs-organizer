// src/types.ts

export interface Category {
    id: number;
    name: string;
    classification: string;
    count: number;
    confidence: number;
    possibleClassifications: string[];
  }
  
  export interface Folder {
    id: number;
    folderName: string;
    folderPath: string;
    depth: number;
    cleanedName: string;
    categories: Category[];
  }
  