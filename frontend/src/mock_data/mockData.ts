// src/mockData.ts

import { Folder, Category } from "../types";

export const mockFolders: Folder[] = [
  {
    id: 1,
    folderName: "Dragon Lair",
    folderPath: "/home/user/dragon-lair",
    depth: 1,
    cleanedName: "dragon-lair",
    categories: [
      {
        id: 101,
        name: "Dragon Lair Blue",
        classification: "Subject",
        count: 85,
        confidence: 85,
        possibleClassifications: ["Subject", "Category"],
      },
    ],
  },
  {
    id: 2,
    folderName: "Wizard Tower",
    folderPath: "/home/user/wizard-tower",
    depth: 1,
    cleanedName: "wizard-tower",
    categories: [
      {
        id: 102,
        name: "Wizard Tower Green",
        classification: "Category",
        count: 60,
        confidence: 90,
        possibleClassifications: ["Category", "Unknown"],
      },
    ],
  },
];

export const mockCategoryData: Category[] = [
  {
    id: 1,
    name: 'Electronics',
    classification: 'Consumer Goods',
    count: 1243,
    confidence: 98.5
  },
  {
    id: 2,
    name: 'Books',
    classification: 'Media',
    count: 856,
    confidence: 99.2
  },
  {
    id: 3,
    name: 'Clothing',
    classification: 'Fashion',
    count: 2134,
    confidence: 97.8
  },
  {
    id: 4,
    name: 'Sports Equipment',
    classification: 'Recreation',
    count: 567,
    confidence: 80.4
  },
  {
    id: 5,
    name: 'Home Decor',
    classification: 'Furnishings',
    count: 923,
    confidence: 50.7
  },

];

// Example usage:
// import { mockCategoryData } from './mockData';
// 
// <CategoryTable 
//   data={mockCategoryData}
//   onRowSelect={(row) => console.log('Selected:', row)}
// />