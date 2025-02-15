// src/mockData.ts

import { Folder, Category } from "../types/types";

export const mockFolders: Folder[] = [];

export const mockCategoryData: Category[] = [
  {
    id: 1,
    name: "Dragon Lair",
    classification: "Subject",
    count: 85,
    confidence: 85,
    possibleClassifications: ["Subject: 10", "Category: 5"],
    isExpanded: false,
    children: [
      {
        id: 1,
        name: "Wizard Tower Red",
        classification: "Category",
        original_filename: "Wizard Tower Green Interior",
        cleaned_name: "Wizard Tower Green",
        confidence: 90,
        original_path: "C:\\gaming\\maps\\wizard-tower",
        processed_names: ["Wizard Tower Green"],
      },
      {
        id: 2,
        name: "Wizard Tower Orange",
        classification: "Category",
        original_filename: "Wizard Tower Blue Interior",
        cleaned_name: "Wizard Tower Blue",
        confidence: 90,
        original_path: "C:\\gaming\\maps\\wizard-tower",
        processed_names: ["Wizard Tower Blue"],
      },
    ],
  },
  {
    id: 2,
    name: "Wizard Tower",
    classification: "Category",
    count: 60,
    confidence: 20,
    possibleClassifications: ["Category: 5", "Unknown: 8"],
    isExpanded: true,
    children: [
      {
        id: 3,
        name: "Wizard Tower Green",
        classification: "Category",
        original_filename: "Wizard Tower Green Interior",
        cleaned_name: "Wizard Tower Green",
        confidence: 90,
        original_path: "C:\\gaming\\maps\\wizard-tower",
        processed_names: ["Wizard Tower Green"],
      },
      {
        id: 4,
        name: "Wizard Tower Blue",
        classification: "Category",
        original_filename: "Wizard Tower Blue Interior",
        cleaned_name: "Wizard Tower Blue",
        confidence: 90,
        original_path: "C:\\gaming\\maps\\wizard-tower",
        processed_names: ["Wizard Tower Blue"],
      },
      {
        id: 5,
        name: "Wizard Tower Yellow",
        classification: "Category",
        original_filename: "Wizard Tower Blue Interior",
        cleaned_name: "Wizard Tower Blue",
        confidence: 90,
        original_path: "C:\\gaming\\maps\\wizard-tower",
        processed_names: ["Wizard Tower Blue"],
      },
    ],
  },
  {
    id: 3,
    name: "Into the wilds",
    classification: "Subject",
    count: 85,
    confidence: 85,
    possibleClassifications: ["Subject: 10", "Category: 5"],
    isExpanded: false,
  },
  {
    id: 4,
    name: "Music",
    classification: "Subject",
    count: 85,
    confidence: 85,
    possibleClassifications: ["Subject: 10", "Category: 5"],
    isExpanded: false,
  },
];

// Example usage:
// import { mockCategoryData } from './mockData';
//
// <CategoryTable
//   data={mockCategoryData}
//   onRowSelect={(row) => console.log('Selected:', row)}
// />
