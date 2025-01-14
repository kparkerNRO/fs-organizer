// src/mockData.ts

import { Folder } from "../types";

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
