// src/mockData.ts

import { Folder, Category } from "../types/types";
import { FolderNode } from "../api";

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

export const mockFolderStructure: FolderNode = {
  id: "root",
  name: "Root",
  children: [
    {
      id: "gaming",
      name: "Gaming",
      path: "/gaming",
      children: [
        {
          id: "gaming/maps",
          name: "Maps",
          path: "/gaming/maps",
          children: [
            {
              id: "gaming/maps/wizard-tower",
              name: "Wizard Tower",
              path: "/gaming/maps/wizard-tower",
              children: [
                {
                  id: "gaming/maps/wizard-tower/green",
                  name: "Green",
                  path: "/gaming/maps/wizard-tower/green"
                },
                {
                  id: "gaming/maps/wizard-tower/blue",
                  name: "Blue",
                  path: "/gaming/maps/wizard-tower/blue"
                }
              ]
            },
            {
              id: "gaming/maps/dragon-lair",
              name: "Dragon Lair",
              path: "/gaming/maps/dragon-lair"
            }
          ]
        },
        {
          id: "gaming/assets",
          name: "Assets",
          path: "/gaming/assets",
          children: [
            {
              id: "gaming/assets/characters",
              name: "Characters",
              path: "/gaming/assets/characters"
            },
            {
              id: "gaming/assets/props",
              name: "Props",
              path: "/gaming/assets/props"
            }
          ]
        }
      ]
    },
    {
      id: "music",
      name: "Music",
      path: "/music",
      children: [
        {
          id: "music/classical",
          name: "Classical",
          path: "/music/classical"
        },
        {
          id: "music/jazz",
          name: "Jazz",
          path: "/music/jazz"
        }
      ]
    }
  ]
};
