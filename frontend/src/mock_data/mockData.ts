// src/mockData.ts

import { Folder, Category } from "../types/types";
import { FolderNode, FileNode } from "../api";

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
                  path: "/gaming/maps/wizard-tower/green",
                  children: [
                    {
                      id: "gaming/maps/wizard-tower/green/map1.png",
                      name: "map1.png",
                      path: "/gaming/maps/wizard-tower/green/map1.png",
                      fileType: "image/png",
                      size: "2.4 MB",
                      categories: ["Wizard Tower", "Green"],
                      confidence: 92,
                      originalPath: "/original/path/to/wizard_tower_green_map1.png"
                    },
                    {
                      id: "gaming/maps/wizard-tower/green/map2.jpg",
                      name: "map2.jpg",
                      path: "/gaming/maps/wizard-tower/green/map2.jpg",
                      fileType: "image/jpeg",
                      size: "1.8 MB",
                      categories: ["Wizard Tower", "Green"],
                      confidence: 85,
                      originalPath: "/original/path/to/green_wizard_tower_map2.jpg"
                    }
                  ]
                },
                {
                  id: "gaming/maps/wizard-tower/blue",
                  name: "Blue",
                  path: "/gaming/maps/wizard-tower/blue",
                  children: [
                    {
                      id: "gaming/maps/wizard-tower/blue/blueprint.pdf",
                      name: "blueprint.pdf",
                      path: "/gaming/maps/wizard-tower/blue/blueprint.pdf",
                      fileType: "application/pdf",
                      size: "3.2 MB",
                      categories: ["Wizard Tower", "Blueprint"],
                      confidence: 78,
                      originalPath: "/original/path/to/blue_tower_blueprint.pdf"
                    }
                  ]
                }
              ]
            },
            {
              id: "gaming/maps/dragon-lair",
              name: "Dragon Lair",
              path: "/gaming/maps/dragon-lair",
              children: [
                {
                  id: "gaming/maps/dragon-lair/entrance.png",
                  name: "entrance.png",
                  path: "/gaming/maps/dragon-lair/entrance.png",
                  fileType: "image/png",
                  size: "4.1 MB",
                  categories: ["Dragon Lair", "Entrance"],
                  confidence: 95,
                  originalPath: "/original/path/to/dragon_lair_entrance.png"
                },
                {
                  id: "gaming/maps/dragon-lair/treasure-room.png",
                  name: "treasure-room.png",
                  path: "/gaming/maps/dragon-lair/treasure-room.png",
                  fileType: "image/png",
                  size: "3.8 MB",
                  categories: ["Dragon Lair", "Treasure"],
                  confidence: 89,
                  originalPath: "/original/path/to/dragon_treasure_room.png"
                }
              ]
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
              path: "/gaming/assets/characters",
              children: [
                {
                  id: "gaming/assets/characters/wizard.fbx",
                  name: "wizard.fbx",
                  path: "/gaming/assets/characters/wizard.fbx",
                  fileType: "model/fbx",
                  size: "12.6 MB",
                  categories: ["Character", "Wizard"],
                  confidence: 91,
                  originalPath: "/original/path/to/wizard_character.fbx"
                },
                {
                  id: "gaming/assets/characters/dragon.fbx",
                  name: "dragon.fbx",
                  path: "/gaming/assets/characters/dragon.fbx",
                  fileType: "model/fbx",
                  size: "28.3 MB",
                  categories: ["Character", "Dragon"],
                  confidence: 94,
                  originalPath: "/original/path/to/red_dragon_model.fbx"
                }
              ]
            },
            {
              id: "gaming/assets/props",
              name: "Props",
              path: "/gaming/assets/props",
              children: [
                {
                  id: "gaming/assets/props/treasure-chest.obj",
                  name: "treasure-chest.obj",
                  path: "/gaming/assets/props/treasure-chest.obj",
                  fileType: "model/obj",
                  size: "5.2 MB",
                  categories: ["Prop", "Treasure"],
                  confidence: 87,
                  originalPath: "/original/path/to/gold_treasure_chest.obj"
                }
              ]
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
          path: "/music/classical",
          children: [
            {
              id: "music/classical/beethoven.mp3",
              name: "beethoven.mp3",
              path: "/music/classical/beethoven.mp3",
              fileType: "audio/mp3",
              size: "8.7 MB",
              categories: ["Classical", "Beethoven"],
              confidence: 96,
              originalPath: "/original/path/to/beethoven_symphony_no9.mp3"
            }
          ]
        },
        {
          id: "music/jazz",
          name: "Jazz",
          path: "/music/jazz",
          children: [
            {
              id: "music/jazz/miles-davis.mp3",
              name: "miles-davis.mp3",
              path: "/music/jazz/miles-davis.mp3",
              fileType: "audio/mp3",
              size: "7.2 MB",
              categories: ["Jazz", "Miles Davis"],
              confidence: 93,
              originalPath: "/original/path/to/miles_davis_kind_of_blue.mp3"
            },
            {
              id: "music/jazz/coltrane.mp3",
              name: "coltrane.mp3",
              path: "/music/jazz/coltrane.mp3",
              fileType: "audio/mp3",
              size: "6.9 MB",
              categories: ["Jazz", "John Coltrane"],
              confidence: 88,
              originalPath: "/original/path/to/coltrane_a_love_supreme.mp3"
            }
          ]
        }
      ]
    }
  ]
};
