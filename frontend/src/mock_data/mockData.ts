// src/mockData.ts

import { Folder, FolderV2 } from "../types/types";


export const mockCategoryData: Folder[] = [
  {
    name: "Dragon Lair",
    count: 85,
    confidence: 85,
    children: [],
    isExpanded: false,
  },
  {
    name: "Wizard Tower",
    count: 60,
    confidence: 20,
    children: [],
    isExpanded: true,
  },
  {
    name: "Into the wilds",
    count: 85,
    confidence: 85,
    children: [],
    isExpanded: false,
  },
  {
    name: "Music",
    count: 85,
    confidence: 85,
    children: [],
    isExpanded: false,
  },
];

export const mockOriginalFolderStructure: FolderV2 = {
  name: "Root_old",
  count: 11,
  confidence: 100,
  children: [
    {
      name: "Gaming_old",
      count: 8,
      confidence: 100,
      children: [
        {
          name: "Maps_old",
          count: 5,
          confidence: 100,
          children: [
            {
              name: "Wizard Tower_old",
              count: 3,
              confidence: 100,
              children: [
                {
                  name: "Green_old",
                  count: 2,
                  confidence: 100,
                  children: [
                    {
                      id: 1,
                      name: "map1.png",
                      confidence: 100,
                      possibleClassifications: ["Wizard Tower", "Green"],
                      originalPath: "/original/path/to/wizard_tower_green_map1.png",
                      newPath: "/gaming/maps/wizard-tower/green/map1.png"
                    },
                    {
                      id: 2,
                      name: "map2.jpg",
                      confidence: 100,
                      possibleClassifications: ["Wizard Tower", "Green"],
                      originalPath: "/original/path/to/green_wizard_tower_map2.jpg",
                      newPath: "/gaming/maps/wizard-tower/green/map2.jpg"
                    }
                  ]
                },
                {
                  name: "Blue_old",
                  count: 1,
                  confidence: 100,
                  children: [
                    {
                      id: 3,
                      name: "blueprint.pdf",
                      confidence: 100,
                      possibleClassifications: ["Wizard Tower", "Blueprint"],
                      originalPath: "/original/path/to/blue_tower_blueprint.pdf",
                      newPath: "/gaming/maps/wizard-tower/blue/blueprint.pdf"
                    }
                  ]
                }
              ]
            },
            {
              name: "Dragon Lair_old",
              count: 2,
              confidence: 100,
              children: [
                {
                  id: 4,
                  name: "entrance.png",
                  confidence: 100,
                  possibleClassifications: ["Dragon Lair", "Entrance"],
                  originalPath: "/original/path/to/dragon_lair_entrance.png",
                  newPath: "/gaming/maps/dragon-lair/entrance.png"
                },
                {
                  id: 5,
                  name: "treasure-room.png",
                  confidence: 100,
                  possibleClassifications: ["Dragon Lair", "Treasure"],
                  originalPath: "/original/path/to/dragon_treasure_room.png",
                  newPath: "/gaming/maps/dragon-lair/treasure-room.png"
                }
              ]
            }
          ]
        },
        {
          name: "Assets_old",
          count: 3,
          confidence: 100,
          children: [
            {
              name: "Characters_old",
              count: 2,
              confidence: 100,
              children: [
                {
                  id: 6,
                  name: "wizard.fbx",
                  confidence: 100,
                  possibleClassifications: ["Character", "Wizard"],
                  originalPath: "/original/path/to/wizard_character.fbx",
                  newPath: "/gaming/assets/characters/wizard.fbx"
                },
                {
                  id: 7,
                  name: "dragon.fbx",
                  confidence: 100,
                  possibleClassifications: ["Character", "Dragon"],
                  originalPath: "/original/path/to/red_dragon_model.fbx",
                  newPath: "/gaming/assets/characters/dragon.fbx"
                }
              ]
            },
            {
              name: "Props_old",
              count: 1,
              confidence: 100,
              children: [
                {
                  id: 8,
                  name: "treasure-chest.obj",
                  confidence: 100,
                  possibleClassifications: ["Prop", "Treasure"],
                  originalPath: "/original/path/to/gold_treasure_chest.obj",
                  newPath: "/gaming/assets/props/treasure-chest.obj"
                }
              ]
            }
          ]
        }
      ]
    },
    {
      name: "Music_old",
      count: 3,
      confidence: 100,
      children: [
        {
          name: "Classical_old",
          count: 1,
          confidence: 100,
          children: [
            {
              id: 9,
              name: "beethoven.mp3",
              confidence: 100,
              possibleClassifications: ["Classical", "Beethoven"],
              originalPath: "/original/path/to/beethoven_symphony_no9.mp3",
              newPath: "/music/classical/beethoven.mp3"
            }
          ]
        },
        {
          name: "Jazz_old",
          count: 2,
          confidence: 100,
          children: [
            {
              id: 10,
              name: "miles-davis.mp3",
              confidence: 100,
              possibleClassifications: ["Jazz", "Miles Davis"],
              originalPath: "/original/path/to/miles_davis_kind_of_blue.mp3",
              newPath: "/music/jazz/miles-davis.mp3"
            },
            {
              id: 11,
              name: "coltrane.mp3",
              confidence: 100,
              possibleClassifications: ["Jazz", "John Coltrane"],
              originalPath: "/original/path/to/coltrane_a_love_supreme.mp3",
              newPath: "/music/jazz/coltrane.mp3"
            }
          ]
        }
      ]
    }
  ]
};

export const mockFolderStructure: FolderV2 = {
  name: "Root_new",
  count: 11,
  confidence: 100,
  children: [
    {
      name: "Gaming_new",
      count: 8,
      confidence: 90,
      children: [
        {
          name: "Maps_new",
          count: 5,
          confidence: 95,
          children: [
            {
              name: "Wizard Tower_new",
              count: 3,
              confidence: 82,
              children: [
                {
                  name: "Green_new",
                  count: 2,
                  confidence: 70,
                  children: [
                    {
                      id: 1,
                      name: "map1.png",
                      confidence: 92,
                      possibleClassifications: ["Wizard Tower", "Green"],
                      originalPath: "/original/path/to/wizard_tower_green_map1.png",
                      newPath: "/gaming/maps/wizard-tower/green/map1.png"
                    },
                    {
                      id: 2,
                      name: "map2.jpg",
                      confidence: 85,
                      possibleClassifications: ["Wizard Tower", "Green"],
                      originalPath: "/original/path/to/green_wizard_tower_map2.jpg",
                      newPath: "/gaming/maps/wizard-tower/green/map2.jpg"
                    }
                  ]
                },
                {
                  name: "Blue_new",
                  count: 1,
                  confidence: 50,
                  children: [
                    {
                      id: 3,
                      name: "blueprint.pdf",
                      confidence: 78,
                      possibleClassifications: ["Wizard Tower", "Blueprint"],
                      originalPath: "/original/path/to/blue_tower_blueprint.pdf",
                      newPath: "/gaming/maps/wizard-tower/blue/blueprint.pdf"
                    }
                  ]
                }
              ]
            },
            {
              name: "Dragon Lair_new",
              count: 2,
              confidence: 93,
              children: [
                {
                  id: 4,
                  name: "entrance.png",
                  confidence: 95,
                  possibleClassifications: ["Dragon Lair", "Entrance"],
                  originalPath: "/original/path/to/dragon_lair_entrance.png",
                  newPath: "/gaming/maps/dragon-lair/entrance.png"
                },
                {
                  id: 5,
                  name: "treasure-room.png",
                  confidence: 89,
                  possibleClassifications: ["Dragon Lair", "Treasure"],
                  originalPath: "/original/path/to/dragon_treasure_room.png",
                  newPath: "/gaming/maps/dragon-lair/treasure-room.png"
                }
              ]
            }
          ]
        },
        {
          name: "Assets_new",
          count: 3,
          confidence: 80,
          children: [
            {
              name: "Characters_new",
              count: 2,
              confidence: 35,
              children: [
                {
                  id: 6,
                  name: "wizard.fbx",
                  confidence: 91,
                  possibleClassifications: ["Character", "Wizard"],
                  originalPath: "/original/path/to/wizard_character.fbx",
                  newPath: "/gaming/assets/characters/wizard.fbx"
                },
                {
                  id: 7,
                  name: "dragon.fbx",
                  confidence: 94,
                  possibleClassifications: ["Character", "Dragon"],
                  originalPath: "/original/path/to/red_dragon_model.fbx",
                  newPath: "/gaming/assets/characters/dragon.fbx"
                }
              ]
            },
            {
              name: "Props_new",
              count: 1,
              confidence: 60,
              children: [
                {
                  id: 8,
                  name: "treasure-chest.obj",
                  confidence: 87,
                  possibleClassifications: ["Prop", "Treasure"],
                  originalPath: "/original/path/to/gold_treasure_chest.obj",
                  newPath: "/gaming/assets/props/treasure-chest.obj"
                }
              ]
            }
          ]
        }
      ]
    },
    {
      name: "Music_new",
      count: 3,
      confidence: 97,
      children: [
        {
          name: "Classical_new",
          count: 1,
          confidence: 99,
          children: [
            {
              id: 9,
              name: "beethoven.mp3",
              confidence: 96,
              possibleClassifications: ["Classical", "Beethoven"],
              originalPath: "/original/path/to/beethoven_symphony_no9.mp3",
              newPath: "/music/classical/beethoven.mp3"
            }
          ]
        },
        {
          name: "Jazz_new",
          count: 2,
          confidence: 20,
          children: [
            {
              id: 10,
              name: "miles-davis.mp3",
              confidence: 93,
              possibleClassifications: ["Jazz", "Miles Davis"],
              originalPath: "/original/path/to/miles_davis_kind_of_blue.mp3",
              newPath: "/music/jazz/miles-davis.mp3"
            },
            {
              id: 11,
              name: "coltrane.mp3",
              confidence: 88,
              possibleClassifications: ["Jazz", "John Coltrane"],
              originalPath: "/original/path/to/coltrane_a_love_supreme.mp3",
              newPath: "/music/jazz/coltrane.mp3"
            }
          ]
        }
      ]
    }
  ]
};