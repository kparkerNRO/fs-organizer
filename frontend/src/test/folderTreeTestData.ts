import { FolderV2, File, FolderViewResponse } from '../types/types';

// Test file data
export const mockFile1: File = {
  id: 1,
  name: 'document1.pdf',
  fileType: 'pdf',
  size: '2.5 MB',
};

export const mockFile2: File = {
  id: 2,
  name: 'image.jpg',
  fileType: 'jpg',
  size: '1.2 MB',
};

export const mockFile3: File = {
  id: 3,
  name: 'script.js',
  fileType: 'js',
  size: '15 KB',
};

export const mockFile4: File = {
  id: 4,
  name: 'readme.txt',
  fileType: 'txt',
  size: '512 B',
};

// Test folder data
export const mockSubfolder1: FolderV2 = {
  name: 'images',
  path: '/root/documents/images',
  confidence: 0.95,
  children: [mockFile2],
};

export const mockSubfolder2: FolderV2 = {
  name: 'scripts',
  path: '/root/code/scripts',
  confidence: 0.88,
  children: [mockFile3],
};

export const mockFolder1: FolderV2 = {
  name: 'documents',
  path: '/root/documents',
  confidence: 0.92,
  children: [mockFile1, mockSubfolder1],
};

export const mockFolder2: FolderV2 = {
  name: 'code',
  path: '/root/code',
  confidence: 0.85,
  children: [mockSubfolder2, mockFile4],
};

// Root folder with complex structure
export const mockRootFolder: FolderV2 = {
  name: 'root',
  path: '/root',
  confidence: 1.0,
  children: [mockFolder1, mockFolder2],
};

// Alternative simple tree for basic tests
export const simpleTestTree: FolderV2 = {
  name: 'simple',
  path: '/simple',
  confidence: 1.0,
  children: [
    {
      name: 'folder1',
      path: '/simple/folder1',
      confidence: 0.9,
      children: [
        {
          id: 10,
          name: 'file1.txt',
          fileType: 'txt',
          size: '100 B',
        },
      ],
    },
    {
      id: 11,
      name: 'file2.txt',
      fileType: 'txt',
      size: '200 B',
    },
  ],
};

// Flat structure for testing merge operations
export const flatTestTree: FolderV2 = {
  name: 'flat',
  path: '/flat',
  confidence: 1.0,
  children: [
    {
      name: 'folder_a',
      path: '/flat/folder_a',
      confidence: 0.8,
      children: [
        {
          id: 20,
          name: 'file_a1.txt',
          fileType: 'txt',
          size: '50 B',
        },
      ],
    },
    {
      name: 'folder_b',
      path: '/flat/folder_b',
      confidence: 0.7,
      children: [
        {
          id: 21,
          name: 'file_b1.txt',
          fileType: 'txt',
          size: '60 B',
        },
      ],
    },
    {
      name: 'folder_c',
      path: '/flat/folder_c',
      confidence: 0.9,
      children: [
        {
          id: 22,
          name: 'file_c1.txt',
          fileType: 'txt',
          size: '70 B',
        },
      ],
    },
  ],
};

// Mock FolderViewResponse
export const mockFolderViewResponse: FolderViewResponse = {
  original: mockRootFolder,
  new: {
    name: 'reorganized',
    path: '/reorganized',
    confidence: 1.0,
    children: [
      {
        name: 'all_documents',
        path: '/reorganized/all_documents',
        confidence: 0.95,
        children: [mockFile1, mockFile2],
      },
      {
        name: 'all_code',
        path: '/reorganized/all_code',
        confidence: 0.9,
        children: [mockFile3, mockFile4],
      },
    ],
  },
};

// Test cases for path operations
export const pathTestCases = [
  { parentPath: '', nodeName: 'root', expected: 'root' },
  { parentPath: 'root', nodeName: 'folder1', expected: 'root/folder1' },
  { parentPath: 'root/folder1', nodeName: 'file.txt', expected: 'root/folder1/file.txt' },
];

// Test cases for name validation
export const nameValidationTestCases = [
  { name: 'valid_name', expected: { valid: true } },
  { name: 'valid name with spaces', expected: { valid: true } },
  { name: 'valid.file.txt', expected: { valid: true } },
  { name: '', expected: { valid: false, error: 'Name cannot be empty' } },
  { name: '   ', expected: { valid: false, error: 'Name cannot be empty' } },
  { name: 'invalid/name', expected: { valid: false, error: 'Name cannot contain forward slashes' } },
  { name: '.', expected: { valid: false, error: 'Name cannot be just a dot' } },
  { name: '..valid', expected: { valid: true } },
];

// Expected paths for different operations
export const expectedPaths = {
  allFileIds: [1, 2, 3, 4],
  allFolderPaths: [
    'root',
    'root/documents',
    'root/documents/images',
    'root/code',
    'root/code/scripts',
  ],
  documentsDescendants: [
    'root/documents/images',
  ],
  codeDescendants: [
    'root/code/scripts',
  ],
};