import { vi } from 'vitest'
import { FetchCategoriesResponse } from '../api'
import { FolderV2, FolderViewResponse } from '../types/types'

// Mock fetch globally
global.fetch = vi.fn()

// Mock API responses
export const mockCategoriesResponse: FetchCategoriesResponse = {
  data: [
    {
      id: 1,
      name: 'Test Category',
      folder_count: 5,
      confidence: 0.95,
      is_uncertain: false,
    },
  ],
  totalItems: 1,
  totalPages: 1,
  currentPage: 1,
}

export const mockFolderStructure: FolderV2 = {
  name: 'Root',
  count: 10,
  confidence: 1.0,
  children: [
    {
      name: 'Child Folder',
      count: 5,
      confidence: 0.8,
      children: [],
    },
  ],
}

export const mockFolderViewResponse: FolderViewResponse = {
  original: mockFolderStructure,
  new: mockFolderStructure,
}

// Helper to mock successful fetch responses
export const mockFetchSuccess = (data: unknown) => {
  (fetch as unknown).mockResolvedValueOnce({
    ok: true,
    json: async () => data,
  })
}

// Helper to mock failed fetch responses
export const mockFetchError = (message: string = 'Network error') => {
  (fetch as unknown).mockRejectedValueOnce(new Error(message))
}

// Reset mocks helper
export const resetMocks = () => {
  vi.clearAllMocks()
}