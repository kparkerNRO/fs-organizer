import { describe, it, expect, beforeEach, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { usePersistedCategories } from './usePersistedCategories'
import { Folder } from '../types/types'

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  clear: vi.fn(),
}
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
})

const mockInitialCategories: Folder[] = [
  {
    id: 1,
    name: 'Category 1',
    folder_count: 5,
    confidence: 0.8,
    is_uncertain: false,
  },
  {
    id: 2,
    name: 'Category 2',
    folder_count: 3,
    confidence: 0.9,
    is_uncertain: false,
  },
]

describe('usePersistedCategories', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorageMock.getItem.mockReturnValue(null)
    console.log = vi.fn() // Mock console.log
    console.error = vi.fn() // Mock console.error
  })

  it('initializes with initial categories when localStorage is empty', () => {
    const { result } = renderHook(() => 
      usePersistedCategories(mockInitialCategories)
    )

    expect(result.current.categories).toEqual(mockInitialCategories)
  })

  it('loads categories from localStorage when available', () => {
    const savedCategories: Folder[] = [
      {
        id: 3,
        name: 'Saved Category',
        folder_count: 10,
        confidence: 0.7,
        is_uncertain: true,
      },
    ]
    localStorageMock.getItem.mockReturnValue(JSON.stringify(savedCategories))

    const { result } = renderHook(() => 
      usePersistedCategories(mockInitialCategories)
    )

    expect(result.current.categories).toEqual(savedCategories)
  })

  it('falls back to initial categories when localStorage contains invalid JSON', () => {
    localStorageMock.getItem.mockReturnValue('invalid json')

    const { result } = renderHook(() => 
      usePersistedCategories(mockInitialCategories)
    )

    expect(result.current.categories).toEqual(mockInitialCategories)
    expect(console.error).toHaveBeenCalled()
  })

  it('updates categories correctly', () => {
    const { result } = renderHook(() => 
      usePersistedCategories(mockInitialCategories)
    )

    const newCategories: Folder[] = [
      {
        id: 4,
        name: 'New Category',
        folder_count: 7,
        confidence: 0.85,
        is_uncertain: false,
      },
    ]

    act(() => {
      result.current.setCategories(newCategories)
    })

    expect(result.current.categories).toEqual(newCategories)
  })

  it('saves categories to localStorage when categories change', () => {
    const { result } = renderHook(() => 
      usePersistedCategories(mockInitialCategories)
    )

    const newCategories: Folder[] = [
      {
        id: 4,
        name: 'New Category',
        folder_count: 7,
        confidence: 0.85,
        is_uncertain: false,
      },
    ]

    act(() => {
      result.current.setCategories(newCategories)
    })

    expect(localStorageMock.setItem).toHaveBeenCalledWith(
      'categoriesData',
      JSON.stringify(newCategories)
    )
  })

  it('resets to initial categories', () => {
    const { result } = renderHook(() => 
      usePersistedCategories(mockInitialCategories)
    )

    // First, change categories
    const newCategories: Folder[] = [
      {
        id: 4,
        name: 'New Category',
        folder_count: 7,
        confidence: 0.85,
        is_uncertain: false,
      },
    ]

    act(() => {
      result.current.setCategories(newCategories)
    })

    expect(result.current.categories).toEqual(newCategories)

    // Then reset
    act(() => {
      result.current.resetToInitial()
    })

    expect(result.current.categories).toEqual(mockInitialCategories)
  })

  it('calls callback function when resetting', () => {
    const { result } = renderHook(() => 
      usePersistedCategories(mockInitialCategories)
    )

    const callbackMock = vi.fn()

    act(() => {
      result.current.resetToInitial(callbackMock)
    })

    expect(callbackMock).toHaveBeenCalled()
  })

  it('handles localStorage save errors gracefully', () => {
    localStorageMock.setItem.mockImplementation(() => {
      throw new Error('Storage error')
    })

    const { result } = renderHook(() => 
      usePersistedCategories(mockInitialCategories)
    )

    const newCategories: Folder[] = [
      {
        id: 4,
        name: 'New Category',
        folder_count: 7,
        confidence: 0.85,
        is_uncertain: false,
      },
    ]

    act(() => {
      result.current.setCategories(newCategories)
    })

    expect(console.error).toHaveBeenCalled()
  })
})