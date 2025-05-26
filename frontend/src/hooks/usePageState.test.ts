import { describe, it, expect, beforeEach, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { usePageState } from './usePageState'
import { SORT_FIELD, SORT_ORDER } from '../types/enums'

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  clear: vi.fn(),
}
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
})

describe('usePageState', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorageMock.getItem.mockReturnValue(null)
  })

  it('initializes with default state when localStorage is empty', () => {
    const { result } = renderHook(() => usePageState())

    expect(result.current.pageState).toEqual({
      sortConfig: {
        field: SORT_FIELD.NAME,
        direction: SORT_ORDER.ASC,
      },
      expandedCategories: [],
      selectedItem: null,
    })
  })

  it('loads state from localStorage when available', () => {
    const savedState = {
      sortConfig: {
        field: SORT_FIELD.CONFIDENCE,
        direction: SORT_ORDER.DESC,
      },
      expandedCategories: [1, 2, 3],
      selectedItem: 5,
    }
    localStorageMock.getItem.mockReturnValue(JSON.stringify(savedState))

    const { result } = renderHook(() => usePageState())

    expect(result.current.pageState).toEqual(savedState)
  })

  it('falls back to default state when localStorage contains invalid JSON', () => {
    localStorageMock.getItem.mockReturnValue('invalid json')
    console.error = vi.fn() // Mock console.error

    const { result } = renderHook(() => usePageState())

    expect(result.current.pageState.sortConfig.field).toBe(SORT_FIELD.NAME)
    expect(console.error).toHaveBeenCalled()
  })

  it('updates sort config correctly', () => {
    const { result } = renderHook(() => usePageState())

    act(() => {
      result.current.updateSortConfig({
        field: SORT_FIELD.CONFIDENCE,
        direction: SORT_ORDER.DESC,
      })
    })

    expect(result.current.pageState.sortConfig).toEqual({
      field: SORT_FIELD.CONFIDENCE,
      direction: SORT_ORDER.DESC,
    })
  })

  it('updates expanded categories correctly', () => {
    const { result } = renderHook(() => usePageState())

    act(() => {
      result.current.updateExpandedCategories([1, 2, 3])
    })

    expect(result.current.pageState.expandedCategories).toEqual([1, 2, 3])
  })

  it('updates selected item correctly', () => {
    const { result } = renderHook(() => usePageState())

    act(() => {
      result.current.updateSelectedItem(42)
    })

    expect(result.current.pageState.selectedItem).toBe(42)
  })

  it('resets to default state', () => {
    const { result } = renderHook(() => usePageState())

    // First, modify the state
    act(() => {
      result.current.updateSelectedItem(42)
      result.current.updateExpandedCategories([1, 2, 3])
    })

    // Then reset
    act(() => {
      result.current.resetPageState()
    })

    expect(result.current.pageState).toEqual({
      sortConfig: {
        field: SORT_FIELD.NAME,
        direction: SORT_ORDER.ASC,
      },
      expandedCategories: [],
      selectedItem: null,
    })
  })

  it('saves state to localStorage when state changes', () => {
    const { result } = renderHook(() => usePageState())

    act(() => {
      result.current.updateSelectedItem(42)
    })

    expect(localStorageMock.setItem).toHaveBeenCalledWith(
      'categoryPageState',
      expect.stringContaining('"selectedItem":42')
    )
  })

  it('handles localStorage save errors gracefully', () => {
    localStorageMock.setItem.mockImplementation(() => {
      throw new Error('Storage error')
    })
    console.error = vi.fn()

    const { result } = renderHook(() => usePageState())

    act(() => {
      result.current.updateSelectedItem(42)
    })

    expect(console.error).toHaveBeenCalled()
  })
})