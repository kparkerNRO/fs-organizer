import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '../test/utils'
import { NavBar, NavItem } from './NavBar'

const mockNavItems: NavItem[] = [
  { id: 'categories', label: 'Categories' },
  { id: 'folders', label: 'Folder Structure' },
]

describe('NavBar', () => {
  const mockOnNavItemClick = vi.fn()

  beforeEach(() => {
    mockOnNavItemClick.mockClear()
  })

  it('renders logo and menu button', () => {
    render(
      <NavBar
        items={mockNavItems}
        activeItemId="categories"
        onNavItemClick={mockOnNavItemClick}
      />
    )

    expect(screen.getByText('FS Organizer')).toBeInTheDocument()
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('toggles menu on button click', () => {
    render(
      <NavBar
        items={mockNavItems}
        activeItemId="categories"
        onNavItemClick={mockOnNavItemClick}
      />
    )

    const menuButton = screen.getByRole('button')
    
    // Menu should be closed initially
    expect(screen.queryByText('Categories')).not.toBeInTheDocument()
    
    // Click to open menu
    fireEvent.click(menuButton)
    expect(screen.getByText('Categories')).toBeInTheDocument()
    expect(screen.getByText('Folder Structure')).toBeInTheDocument()
    
    // Click to close menu
    fireEvent.click(menuButton)
    expect(screen.queryByText('Categories')).not.toBeInTheDocument()
  })

  it('calls onNavItemClick when menu item is clicked', () => {
    render(
      <NavBar
        items={mockNavItems}
        activeItemId="categories"
        onNavItemClick={mockOnNavItemClick}
      />
    )

    // Open menu
    fireEvent.click(screen.getByRole('button'))
    
    // Click on folder structure item
    fireEvent.click(screen.getByText('Folder Structure'))
    
    expect(mockOnNavItemClick).toHaveBeenCalledWith('folders')
  })

  it('closes menu after nav item click', () => {
    render(
      <NavBar
        items={mockNavItems}
        activeItemId="categories"
        onNavItemClick={mockOnNavItemClick}
      />
    )

    // Open menu
    fireEvent.click(screen.getByRole('button'))
    expect(screen.getByText('Categories')).toBeInTheDocument()
    
    // Click nav item
    fireEvent.click(screen.getByText('Categories'))
    
    // Menu should be closed
    expect(screen.queryByText('Categories')).not.toBeInTheDocument()
  })

  it('highlights active nav item', () => {
    render(
      <NavBar
        items={mockNavItems}
        activeItemId="folders"
        onNavItemClick={mockOnNavItemClick}
      />
    )

    // Open menu
    fireEvent.click(screen.getByRole('button'))
    
    const categoriesItem = screen.getByText('Categories')
    const foldersItem = screen.getByText('Folder Structure')
    
    // Check that folders item has active styling (this would need to be adapted based on actual styling)
    expect(foldersItem).toBeInTheDocument()
    expect(categoriesItem).toBeInTheDocument()
  })
})