// src/components/NavBar.tsx
import React, { useState } from "react";
import styled from "styled-components";
import { Menu, X } from "lucide-react";

export interface NavItem {
  id: string;
  label: string;
}

interface NavBarProps {
  items: NavItem[];
  activeItemId: string;
  onNavItemClick: (itemId: string) => void;
}

export const NavBar: React.FC<NavBarProps> = ({
  items,
  activeItemId,
  onNavItemClick,
}) => {
  const [menuOpen, setMenuOpen] = useState(false);

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  const handleNavItemClick = (itemId: string) => {
    onNavItemClick(itemId);
    setMenuOpen(false);
  };

  return (
    <NavContainer>
      <NavHeader>
        <MenuButton onClick={toggleMenu}>
          {menuOpen ? <X size={24} /> : <Menu size={24} />}
        </MenuButton>
        <Logo>FS Organizer</Logo>
      </NavHeader>

      {menuOpen && (
        <NavMenu>
          {items.map((item) => (
            <NavMenuItem
              key={item.id}
              $active={item.id === activeItemId}
              onClick={() => handleNavItemClick(item.id)}
            >
              {item.label}
            </NavMenuItem>
          ))}
        </NavMenu>
      )}
    </NavContainer>
  );
};

const NavContainer = styled.nav`
  position: relative;
`;

const NavHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.5rem;
  background-color: #ffffff;
  border-bottom: 1px solid #e5e7eb;
`;

const Logo = styled.div`
  font-size: 1.25rem;
  font-weight: 600;
  color: #1f2937;
`;

const MenuButton = styled.button`
  background: none;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #4b5563;

  &:hover {
    color: #1f2937;
  }
`;

const NavMenu = styled.div`
  position: absolute;
  top: 100%;
  left: 0;
  width: 250px;
  background-color: #ffffff;
  box-shadow:
    0 4px 6px -1px rgba(0, 0, 0, 0.1),
    0 2px 4px -1px rgba(0, 0, 0, 0.06);
  border-radius: 0.5rem;
  margin-top: 0.5rem;
  margin-left: 1rem;
  overflow: hidden;
  z-index: 50;
`;

const NavMenuItem = styled.div<{ $active: boolean }>`
  padding: 0.75rem 1.5rem;
  cursor: pointer;
  color: ${(props) => (props.$active ? "#2563eb" : "#4b5563")};
  background-color: ${(props) => (props.$active ? "#eff6ff" : "transparent")};
  font-weight: ${(props) => (props.$active ? "600" : "400")};

  &:hover {
    background-color: #f3f4f6;
  }
`;
