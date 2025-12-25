// src/components/NavBar.tsx
import React, { useState } from "react";
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
  onNavItemClick
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
    <nav className="relative">
      <div className="flex justify-between items-center p-4 px-6 bg-white border-b border-gray-200">
        <button
          onClick={toggleMenu}
          className="bg-transparent border-none cursor-pointer flex items-center justify-center text-gray-600 hover:text-gray-800"
        >
          {menuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
        <div className="text-xl font-semibold text-gray-800">FS Organizer</div>
      </div>

      {menuOpen && (
        <div className="absolute top-full left-0 w-64 bg-white shadow-lg rounded-lg mt-2 ml-4 overflow-hidden z-50">
          {items.map((item) => (
            <div
              key={item.id}
              onClick={() => handleNavItemClick(item.id)}
              className={`py-3 px-6 cursor-pointer ${
                item.id === activeItemId
                  ? "text-blue-600 bg-blue-50 font-semibold"
                  : "text-gray-600 bg-transparent font-normal"
              } hover:bg-gray-100`}
            >
              {item.label}
            </div>
          ))}
        </div>
      )}
    </nav>
  );
};