import React from 'react';
import styled from 'styled-components';

interface ContextMenuProps {
  x: number;
  y: number;
  onClose: () => void;
  menu_items: MenuItemProps[];
}

interface MenuItemProps {
    onClick: () => void;
    text: string;
}

export const ContextMenu: React.FC<ContextMenuProps> = ({
  x,
  y,
  onClose,
  menu_items,
}) => {
  return (
    <>
      <Overlay onClick={onClose} />
      <MenuContainer style={{ top: y, left: x }}>
        {menu_items.map((item, index) => (
          <MenuItem key={index} onClick={item.onClick}>
            {item.text}
          </MenuItem>
        ))}
      </MenuContainer>
    </>
  );
};




const Overlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 100;
`;

const MenuContainer = styled.div`
  position: fixed;
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 0.375rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  z-index: 101;
  min-width: 160px;
`;

const MenuItem = styled.div`
  padding: 0.5rem 1rem;
  cursor: pointer;
  
  &:hover {
    background-color: #f3f4f6;
  }
  
  &:first-child {
    border-top-left-radius: 0.375rem;
    border-top-right-radius: 0.375rem;
  }
  
  &:last-child {
    border-bottom-left-radius: 0.375rem;
    border-bottom-right-radius: 0.375rem;
  }
`;