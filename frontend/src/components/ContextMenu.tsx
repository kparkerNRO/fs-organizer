import React from 'react';

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
      <div
        onClick={onClose}
        className="fixed inset-0 z-[100]"
      />
      <div
        style={{ top: y, left: x }}
        className="fixed bg-white border border-gray-200 rounded-md shadow-lg z-[101] min-w-[160px]"
      >
        {menu_items.map((item, index) => (
          <div
            key={index}
            onClick={item.onClick}
            className={`py-2 px-4 cursor-pointer hover:bg-gray-100 ${
              index === 0 ? 'rounded-t-md' : ''
            } ${index === menu_items.length - 1 ? 'rounded-b-md' : ''}`}
          >
            {item.text}
          </div>
        ))}
      </div>
    </>
  );
};