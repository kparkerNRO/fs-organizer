/**
 * HierarchyBrowser component for rendering dual representation hierarchies.
 *
 * This component renders either the node or category hierarchy from a DualRepresentation.
 * It provides a tree view with support for highlighting and selection.
 */

import React from "react";
import styled from "styled-components";
import { HierarchyItem, ItemStore, Hierarchy } from "../types/types";

interface HierarchyBrowserProps {
  items: ItemStore;
  hierarchy: Hierarchy;
  rootId: string;
  highlightedItemId?: string | null;
  onItemClick?: (itemId: string) => void;
  onItemHover?: (itemId: string | null) => void;
  showPath?: boolean;
}

export const HierarchyBrowser: React.FC<HierarchyBrowserProps> = ({
  items,
  hierarchy,
  rootId,
  highlightedItemId,
  onItemClick,
  onItemHover,
  showPath = false,
}) => {
  const renderItem = (itemId: string, depth: number = 0): JSX.Element | null => {
    const item = items[itemId];
    if (!item) return null;

    const children = hierarchy[itemId] || [];
    const isHighlighted = highlightedItemId === itemId;
    const isCategory = item.type === 'category';
    const isRoot = itemId === rootId;

    // Don't render the root itself, just its children
    if (isRoot) {
      return (
        <>
          {children.map(childId => renderItem(childId, depth))}
        </>
      );
    }

    return (
      <ItemContainer key={itemId}>
        <ItemRow
          depth={depth}
          isHighlighted={isHighlighted}
          isCategory={isCategory}
          onClick={() => onItemClick?.(itemId)}
          onMouseEnter={() => onItemHover?.(itemId)}
          onMouseLeave={() => onItemHover?.(null)}
        >
          <ItemIcon isCategory={isCategory}>
            {isCategory ? '📁' : '📄'}
          </ItemIcon>
          <ItemName isCategory={isCategory}>
            {item.name}
          </ItemName>
          {showPath && item.originalPath && (
            <ItemPath>{item.originalPath}</ItemPath>
          )}
          {children.length > 0 && (
            <ChildCount>({children.length})</ChildCount>
          )}
        </ItemRow>
        {children.length > 0 && (
          <ChildrenContainer>
            {children.map(childId => renderItem(childId, depth + 1))}
          </ChildrenContainer>
        )}
      </ItemContainer>
    );
  };

  return (
    <BrowserContainer>
      {renderItem(rootId)}
    </BrowserContainer>
  );
};

const BrowserContainer = styled.div`
  width: 100%;
  height: 100%;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 0.5rem;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
`;

const ItemContainer = styled.div`
  width: 100%;
`;

const ItemRow = styled.div<{
  depth: number;
  isHighlighted: boolean;
  isCategory: boolean;
}>`
  display: flex;
  align-items: center;
  padding: 0.375rem 0.5rem;
  padding-left: ${props => props.depth * 1.25 + 0.5}rem;
  cursor: pointer;
  border-radius: 0.25rem;
  transition: all 0.15s ease;
  background-color: ${props => {
    if (props.isHighlighted) return '#e3f2fd';
    return 'transparent';
  }};
  font-weight: ${props => props.isCategory ? 600 : 400};

  &:hover {
    background-color: ${props => props.isHighlighted ? '#bbdefb' : '#f5f5f5'};
  }
`;

const ItemIcon = styled.span<{ isCategory: boolean }>`
  margin-right: 0.5rem;
  font-size: 1rem;
  flex-shrink: 0;
`;

const ItemName = styled.span<{ isCategory: boolean }>`
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.875rem;
  color: ${props => props.isCategory ? '#1a202c' : '#4a5568'};
`;

const ItemPath = styled.span`
  font-size: 0.75rem;
  color: #a0aec0;
  margin-left: 0.5rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 200px;
`;

const ChildCount = styled.span`
  font-size: 0.75rem;
  color: #718096;
  margin-left: 0.5rem;
  flex-shrink: 0;
`;

const ChildrenContainer = styled.div`
  width: 100%;
`;
