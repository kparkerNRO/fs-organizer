/**
 * Example component demonstrating dual representation usage.
 *
 * This component shows:
 * - Fetching and displaying dual representation data
 * - Rendering both node and category hierarchies
 * - Synchronized highlighting between views
 * - Making and applying changes
 */

import React from "react";
import { useDualRepresentation } from "../hooks/useDualRepresentation";

export const DualRepresentationExample: React.FC = () => {
  const {
    dualRep,
    isLoading,
    error,
    pendingDiff,
    hasPendingChanges,
    highlightedItemId,
    selectedView,
    fetchDualRepresentation,
    applyPendingChanges,
    clearPendingChanges,
    highlightItem,
    setView,
    getItem,
    getChildren,
  } = useDualRepresentation();

  // Render a single item in the hierarchy
  const renderItem = (itemId: string, hierarchy: 'node' | 'category', depth: number = 0) => {
    const item = getItem(itemId);
    if (!item) return null;

    const children = getChildren(itemId, hierarchy);
    const isHighlighted = highlightedItemId === itemId;

    return (
      <div
        key={itemId}
        style={{
          marginLeft: `${depth * 20}px`,
          padding: '4px 8px',
          backgroundColor: isHighlighted ? '#e3f2fd' : 'transparent',
          cursor: 'pointer',
          borderRadius: '4px',
        }}
        onClick={() => highlightItem(itemId)}
        onMouseEnter={() => highlightItem(itemId)}
        onMouseLeave={() => highlightItem(null)}
      >
        <div style={{ fontWeight: item.type === 'category' ? 'bold' : 'normal' }}>
          {item.type === 'node' ? '📄' : '📁'} {item.name}
          {item.originalPath && (
            <span style={{ fontSize: '0.8em', color: '#666', marginLeft: '8px' }}>
              {item.originalPath}
            </span>
          )}
        </div>
        {children.length > 0 && (
          <div>
            {children.map(childId => renderItem(childId, hierarchy, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  // Render a hierarchy (node or category)
  const renderHierarchy = (hierarchy: 'node' | 'category') => {
    if (!dualRep) return null;

    const rootId = hierarchy === 'node' ? 'node-root' : 'category-root';
    const rootChildren = getChildren(rootId, hierarchy);

    return (
      <div style={{ flex: 1, padding: '16px', border: '1px solid #ddd', borderRadius: '8px' }}>
        <h3>{hierarchy === 'node' ? 'File System Structure' : 'Category Structure'}</h3>
        <div>
          {rootChildren.map(childId => renderItem(childId, hierarchy))}
        </div>
      </div>
    );
  };

  // Render pending changes
  const renderPendingChanges = () => {
    if (!hasPendingChanges) return null;

    return (
      <div style={{
        padding: '16px',
        backgroundColor: '#fff3e0',
        borderRadius: '8px',
        marginTop: '16px',
      }}>
        <h4>Pending Changes</h4>
        <div>
          {Object.keys(pendingDiff.added).length > 0 && (
            <div>
              <strong>Added:</strong>
              <pre>{JSON.stringify(pendingDiff.added, null, 2)}</pre>
            </div>
          )}
          {Object.keys(pendingDiff.deleted).length > 0 && (
            <div>
              <strong>Deleted:</strong>
              <pre>{JSON.stringify(pendingDiff.deleted, null, 2)}</pre>
            </div>
          )}
        </div>
        <div style={{ marginTop: '8px', display: 'flex', gap: '8px' }}>
          <button
            onClick={applyPendingChanges}
            style={{
              padding: '8px 16px',
              backgroundColor: '#4caf50',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Apply Changes
          </button>
          <button
            onClick={clearPendingChanges}
            style={{
              padding: '8px 16px',
              backgroundColor: '#f44336',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Clear Changes
          </button>
        </div>
      </div>
    );
  };

  // Loading state
  if (isLoading && !dualRep) {
    return (
      <div style={{ padding: '16px' }}>
        <p>Loading dual representation...</p>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div style={{ padding: '16px', color: 'red' }}>
        <p>Error: {error.message}</p>
        <button
          onClick={fetchDualRepresentation}
          style={{
            padding: '8px 16px',
            marginTop: '8px',
            cursor: 'pointer',
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  // No data state
  if (!dualRep) {
    return (
      <div style={{ padding: '16px' }}>
        <p>No data available. Please run gather and group first.</p>
      </div>
    );
  }

  // Main render
  return (
    <div style={{ padding: '16px' }}>
      <div style={{ marginBottom: '16px' }}>
        <h2>Dual Representation Viewer</h2>
        <p style={{ color: '#666' }}>
          Hover over items to see synchronized highlighting across both views.
        </p>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <button
          onClick={() => setView('node')}
          style={{
            padding: '8px 16px',
            marginRight: '8px',
            backgroundColor: selectedView === 'node' ? '#2196f3' : '#e0e0e0',
            color: selectedView === 'node' ? 'white' : 'black',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Node View
        </button>
        <button
          onClick={() => setView('category')}
          style={{
            padding: '8px 16px',
            backgroundColor: selectedView === 'category' ? '#2196f3' : '#e0e0e0',
            color: selectedView === 'category' ? 'white' : 'black',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Category View
        </button>
      </div>

      <div style={{ display: 'flex', gap: '16px' }}>
        {renderHierarchy('node')}
        {renderHierarchy('category')}
      </div>

      {renderPendingChanges()}

      <div style={{ marginTop: '16px', padding: '16px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
        <h4>Statistics</h4>
        <p>Total Items: {Object.keys(dualRep.items).length}</p>
        <p>
          Nodes: {Object.values(dualRep.items).filter(item => item.type === 'node').length}
        </p>
        <p>
          Categories: {Object.values(dualRep.items).filter(item => item.type === 'category').length}
        </p>
        {highlightedItemId && (
          <p>Highlighted: {getItem(highlightedItemId)?.name}</p>
        )}
      </div>
    </div>
  );
};

export default DualRepresentationExample;
