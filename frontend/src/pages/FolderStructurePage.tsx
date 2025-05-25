// src/pages/FolderStructurePage.tsx
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { fetchFolderStructure, FolderNode, FileNode } from "../api";
import { ChevronDown, ChevronRight, FolderOpen, Folder as FolderIcon, File as FileIcon, Move } from "lucide-react";
import { CategoryDetails } from "../components/CategoryDetails";
import { CategoryDetailsProps, Folder, FileItem } from "../types/types";

export const FolderStructurePage: React.FC = () => {
  const [folderStructure, setFolderStructure] = useState<FolderNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(["root"]));
  const [selectedItem, setSelectedItem] = useState<CategoryDetailsProps>({category: null, folder: null, file: null});
  
  // Track selected items for multi-select
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set());
  const [draggedNodes, setDraggedNodes] = useState<(FolderNode | FileNode)[]>([]);
  const [draggedOverId, setDraggedOverId] = useState<string | null>(null);

  useEffect(() => {
    const loadFolderStructure = async () => {
      try {
        setLoading(true);
        const data = await fetchFolderStructure();
        setFolderStructure(data);
      } catch (error) {
        console.error("Error loading folder structure:", error);
      } finally {
        setLoading(false);
      }
    };

    loadFolderStructure();
  }, []);
  
  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Escape key: clear selection
      if (e.key === 'Escape') {
        setSelectedNodes(new Set());
      }
      
      // Delete key: remove selected items
      if (e.key === 'Delete' && selectedNodes.size > 0 && folderStructure) {
        // Create a deep copy of the structure
        const updatedStructure = JSON.parse(JSON.stringify(folderStructure));
        
        // For each selected node, find its parent and remove it
        let hasRemovedItems = false;
        
        selectedNodes.forEach(nodeId => {
          const parentNode = findParentNode(nodeId, updatedStructure);
          
          if (parentNode && parentNode.children) {
            parentNode.children = parentNode.children.filter(child => child.id !== nodeId);
            hasRemovedItems = true;
          }
        });
        
        if (hasRemovedItems) {
          setFolderStructure(updatedStructure);
          setSelectedNodes(new Set());
        }
      }
      
      // Ctrl+A: select all visible items in the current view
      if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        e.preventDefault();
        
        // Find all visible nodes (only expanded folders and their children)
        const visibleNodeIds = new Set<string>();
        
        const collectVisibleNodeIds = (node: FolderNode | FileNode, isVisible: boolean = true) => {
          if (isVisible) {
            visibleNodeIds.add(node.id);
          }
          
          // For folder nodes with children that are expanded
          if (!isFileNode(node) && node.children && (isVisible && (node.id === 'root' || expandedFolders.has(node.id)))) {
            node.children.forEach(child => {
              collectVisibleNodeIds(child, true);
            });
          }
        };
        
        if (folderStructure) {
          collectVisibleNodeIds(folderStructure);
        }
        
        setSelectedNodes(visibleNodeIds);
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedNodes, folderStructure, expandedFolders]);

  const toggleFolder = (folderId: string) => {
    setExpandedFolders(prev => {
      const newSet = new Set(prev);
      if (newSet.has(folderId)) {
        newSet.delete(folderId);
      } else {
        newSet.add(folderId);
      }
      return newSet;
    });
  };

  // Helper functions
  const isFileNode = (node: FolderNode | FileNode): node is FileNode => {
    return 'fileType' in node || node.path?.includes('.') || false;
  };
  
  const findNodeById = (id: string, rootNode: FolderNode): (FolderNode | FileNode | null) => {
    if (rootNode.id === id) return rootNode;
    
    if (!rootNode.children) return null;
    
    for (const child of rootNode.children) {
      if (child.id === id) return child;
      
      if (!isFileNode(child) && child.children) {
        const found = findNodeById(id, child);
        if (found) return found;
      }
    }
    
    return null;
  };
  
  const findParentNode = (childId: string, rootNode: FolderNode): FolderNode | null => {
    if (!rootNode.children) return null;
    
    for (const child of rootNode.children) {
      if (child.id === childId) return rootNode;
      
      if (!isFileNode(child) && child.children) {
        const parent = findParentNode(childId, child);
        if (parent) return parent;
      }
    }
    
    return null;
  };

  // Node selection functions
  const handleNodeSelection = (node: FolderNode | FileNode, e: React.MouseEvent) => {
    // Handle multi-select with Ctrl/Cmd key
    if (e.ctrlKey || e.metaKey) {
      e.stopPropagation();
      setSelectedNodes(prev => {
        const newSelection = new Set(prev);
        
        if (newSelection.has(node.id)) {
          newSelection.delete(node.id);
        } else {
          newSelection.add(node.id);
        }
        
        return newSelection;
      });
    } else {
      // Clear selection if clicking without Ctrl/Cmd
      if (!selectedNodes.has(node.id)) {
        setSelectedNodes(new Set([node.id]));
      }
      
      // Show details for the selected item
      selectItem(node);
    }
  };
  
  // Drag and drop handlers
  const handleDragStart = (e: React.DragEvent, node: FolderNode | FileNode) => {
    e.stopPropagation();
    
    // If the dragged node is not in the selection, make it the only selected node
    if (!selectedNodes.has(node.id)) {
      setSelectedNodes(new Set([node.id]));
    }
    
    // Collect all selected nodes for the drag operation
    const nodesToDrag: (FolderNode | FileNode)[] = [];
    
    if (folderStructure) {
      selectedNodes.forEach(id => {
        const selectedNode = findNodeById(id, folderStructure);
        if (selectedNode) {
          nodesToDrag.push(selectedNode);
        }
      });
    }
    
    setDraggedNodes(nodesToDrag);
    
    // Add visual feedback
    if (e.dataTransfer) {
      e.dataTransfer.setData('text/plain', JSON.stringify({
        nodeIds: Array.from(selectedNodes)
      }));
      e.dataTransfer.effectAllowed = 'move';
    }
  };
  
  const handleDragOver = (e: React.DragEvent, node: FolderNode | FileNode) => {
    e.preventDefault();
    e.stopPropagation();
    
    // Only allow dropping onto folders, not files
    if (!isFileNode(node)) {
      e.dataTransfer.dropEffect = 'move';
      setDraggedOverId(node.id);
    } else {
      e.dataTransfer.dropEffect = 'none';
    }
  };
  
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDraggedOverId(null);
  };
  
  const handleDrop = (e: React.DragEvent, targetNode: FolderNode | FileNode) => {
    e.preventDefault();
    e.stopPropagation();
    setDraggedOverId(null);
    
    // Only allow dropping onto folders
    if (isFileNode(targetNode) || !folderStructure) return;
    
    try {
      // Make a deep copy of the folder structure to work with
      const updatedStructure = JSON.parse(JSON.stringify(folderStructure));
      
      // Find the target folder in our copy
      const targetFolder = findNodeById(targetNode.id, updatedStructure) as FolderNode;
      if (!targetFolder || isFileNode(targetFolder)) return;
      
      // Ensure the target folder has a children array
      if (!targetFolder.children) {
        targetFolder.children = [];
      }
      
      // For each dragged node
      draggedNodes.forEach(draggedNode => {
        // Find the parent of the dragged node
        const parentNode = findParentNode(draggedNode.id, updatedStructure);
        if (!parentNode) return;
        
        // Skip if trying to drop onto itself or its parent
        if (draggedNode.id === targetFolder.id || parentNode.id === targetFolder.id) return;
        
        // Find the dragged node in our structure copy
        const nodeToDrag = findNodeById(draggedNode.id, updatedStructure);
        if (!nodeToDrag) return;
        
        // Check for name conflicts with existing items in the target folder
        const isFolder = !isFileNode(nodeToDrag);
        const existingItem = targetFolder.children.find(child => child.name === nodeToDrag.name);
        
        if (existingItem) {
          // Case 1: Folder with same name exists - merge contents
          if (isFolder && !isFileNode(existingItem)) {
            // Merge folder contents
            const nodeToMerge = nodeToDrag as FolderNode;
            if (nodeToMerge.children && nodeToMerge.children.length > 0) {
              if (!existingItem.children) {
                existingItem.children = [];
              }
              
              // Recursively add children, handling any further conflicts
              nodeToMerge.children.forEach(child => {
                const childConflict = existingItem.children!.find(existing => existing.name === child.name);
                
                if (childConflict) {
                  // Handle recursive conflicts
                  if (!isFileNode(child) && !isFileNode(childConflict)) {
                    // Merge sub-folders
                    if (child.children) {
                      if (!childConflict.children) childConflict.children = [];
                      child.children.forEach(subChild => {
                        childConflict.children!.push(subChild);
                      });
                    }
                  } else if (isFileNode(child)) {
                    // Rename conflicting file
                    const baseName = child.name.split('.')[0] || 'file';
                    const extension = child.name.includes('.') ? `.${child.name.split('.').pop()}` : '';
                    let newName = '';
                    let counter = 1;
                    
                    // Find a non-conflicting name
                    do {
                      newName = `${baseName}_${counter}${extension}`;
                      counter++;
                    } while (existingItem.children!.some(c => c.name === newName));
                    
                    // Create a copy with the new name
                    const renamedChild = { ...child, name: newName, id: `${child.id}_copy${counter}` };
                    existingItem.children!.push(renamedChild);
                  }
                } else {
                  // No conflict, add directly
                  existingItem.children!.push(child);
                }
              });
            }
          } 
          // Case 2: File with same name exists - rename the file being moved
          else if (isFileNode(nodeToDrag)) {
            const baseName = nodeToDrag.name.split('.')[0] || 'file';
            const extension = nodeToDrag.name.includes('.') ? `.${nodeToDrag.name.split('.').pop()}` : '';
            let newName = '';
            let counter = 1;
            
            // Find a non-conflicting name
            do {
              newName = `${baseName}_${counter}${extension}`;
              counter++;
            } while (targetFolder.children.some(child => child.name === newName));
            
            // Create a copy with the new name
            const renamedNode = { ...nodeToDrag, name: newName, id: `${nodeToDrag.id}_copy${counter}` };
            targetFolder.children.push(renamedNode);
          }
        } else {
          // No conflict, add directly
          targetFolder.children.push(nodeToDrag);
        }
        
        // Remove the dragged node from its original parent
        if (parentNode.children) {
          parentNode.children = parentNode.children.filter(child => child.id !== draggedNode.id);
        }
      });
      
      // Update the folder structure
      setFolderStructure(updatedStructure);
      
      // Clear selection after drop
      setSelectedNodes(new Set());
      setDraggedNodes([]);
    } catch (error) {
      console.error('Error during drag and drop operation:', error);
    }
  };
  
  const selectItem = (node: FolderNode | FileNode) => {
    const nodeId = parseInt(node.id, 10) || Math.floor(Math.random() * 1000);
    
    if (isFileNode(node)) {
      // Create file item
      const fileData: FileItem = {
        id: nodeId,
        name: node.name,
        fileType: node.fileType || node.name.split('.').pop() || 'unknown',
        size: node.size,
        original_path: node.originalPath || node.path || '',
        confidence: node.confidence || 90,
        categories: node.categories
      };
      
      setSelectedItem({category: null, folder: null, file: fileData});
    } else {
      // Create folder item
      const folderData: Folder = {
        id: nodeId,
        name: node.name,
        classification: node.name.split('/').pop() || '',
        original_filename: node.name,
        cleaned_name: node.name,
        confidence: 90,
        original_path: node.path || '',
        processed_names: node.name.split('/')
      };
      
      setSelectedItem({category: null, folder: folderData, file: null});
    }
  };

  const renderNode = (node: FolderNode | FileNode, level: number = 0) => {
    const isFile = isFileNode(node);
    const isExpanded = !isFile && expandedFolders.has(node.id);
    const hasChildren = !isFile && (node as FolderNode).children && (node as FolderNode).children!.length > 0;
    const isSelected = selectedNodes.has(node.id);
    const isDraggedOver = draggedOverId === node.id;

    return (
      <div key={node.id}>
        <FolderItem 
          $level={level}
          $isFile={isFile}
          $isSelected={isSelected}
          $isDraggedOver={isDraggedOver && !isFile}
          onClick={(e) => {
            handleNodeSelection(node, e);
            if (!isFile && hasChildren && !e.ctrlKey && !e.metaKey) toggleFolder(node.id);
          }}
          draggable
          onDragStart={(e) => handleDragStart(e, node)}
          onDragOver={(e) => !isFile && handleDragOver(e, node)}
          onDragLeave={handleDragLeave}
          onDrop={(e) => !isFile && handleDrop(e, node)}
        >
          {!isFile && hasChildren ? (
            <ExpandIcon>
              {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </ExpandIcon>
          ) : (
            <ExpandIcon style={{ opacity: 0.7 }}>
              {isFile ? <FileIcon size={14} /> : <FolderIcon size={14} />}
            </ExpandIcon>
          )}
          <FolderName>{node.name}</FolderName>
          {isFile && (node as FileNode).fileType && (
            <FileType>{(node as FileNode).fileType}</FileType>
          )}
          {isFile && (node as FileNode).size && (
            <FileSize>{(node as FileNode).size}</FileSize>
          )}
          {!isFile && node.path && node.id !== 'root' && <FolderPath>{node.path}</FolderPath>}
          {isSelected && <SelectedIndicator />}
        </FolderItem>
        
        {!isFile && isExpanded && hasChildren && (
          <div>
            {(node as FolderNode).children?.map(child => renderNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <PageContainer>
      <Header>
        <Title>Folder Structure</Title>
      </Header>

      <MainContainer>
        <ContentContainer>
          <div style={{ paddingBottom: '0.5rem', borderBottom: '1px solid #e5e7eb', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <FolderIcon size={16} style={{ marginRight: '0.5rem', opacity: 0.7 }} />
              <span style={{ fontWeight: 500 }}>Folders</span>
            </div>
            <InstructionsButton onClick={(e) => {
              e.stopPropagation();
              alert('Instructions:\n\n• Click to select an item\n• Ctrl/Cmd+Click to select multiple items\n• Drag and drop to move items\n• Esc to clear selection\n• Delete to remove selected items\n• Ctrl/Cmd+A to select all visible items');
            }}>
              ?</InstructionsButton>
          </div>
          {loading ? (
            <LoadingMessage>Loading folder structure...</LoadingMessage>
          ) : folderStructure ? (
            <>
              <FolderTree>
                {renderNode(folderStructure)}
              </FolderTree>
              
              {selectedNodes.size > 0 && (
                <StatusBar>
                  <div>
                    {selectedNodes.size} item{selectedNodes.size !== 1 ? 's' : ''} selected
                  </div>
                  <StatusBarActions>
                    <StatusBarButton 
                      title="Delete selected items"
                      onClick={() => {
                        if (folderStructure) {
                          const updatedStructure = JSON.parse(JSON.stringify(folderStructure));
                          let hasRemovedItems = false;
                          
                          selectedNodes.forEach(nodeId => {
                            const parentNode = findParentNode(nodeId, updatedStructure);
                            
                            if (parentNode && parentNode.children) {
                              parentNode.children = parentNode.children.filter(child => child.id !== nodeId);
                              hasRemovedItems = true;
                            }
                          });
                          
                          if (hasRemovedItems) {
                            setFolderStructure(updatedStructure);
                            setSelectedNodes(new Set());
                          }
                        }
                      }}
                    >
                      Delete
                    </StatusBarButton>
                    <StatusBarButton
                      title="Clear selection"
                      onClick={() => setSelectedNodes(new Set())}
                    >
                      Clear
                    </StatusBarButton>
                  </StatusBarActions>
                </StatusBar>
              )}
            </>
          ) : (
            <ErrorMessage>Failed to load folder structure</ErrorMessage>
          )}
        </ContentContainer>
        
        <div className="folder-structure-page" style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden', width: '450px' }}>
          <CategoryDetails 
            category={selectedItem.category} 
            folder={selectedItem.folder}
            file={selectedItem.file} 
          />
        </div>
      </MainContainer>
    </PageContainer>
  );
};

const PageContainer = styled.div`
  background-color: #f3f4f6;
  padding: 2rem;
  height: calc(100vh - 57px); /* Account for navbar height */
  overflow: hidden;
  display: flex;
  flex-direction: column;
`;

const MainContainer = styled.div`
  display: flex;
  flex-direction: row;
  gap: 2rem;
  max-width: 1200px;
  margin: 0 auto;
  align-items: stretch;
  flex: 1;
  min-height: 0; /* Critical for proper flexbox behavior with scrolling */
  overflow: hidden;
  justify-content: space-between;
`;

const ContentContainer = styled.div`
  width: 500px;
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  padding: 1rem;
  box-sizing: border-box;
  overflow: hidden;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  min-height: 0; /* Critical for proper flexbox behavior with scrolling */
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
`;

const Title = styled.h1`
  font-size: 2rem;
  font-weight: 600;
  color: #1f2937;
`;

const FolderTree = styled.div`
  margin-top: 0.5rem;
  flex: 1;
  overflow-y: auto;
  min-height: 0; /* Critical for proper flexbox behavior with scrolling */
  
  /* Modern scrollbar styling */
  &::-webkit-scrollbar {
    width: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
  }
`;

const FolderItem = styled.div<{ 
  $level: number, 
  $isFile?: boolean, 
  $isSelected?: boolean,
  $isDraggedOver?: boolean
}>`
  display: flex;
  align-items: center;
  padding: 0.35rem 0.5rem;
  padding-left: ${(props) => props.$level * 0.9 + 0.6}rem;
  cursor: pointer;
  border-radius: 0.15rem;
  transition: all 0.15s ease;
  margin: 1px 0;
  font-size: 0.9rem;
  color: ${(props) => props.$isFile ? '#4b5563' : '#000'};
  background-color: ${(props) => {
    if (props.$isDraggedOver) return '#e2f0fd';
    if (props.$isSelected) return '#dbeafe';
    return 'transparent';
  }};
  border: ${(props) => props.$isDraggedOver ? '1px dashed #3b82f6' : props.$isSelected ? '1px solid #60a5fa' : 'none'};
  position: relative;

  &:hover {
    background-color: ${(props) => props.$isSelected ? '#c7d2fe' : '#e7eefa'};
  }
  
  &:active {
    background-color: #dce6f8;
  }
`;

const ExpandIcon = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-right: 0.3rem;
  width: 16px;
  height: 16px;
`;

const FolderName = styled.span`
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
`;

const FolderPath = styled.span`
  margin-left: 0.5rem;
  color: #6b7280;
  font-size: 0.75rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 180px;
`;

const FileType = styled.span`
  margin-left: 0.5rem;
  color: #6b7280;
  font-size: 0.75rem;
  background-color: #f3f4f6;
  padding: 0.1rem 0.4rem;
  border-radius: 0.25rem;
`;

const FileSize = styled.span`
  margin-left: 0.5rem;
  color: #6b7280;
  font-size: 0.75rem;
`;

const SelectedIndicator = styled.div`
  position: absolute;
  right: 10px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #3b82f6;
`;

const LoadingMessage = styled.div`
  text-align: center;
  padding: 2rem;
  color: #6b7280;
`;

const StatusBar = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 1rem;
  background-color: #f3f4f6;
  border-top: 1px solid #e5e7eb;
  border-radius: 0 0 0.5rem 0.5rem;
  margin-top: auto;
  font-size: 0.85rem;
`;

const StatusBarActions = styled.div`
  display: flex;
  gap: 0.5rem;
`;

const StatusBarButton = styled.button`
  padding: 0.25rem 0.5rem;
  background-color: #e5e7eb;
  border: none;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.15s ease;
  
  &:hover {
    background-color: #d1d5db;
  }
`;

const InstructionsButton = styled.button`
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background-color: #e5e7eb;
  border: none;
  font-size: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.15s ease;
  
  &:hover {
    background-color: #d1d5db;
  }
`;

const ErrorMessage = styled.div`
  text-align: center;
  padding: 2rem;
  color: #ef4444;
`;