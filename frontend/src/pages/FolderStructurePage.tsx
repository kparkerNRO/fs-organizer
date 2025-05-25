// src/pages/FolderStructurePage.tsx
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { fetchFolderStructureComparison, FolderNode, FileNode, FolderStructureComparison } from "../api";
import { ChevronDown, ChevronRight, FolderOpen, Folder as FolderIcon, File as FileIcon, Move } from "lucide-react";

export const FolderStructurePage: React.FC = () => {
  const [folderComparison, setFolderComparison] = useState<FolderStructureComparison | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedFoldersOriginal, setExpandedFoldersOriginal] = useState<Set<string>>(new Set(["root"]));
  const [expandedFoldersNew, setExpandedFoldersNew] = useState<Set<string>>(new Set(["root"]));
  // const [selectedItem, setSelectedItem] = useState<{category: any, folder: Folder | null, file: FileItem | null}>({category: null, folder: null, file: null});
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  
  const [draggedNodes, setDraggedNodes] = useState<(FolderNode | FileNode)[]>([]);
  const [draggedOverId, setDraggedOverId] = useState<string | null>(null);

  useEffect(() => {
    const loadFolderStructure = async () => {
      try {
        setLoading(true);
        const data = await fetchFolderStructureComparison();
        setFolderComparison(data);
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
      // Escape key: clear file selection
      if (e.key === 'Escape') {
        setSelectedFileId(null);
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const toggleFolder = (folderId: string, isOriginal: boolean) => {
    const setExpandedFolders = isOriginal ? setExpandedFoldersOriginal : setExpandedFoldersNew;
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

  
  // Drag and drop handlers
  const handleDragStart = (e: React.DragEvent, node: FolderNode | FileNode) => {
    e.stopPropagation();
    
    setDraggedNodes([node]);
    
    // Add visual feedback
    if (e.dataTransfer) {
      e.dataTransfer.setData('text/plain', JSON.stringify({
        nodeIds: [node.id]
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
    if (isFileNode(targetNode) || !folderComparison) return;
    
    try {
      // Make a deep copy of the folder structure to work with
      const updatedComparison = JSON.parse(JSON.stringify(folderComparison));
      
      // Find the target folder in our copy
      const targetFolderOriginal = findNodeById(targetNode.id, updatedComparison.original) as FolderNode;
      const targetFolderNew = findNodeById(targetNode.id, updatedComparison.new) as FolderNode;
      const targetFolder = (targetFolderOriginal || targetFolderNew) as FolderNode;
      if (!targetFolder || isFileNode(targetFolder)) return;
      
      // Ensure the target folder has a children array
      if (!targetFolder.children) {
        targetFolder.children = [];
      }
      
      // For each dragged node
      draggedNodes.forEach(draggedNode => {
        // Find the parent of the dragged node
        const parentNodeOriginal = findParentNode(draggedNode.id, updatedComparison.original);
        const parentNodeNew = findParentNode(draggedNode.id, updatedComparison.new);
        const parentNode = parentNodeOriginal || parentNodeNew;
        if (!parentNode) return;
        
        // Skip if trying to drop onto itself or its parent
        if (draggedNode.id === targetFolder.id || parentNode.id === targetFolder.id) return;
        
        // Find the dragged node in our structure copy
        const nodeToDragOriginal = findNodeById(draggedNode.id, updatedComparison.original);
        const nodeToDragNew = findNodeById(draggedNode.id, updatedComparison.new);
        const nodeToDrag = nodeToDragOriginal || nodeToDragNew;
        if (!nodeToDrag) return;
        
        // Check for name conflicts with existing items in the target folder
        const isFolder = !isFileNode(nodeToDrag);
        const existingItem = targetFolder.children!.find((child: FolderNode | FileNode) => child.name === nodeToDrag.name);
        
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
              nodeToMerge.children.forEach((child: FolderNode | FileNode) => {
                const childConflict = existingItem.children!.find((existing: FolderNode | FileNode) => existing.name === child.name);
                
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
                    } while (existingItem.children!.some((c: FolderNode | FileNode) => c.name === newName));
                    
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
            } while (targetFolder.children!.some((child: FolderNode | FileNode) => child.name === newName));
            
            // Create a copy with the new name
            const renamedNode = { ...nodeToDrag, name: newName, id: `${nodeToDrag.id}_copy${counter}` };
            if (!targetFolder.children) {
              targetFolder.children = [];
            }
            targetFolder.children.push(renamedNode);
          }
        } else {
          // No conflict, add directly
          if (!targetFolder.children) {
              targetFolder.children = [];
            }
          targetFolder.children.push(nodeToDrag);
        }
        
        // Remove the dragged node from its original parent
        if (parentNode.children) {
          parentNode.children = parentNode.children.filter(child => child.id !== draggedNode.id);
        }
      });
      
      // Update the folder structure
      setFolderComparison(updatedComparison);
      
      // Clear drag state after drop
      setDraggedNodes([]);
    } catch (error) {
      console.error('Error during drag and drop operation:', error);
    }
  };
  
  const selectItem = (node: FolderNode | FileNode) => {
    // No longer needed since CategoryDetails component was removed
    console.log('Selected item:', node.name);
  };

  // Find a file by ID and return the path of folder IDs leading to it
  const findFileById = (fileId: string, rootNode: FolderNode, currentPath: string[] = []): { node: FileNode, path: string[] } | null => {
    if (!rootNode.children) return null;
    
    for (const child of rootNode.children) {
      if (isFileNode(child) && child.id === fileId) {
        return { node: child, path: currentPath };
      }
      
      if (!isFileNode(child) && child.children) {
        const result = findFileById(fileId, child, [...currentPath, child.id]);
        if (result) return result;
      }
    }
    
    return null;
  };

  // Expand all folders in a given path, including root
  const expandPathInView = (path: string[], isOriginal: boolean) => {
    const setExpandedFolders = isOriginal ? setExpandedFoldersOriginal : setExpandedFoldersNew;
    setExpandedFolders(prev => {
      const newSet = new Set(prev);
      // Always ensure root is expanded
      newSet.add('root');
      // Add all folders in the path
      path.forEach(folderId => newSet.add(folderId));
      return newSet;
    });
  };

  const handleFileClick = (fileNode: FileNode, isFromOriginal: boolean) => {
    if (isFileNode(fileNode)) {
      setSelectedFileId(fileNode.id);
      selectItem(fileNode);
      
      // Find the corresponding file by ID in the other view and expand path to it
      if (folderComparison) {
        const otherRoot = isFromOriginal ? folderComparison.new : folderComparison.original;
        const matchingFile = findFileById(fileNode.id, otherRoot);
        
        if (matchingFile) {
          // Expand all folders in the path to the matching file in the other view
          expandPathInView(matchingFile.path, !isFromOriginal);
        }
      }
    }
  };

  const renderNode = (node: FolderNode | FileNode, level: number = 0, isOriginal: boolean = true, expandedFolders: Set<string>) => {
    const isFile = isFileNode(node);
    const isExpanded = !isFile && expandedFolders.has(node.id);
    const hasChildren = !isFile && (node as FolderNode).children && (node as FolderNode).children!.length > 0;
    const isDraggedOver = draggedOverId === node.id;
    const isHighlighted = isFile && selectedFileId === node.id;

    return (
      <div key={node.id}>
        <FolderItem 
          $level={level}
          $isFile={isFile}
$isSelected={isHighlighted}
          $isDraggedOver={isDraggedOver && !isFile}
          onClick={() => {
            if (isFile) {
              handleFileClick(node as FileNode, isOriginal);
            } else {
              if (hasChildren) toggleFolder(node.id, isOriginal);
            }
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
        </FolderItem>
        
        {!isFile && isExpanded && hasChildren && (
          <div>
            {(node as FolderNode).children?.map(child => renderNode(child, level + 1, isOriginal, expandedFolders))}
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
              alert('Instructions:\n\n• Click files to highlight and auto-expand in other view\n• Click folders to expand/collapse\n• Drag and drop to move items\n• Esc to clear file selection');
            }}>
              ?</InstructionsButton>
          </div>
          {loading ? (
            <LoadingMessage>Loading folder structure...</LoadingMessage>
          ) : folderComparison ? (
            <>
              <FolderTree>
                <div style={{ display: 'flex', gap: '2rem', height: '100%', width: '100%' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '0.5rem', color: '#6b7280' }}>Original Structure</div>
                    {renderNode(folderComparison.original, 0, true, expandedFoldersOriginal)}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '0.5rem', color: '#6b7280' }}>New Structure</div>
                    {renderNode(folderComparison.new, 0, false, expandedFoldersNew)}
                  </div>
                </div>
              </FolderTree>
              
            </>
          ) : (
            <ErrorMessage>Failed to load folder structure</ErrorMessage>
          )}
        </ContentContainer>
        
      </MainContainer>
    </PageContainer>
  );
};

const PageContainer = styled.div`
  background-color: #f3f4f6;
  height: calc(100vh - 57px); /* Account for navbar height */
  overflow: hidden;
  display: flex;
  flex-direction: column;
  width: 100vw;
  box-sizing: border-box;
`;

const MainContainer = styled.div`
  display: flex;
  flex-direction: column;
  width: 100%;
  flex: 1;
  min-height: 0; /* Critical for proper flexbox behavior with scrolling */
  overflow: hidden;
  padding: 1rem;
  box-sizing: border-box;
`;

const ContentContainer = styled.div`
  width: 100%;
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  padding: 1.5rem;
  box-sizing: border-box;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0; /* Critical for proper flexbox behavior with scrolling */
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  padding: 0 0.5rem;
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
  width: 100%;
  
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
  width: 100%;
  box-sizing: border-box;
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


const LoadingMessage = styled.div`
  text-align: center;
  padding: 2rem;
  color: #6b7280;
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