// src/pages/FolderStructurePage.tsx
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { fetchFolderStructure, FolderNode, FileNode } from "../api";
import { ChevronDown, ChevronRight, FolderOpen, Folder as FolderIcon, File as FileIcon } from "lucide-react";
import { CategoryDetails } from "../components/CategoryDetails";
import { CategoryDetailsProps, Folder, FileItem } from "../types/types";

export const FolderStructurePage: React.FC = () => {
  const [folderStructure, setFolderStructure] = useState<FolderNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(["root"]));
  const [selectedItem, setSelectedItem] = useState<CategoryDetailsProps>({category: null, folder: null, file: null});

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

  const isFileNode = (node: FolderNode | FileNode): node is FileNode => {
    return 'fileType' in node || node.path?.includes('.') || false;
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

    return (
      <div key={node.id}>
        <FolderItem 
          $level={level}
          $isFile={isFile}
          onClick={() => {
            selectItem(node);
            if (!isFile && hasChildren) toggleFolder(node.id);
          }}
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
          <div style={{ paddingBottom: '0.5rem', borderBottom: '1px solid #e5e7eb', marginBottom: '0.5rem', display: 'flex', alignItems: 'center' }}>
            <FolderIcon size={16} style={{ marginRight: '0.5rem', opacity: 0.7 }} />
            <span style={{ fontWeight: 500 }}>Folders</span>
          </div>
          {loading ? (
            <LoadingMessage>Loading folder structure...</LoadingMessage>
          ) : folderStructure ? (
            <FolderTree>
              {renderNode(folderStructure)}
            </FolderTree>
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

const FolderItem = styled.div<{ $level: number, $isFile?: boolean }>`
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

  &:hover {
    background-color: #e7eefa;
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

const ErrorMessage = styled.div`
  text-align: center;
  padding: 2rem;
  color: #ef4444;
`;