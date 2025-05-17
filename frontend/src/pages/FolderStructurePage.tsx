// src/pages/FolderStructurePage.tsx
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { fetchFolderStructure, FolderNode } from "../api";
import { ChevronDown, ChevronRight } from "lucide-react";

export const FolderStructurePage: React.FC = () => {
  const [folderStructure, setFolderStructure] = useState<FolderNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(["root"]));

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

  const renderFolderNode = (node: FolderNode, level: number = 0) => {
    const isExpanded = expandedFolders.has(node.id);
    const hasChildren = node.children && node.children.length > 0;

    return (
      <div key={node.id}>
        <FolderItem 
          $level={level} 
          onClick={() => hasChildren && toggleFolder(node.id)}
        >
          {hasChildren && (
            <ExpandIcon>
              {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </ExpandIcon>
          )}
          <FolderName>{node.name}</FolderName>
          {node.path && <FolderPath>{node.path}</FolderPath>}
        </FolderItem>
        
        {isExpanded && hasChildren && (
          <div>
            {node.children?.map(child => renderFolderNode(child, level + 1))}
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

      <ContentContainer>
        {loading ? (
          <LoadingMessage>Loading folder structure...</LoadingMessage>
        ) : folderStructure ? (
          <FolderTree>
            {renderFolderNode(folderStructure)}
          </FolderTree>
        ) : (
          <ErrorMessage>Failed to load folder structure</ErrorMessage>
        )}
      </ContentContainer>
    </PageContainer>
  );
};

const PageContainer = styled.div`
  background-color: #f3f4f6;
  padding: 2rem;
  min-height: calc(100vh - 57px); /* Account for navbar height */
`;

const ContentContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  padding: 1.5rem;
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
  margin-top: 1rem;
`;

const FolderItem = styled.div<{ $level: number }>`
  display: flex;
  align-items: center;
  padding: 0.5rem;
  padding-left: ${(props) => props.$level * 1.5 + 0.5}rem;
  cursor: pointer;
  border-radius: 0.25rem;
  transition: background-color 0.2s;

  &:hover {
    background-color: #f3f4f6;
  }
`;

const ExpandIcon = styled.span`
  display: inline-flex;
  align-items: center;
  margin-right: 0.5rem;
`;

const FolderName = styled.span`
  font-weight: 500;
`;

const FolderPath = styled.span`
  margin-left: 1rem;
  color: #6b7280;
  font-size: 0.875rem;
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