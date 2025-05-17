import React from "react";
import styled from "styled-components";
import { CategoryDetailsProps } from "../types/types";

export const CategoryDetails: React.FC<CategoryDetailsProps> = ({
  category,
  folder,
}) => {
  // If neither category nor folder is provided, return the placeholder
  if (!category && !folder) {
    return (
      <DetailsContainer>
        <PlaceholderText>Select a folder to view details</PlaceholderText>
      </DetailsContainer>
    );
  }
  if (folder != null) {
    return (
      <DetailsContainer>
        <DetailsGrid>
          <FieldContainer>
            <Label>Category Name</Label>
            <Input type="text" value={category?.name || ""} readOnly />
          </FieldContainer>

          <FieldContainer>
            <Label>Name:</Label>
            <Input type="text" value={folder.name || ""} readOnly />
          </FieldContainer>

          <FieldContainer>
            <Label>Classification</Label>
            <SelectInput>
              <option value={folder.classification}>
                {folder.classification}
              </option>
            </SelectInput>
          </FieldContainer>
        </DetailsGrid>

        <FieldRow>
          <Label>Original Filename</Label>
          <Input type="text" value={folder.original_filename || ""} readOnly />
        </FieldRow>

        <FieldRow>
          <Label>Original Path</Label>
          <Input type="text" value={folder.original_path || ""} readOnly />
        </FieldRow>

        <FieldRow>
          <Label>Confidence</Label>
          <ConfidenceBar>
            <ConfidenceFill style={{ width: `${folder.confidence}%` }}>
              {folder.confidence}%
            </ConfidenceFill>
          </ConfidenceBar>
        </FieldRow>

        <ProcessedNamesContainer>
          <Label>Processed Names</Label>
          <ProcessedNamesInput
            value={folder.processed_names?.join("\n") || ""}
            readOnly
          />
        </ProcessedNamesContainer>
      </DetailsContainer>
    );
  }

  if (category != null) {
    // Render Category details
    return (
      <DetailsContainer>
        <DetailsGrid>
          <FieldContainer>
            <Label>Category Name</Label>
            <Input type="text" value={category.name || ""} readOnly />
          </FieldContainer>

          <FieldContainer>
            <Label>Classification</Label>
            <SelectInput>
              <option value={category.classification}>
                {category.classification}
              </option>
            </SelectInput>
          </FieldContainer>

          <FieldContainer>
            <Label>Confidence</Label>
            <ConfidenceBar>
              <ConfidenceFill style={{ width: `${category.confidence}%` }}>
                {category.confidence}%
              </ConfidenceFill>
            </ConfidenceBar>
          </FieldContainer>
        </DetailsGrid>

        <FieldRow>
          <Label>Possible Classifications</Label>
          <Input
            type="text"
            value={category.possibleClassifications?.join(", ") || ""}
            readOnly
          />
        </FieldRow>
      </DetailsContainer>
    );
  }
  if (!category && !folder) {
    return (
      <DetailsContainer>
        <PlaceholderText>Select a category to view details</PlaceholderText>
      </DetailsContainer>
    );
  }
};

const DetailsContainer = styled.div`
  padding: 1.5rem;
  background: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  width: 100%;
  box-sizing: border-box;
  overflow-x: hidden;
  overflow-y: auto;
  
  .folder-structure-page & {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0; /* Critical for proper flexbox behavior with scrolling */
    overflow-y: auto;
  }
  
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

const DetailsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1.5rem;
  margin-bottom: 1.5rem;
  
  .folder-structure-page & {
    grid-template-columns: 1fr;
  }
`;

const FieldContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
`;
const FieldRow = styled.div`
  margin-top: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  width: 100%;
`;

const Label = styled.label`
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
`;

const Input = styled.input`
  width: 100%;
  padding: 0.5rem 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 0.5rem;
  outline: none;
  box-sizing: border-box;

  &:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
  }
`;

const SelectInput = styled.select`
  width: 100%;
  padding: 0.5rem 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 0.5rem;
  outline: none;
  background-color: white;

  &:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
  }
`;

const ProcessedNamesContainer = styled.div`
  margin-top: 1rem;
`;

const ProcessedNamesInput = styled(Input)`
  min-height: 100px;
  resize: vertical;
  width: 100%;
  box-sizing: border-box;
`;

const ConfidenceBar = styled.div`
  width: 100%;
  height: 2.5rem;
  background-color: #d1d5db;
  border-radius: 0.5rem;
  overflow: hidden;
`;

const ConfidenceFill = styled.div`
  height: 100%;
  background-color: #6b7280;
  border-radius: 0.5rem;
  transition: width 0.3s ease;
`;

const GroupMembersSection = styled.div`
  margin-top: 1.5rem;
`;

const GroupTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 500;
  margin-bottom: 1rem;
`;

const MembersGrid = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const HeaderRow = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
  align-items: center;
`;

const MemberRow = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
`;

const PlaceholderText = styled.p`
  color: #6b7280;
`;
