import React from "react";
import styled from "styled-components";
import { Category, Folder } from "../types/types";

interface CategoryDetailsProps {
  item: Category | Folder | null;
}

export const CategoryDetails: React.FC<CategoryDetailsProps> = ({ item }) => {
  if (!item) {
    return (
      <DetailsContainer>
        <PlaceholderText>Select a category to view details</PlaceholderText>
      </DetailsContainer>
    );
  }

  // TODO is this really the best way?
  const isFolder = (item: Category | Folder): item is Folder => {
    return "original_path" in item;
  };

  if (isFolder(item)) {
    return (
      <DetailsContainer>
        <DetailsGrid>
          <FieldContainer>
            <Label>Category Name</Label>
            <Input type="text" value={item.cleanedName} readOnly />
          </FieldContainer>

          <FieldContainer>
            <Label>Name:</Label>
            <Input type="text" value={item.name} readOnly />
          </FieldContainer>

          <FieldContainer>
            <Label>Classification</Label>
            <SelectInput>
              <option value={item.classification}>{item.classification}</option>
            </SelectInput>
          </FieldContainer>
        </DetailsGrid>

        <FieldRow>
          <Label>Original Filename</Label>
          <Input type="text" value={item.original_filename} readOnly />
        </FieldRow>

        <FieldRow>
          <Label>Original Path</Label>
          <Input type="text" value={item.original_path} readOnly />
        </FieldRow>

        <FieldRow>
          <Label>Confidence</Label>
          <ConfidenceBar>
            <ConfidenceFill style={{ width: `${item.confidence}%` }}>
              {item.confidence}%
            </ConfidenceFill>
          </ConfidenceBar>
        </FieldRow>

        <ProcessedNamesContainer>
          <Label>Processed Names</Label>
          <ProcessedNamesInput
            value={item.processed_names?.join("\n") || ""}
            readOnly
          />
        </ProcessedNamesContainer>
      </DetailsContainer>
    );
  }

  // Render Category details
  return (
    <DetailsContainer>
      <DetailsGrid>
        <FieldContainer>
          <Label>Category Name</Label>
          <Input type="text" value={item.name} readOnly />
        </FieldContainer>

        <FieldContainer>
          <Label>Classification</Label>
          <SelectInput>
            <option value={item.classification}>{item.classification}</option>
          </SelectInput>
        </FieldContainer>

        <FieldContainer>
          <Label>Confidence</Label>
          <ConfidenceBar>
            <ConfidenceFill style={{ width: `${item.confidence}%` }}>
              {item.confidence}%
            </ConfidenceFill>
          </ConfidenceBar>
        </FieldContainer>
      </DetailsGrid>

      <FieldRow>
        <Label>Possible Classifications</Label>
        <Input
          type="text"
          value={item.possibleClassifications?.join(", ") || ""}
          readOnly
        />
      </FieldRow>
    </DetailsContainer>
  );
};

const DetailsContainer = styled.div`
  margin-top: 1.5rem;
  padding: 1.5rem;
  background: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
`;

const DetailsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
  margin-bottom: 1.5rem;
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
