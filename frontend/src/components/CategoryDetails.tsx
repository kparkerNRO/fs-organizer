import React from "react";
import styled from "styled-components";
import { CategoryDetailsProps } from "../types/types";
import { FileIcon, FolderIcon } from "lucide-react";

export const CategoryDetails: React.FC<CategoryDetailsProps> = ({
  category,
  folder,
  file,
}) => {
  // If no item is selected, return the placeholder
  if (!category && !folder && !file) {
    return (
      <DetailsContainer>
        <DetailHeader>
          <DetailTitle>Details</DetailTitle>
        </DetailHeader>
        <PlaceholderContainer>
          <PlaceholderText>
            Select a folder or file to view details
          </PlaceholderText>
        </PlaceholderContainer>
      </DetailsContainer>
    );
  }

  // Render file details if a file is selected
  if (file) {
    return (
      <DetailsContainer>
        <DetailHeader>
          <FileIcon size={18} style={{ marginRight: "0.5rem", opacity: 0.7 }} />
          <DetailTitle>File Details</DetailTitle>
        </DetailHeader>

        <DetailsGrid>
          <FieldContainer>
            <Label>File Name</Label>
            <Input type="text" value={file.name || ""} readOnly />
          </FieldContainer>

          <FieldContainer>
            <Label>File Type</Label>
            <Input type="text" value={file.fileType || ""} readOnly />
          </FieldContainer>

          {file.size && (
            <FieldContainer>
              <Label>Size</Label>
              <Input type="text" value={file.size} readOnly />
            </FieldContainer>
          )}
        </DetailsGrid>

        <FieldRow>
          <Label>Original Path</Label>
          <Input type="text" value={file.original_path || ""} readOnly />
        </FieldRow>

        {file.categories && file.categories.length > 0 && (
          <FieldRow>
            <Label>Categories</Label>
            <Input type="text" value={file.categories.join(", ")} readOnly />
          </FieldRow>
        )}

        <FieldRow>
          <Label>Confidence</Label>
          <ConfidenceBar>
            <ConfidenceFill style={{ width: `${file.confidence}%` }}>
              {file.confidence}%
            </ConfidenceFill>
          </ConfidenceBar>
        </FieldRow>
      </DetailsContainer>
    );
  }
  // Render folder details if a folder is selected
  if (folder != null) {
    // Calculate content based on what's available
    const hasProcessedNames =
      folder.processed_names && folder.processed_names.length > 0;

    return (
      <DetailsContainer>
        <DetailHeader>
          <FolderIcon
            size={18}
            style={{ marginRight: "0.5rem", opacity: 0.7 }}
          />
          <DetailTitle>Folder Details</DetailTitle>
        </DetailHeader>

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

        {hasProcessedNames && (
          <FieldRow>
            <Label>Categories</Label>
            <Input
              type="text"
              value={folder.processed_names!.join(", ")}
              readOnly
            />
          </FieldRow>
        )}
        {!hasProcessedNames && <EmptySpace />}

        <FieldRow>
          <Label>Confidence</Label>
          <ConfidenceBar>
            <ConfidenceFill style={{ width: `${folder.confidence}%` }}>
              {folder.confidence}%
            </ConfidenceFill>
          </ConfidenceBar>
        </FieldRow>

        {hasProcessedNames && (
          <ProcessedNamesContainer>
            <Label>Processed Names</Label>
            <ProcessedNamesInput
              value={folder.processed_names!.join("\n")}
              readOnly
            />
          </ProcessedNamesContainer>
        )}
        {!hasProcessedNames && <EmptySpace />}
      </DetailsContainer>
    );
  }

  if (category != null) {
    // Render Category details
    return (
      <DetailsContainer>
        <DetailHeader>
          <DetailTitle>Category Details</DetailTitle>
        </DetailHeader>

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
    width: 450px;
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

const DetailHeader = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 1.25rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #e5e7eb;
`;

const DetailTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 500;
  color: #111827;
  margin: 0;
`;

const PlaceholderContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  min-height: 200px;
`;

const EmptySpace = styled.div`
  height: 80px;
  width: 100%;
`;

const PlaceholderText = styled.p`
  color: #6b7280;
  font-size: 0.95rem;
`;
