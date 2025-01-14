import React from "react";
import styled from "styled-components";
import { Category } from "../types";
import { ChevronDown } from "lucide-react";

interface CategoryDetailsProps {
  category: Category | null;
}

export const CategoryDetails: React.FC<CategoryDetailsProps> = ({ category }) => {
  if (!category) {
    return (
      <DetailsContainer>
        <PlaceholderText>Select a category to view details</PlaceholderText>
      </DetailsContainer>
    );
  }

  return (
    <DetailsContainer>
      <DetailsGrid>
        <FieldContainer>
          <Label>Category Name</Label>
          <Input type="text" value={category.name} readOnly />
        </FieldContainer>

        <FieldContainer>
          <Label>Confidence</Label>
          <ConfidenceBar>
            <ConfidenceFill style={{ width: `${category.confidence}%` }} />
          </ConfidenceBar>
        </FieldContainer>

        <FieldContainer>
          <Label>Classification</Label>
          <Input type="text" value={category.classification} readOnly />
        </FieldContainer>
      </DetailsGrid>

      <GroupMembersSection>
        <GroupTitle>Group Members</GroupTitle>
        <MembersGrid>
          <HeaderRow>
            <Label>Name</Label>
            <Label>Original Name</Label>
            <Label>
              Subcategories <ChevronDown size={16} />
            </Label>
          </HeaderRow>
          {[1, 2, 3].map((_, index) => (
            <MemberRow key={index}>
              <Input type="text" readOnly />
              <Input type="text" readOnly />
              <Input type="text" readOnly />
            </MemberRow>
          ))}
        </MembersGrid>
      </GroupMembersSection>
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