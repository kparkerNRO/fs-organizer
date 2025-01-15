import React from 'react';
import styled from 'styled-components';

interface ResetButtonProps {
  onReset: () => void;
}

export const ResetButton: React.FC<ResetButtonProps> = ({ onReset }) => (
  <Button onClick={onReset}>
    Reset to Initial Data
  </Button>
);

const Button = styled.button`
  background-color: #ef4444;
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  border: none;
  cursor: pointer;
  font-size: 0.875rem;
  transition: background-color 0.2s;

  &:hover {
    background-color: #dc2626;
  }

  &:focus {
    outline: 2px solid #dc2626;
    outline-offset: 2px;
  }
`;