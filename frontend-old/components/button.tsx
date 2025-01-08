// src/components/Button/Button.tsx
import React from "react";

interface ButtonProps {
  label: string;
  onClick: () => void;
  variant?: "primary" | "secondary";
}

export const Button: React.FC<ButtonProps> = ({ label, onClick, variant = "primary" }) => {
  return (
    <button
      className={`button ${variant}`}
      onClick={onClick}
    >
      {label}
    </button>
  );
};
