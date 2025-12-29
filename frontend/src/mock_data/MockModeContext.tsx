// src/MockModeContext.tsx
import React, { createContext, useContext, useState } from "react";

const MockModeContext = createContext(false);

export const MockModeProvider: React.FC<React.PropsWithChildren<object>> = ({
  children,
}) => {
  const [useMockData] = useState(true); // Set to false for real API
  return (
    <MockModeContext.Provider value={useMockData}>
      {children}
    </MockModeContext.Provider>
  );
};

export const useMockMode = () => {
  const context = useContext(MockModeContext);
  if (context === undefined) {
    throw new Error("useMockMode must be used within a MockModeProvider");
  }
  return context;
};
