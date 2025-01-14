// src/MockModeContext.tsx
import React, { createContext, useContext, useState } from "react";

const MockModeContext = createContext(false);

export const MockModeProvider: React.FC<React.PropsWithChildren<{}>> = ({ children }) => {
  const [useMockData] = useState(true); // Set to false for real API
  return (
    <MockModeContext.Provider value={useMockData}>
      {children}
    </MockModeContext.Provider>
  );
};

export const useMockMode = () => useContext(MockModeContext);
