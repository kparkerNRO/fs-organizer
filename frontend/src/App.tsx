// src/App.tsx
import React, { useState } from "react";
import styled from "styled-components";
import { CategoriesPage } from "./pages/CategoriesPage";
import { FolderStructurePage } from "./pages/FolderStructurePage";
import { NavBar, NavItem } from "./components/NavBar";

// Navigation items
const navItems: NavItem[] = [
  { id: "categorize", label: "Categorize" },
  { id: "folder-structure", label: "Folder Structure" }
];

function App() {
  const [activeView, setActiveView] = useState<string>("categorize");

  const handleNavItemClick = (itemId: string) => {
    setActiveView(itemId);
  };

  return (
    <AppContainer>
      <NavBar
        items={navItems}
        activeItemId={activeView}
        onNavItemClick={handleNavItemClick}
      />
      <MainContent>
        {activeView === "categorize" ? (
          <CategoriesPage />
        ) : (
          <FolderStructurePage />
        )}
      </MainContent>
    </AppContainer>
  );
}

const AppContainer = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
`;

const MainContent = styled.main`
  flex: 1;
`;

export default App;