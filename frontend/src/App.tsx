// src/App.tsx
import React, { useState } from "react";
import styled from "styled-components";
import { CategoriesPage } from "./pages/CategoriesPage";
import { FolderStructurePage } from "./pages/FolderStructurePage";
import { ImportWizardPage } from "./pages/ImportWizardPage";
import { NavBar, NavItem } from "./components/NavBar";

// Navigation items
const navItems: NavItem[] = [
  { id: "categories", label: "Categorize" },
  { id: "folders", label: "Folder Structure" },
  { id: "import", label: "Import Wizard" },
];

function App() {
  const [activeView, setActiveView] = useState<string>("import");

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
        {activeView === "categories" ? (
          <CategoriesPage />
        ) : activeView === "folders" ? (
          <FolderStructurePage />
        ) : (
          <ImportWizardPage />
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
