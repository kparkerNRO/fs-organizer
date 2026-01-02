// src/App.tsx
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { CategoriesPage } from "./pages/CategoriesPage";
import { FolderStructurePage } from "./pages/FolderStructurePage";
import { ImportWizardPage } from "./pages/ImportWizardPage";
import { NavBar, NavItem } from "./components/NavBar";
import { getPipelineStatus } from "./api";

// Navigation items
const navItems: NavItem[] = [
  { id: "categories", label: "Categorize" },
  { id: "folders", label: "Folder Structure" },
  { id: "import", label: "Import Wizard" },
];

function App() {
  const [activeView, setActiveView] = useState<string>("import");
  const [isCheckingStatus, setIsCheckingStatus] = useState<boolean>(true);

  // Check for existing data on mount and auto-navigate to the most recent stage
  useEffect(() => {
    const checkForExistingData = async () => {
      try {
        const status = await getPipelineStatus();

        if (status) {
          // Navigate to the most advanced stage that has data
          if (status.has_folders) {
            console.log(
              "Found existing folders structure, navigating to Folder Structure view",
            );
            setActiveView("folders");
          } else if (status.has_group) {
            console.log(
              "Found existing group structure, navigating to Import Wizard",
            );
            setActiveView("import");
          } else if (status.has_gather) {
            console.log(
              "Found existing gather structure, navigating to Import Wizard",
            );
            setActiveView("import");
          }
        }
      } catch (error) {
        console.error("Error checking for existing data:", error);
      } finally {
        setIsCheckingStatus(false);
      }
    };

    checkForExistingData();
  }, []);

  const handleNavItemClick = (itemId: string) => {
    setActiveView(itemId);
  };

  // Show loading state while checking for existing data
  if (isCheckingStatus) {
    return (
      <AppContainer>
        <LoadingContainer>
          <LoadingSpinner />
          <LoadingText>Loading...</LoadingText>
        </LoadingContainer>
      </AppContainer>
    );
  }

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

const LoadingContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  gap: 1rem;
`;

const LoadingSpinner = styled.div`
  width: 40px;
  height: 40px;
  border: 4px solid #e5e7eb;
  border-top: 4px solid #2563eb;
  border-radius: 50%;
  animation: spin 1s linear infinite;

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

const LoadingText = styled.div`
  color: #374151;
  font-size: 1rem;
  font-weight: 500;
`;

export default App;
