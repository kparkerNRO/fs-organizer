// src/App.tsx
import React, { useState, useEffect } from "react";
import styled from "styled-components";
import { CategoriesPage } from "./pages/CategoriesPage";
import { FolderStructurePage } from "./pages/FolderStructurePage";
import { ImportWizardPage } from "./pages/ImportWizardPage";
import { NavBar, NavItem } from "./components/NavBar";

// Navigation items
const navItems: NavItem[] = [
  { id: "categories", label: "Categorize" },
  { id: "folders", label: "Folder Structure" },
  { id: "import", label: "Import Wizard" }
];

function App() {
  // Get initial view from URL path
  const getInitialView = () => {
    const path = window.location.pathname;
    if (path === '/categories') return 'categories';
    if (path === '/folders') return 'folders';
    if (path === '/import') return 'import';
    return 'categories'; // default
  };

  const [activeView, setActiveView] = useState<string>(getInitialView);

  // Update URL when view changes
  const handleNavItemClick = (itemId: string) => {
    setActiveView(itemId);
    const newPath = `/${itemId}`;
    window.history.pushState(null, '', newPath);
  };

  // Listen for browser back/forward navigation
  useEffect(() => {
    const handlePopState = () => {
      const newView = getInitialView();
      setActiveView(newView);
    };

    window.addEventListener('popstate', handlePopState);
    
    // Set initial URL if needed
    const currentPath = window.location.pathname;
    if (currentPath === '/' || (!currentPath.startsWith('/categories') && !currentPath.startsWith('/folders') && !currentPath.startsWith('/import'))) {
      window.history.replaceState(null, '', '/categories');
      setActiveView('categories');
    }
    
    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

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