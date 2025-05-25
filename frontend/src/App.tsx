// src/App.tsx
import React, { useState, useEffect } from "react";
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
  // Get initial view from URL hash or default to "categorize"
  const getInitialView = () => {
    const hash = window.location.hash.replace('#', '');
    return hash && ['categorize', 'folder-structure'].includes(hash) ? hash : 'categorize';
  };

  const [activeView, setActiveView] = useState<string>(getInitialView);

  // Update URL when view changes
  const handleNavItemClick = (itemId: string) => {
    setActiveView(itemId);
    window.history.replaceState(null, '', `#${itemId}`);
  };

  // Listen for browser back/forward navigation
  useEffect(() => {
    const handleHashChange = () => {
      const newView = getInitialView();
      setActiveView(newView);
    };

    window.addEventListener('hashchange', handleHashChange);
    
    return () => {
      window.removeEventListener('hashchange', handleHashChange);
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