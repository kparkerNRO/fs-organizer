// src/App.tsx
import React, { useState } from "react";
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
  const [activeView, setActiveView] = useState<string>('import');

  const handleNavItemClick = (itemId: string) => {
    setActiveView(itemId);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <NavBar
        items={navItems}
        activeItemId={activeView}
        onNavItemClick={handleNavItemClick}
      />
      <main className="flex-1">
        {activeView === "categories" ? (
          <CategoriesPage />
        ) : activeView === "folders" ? (
          <FolderStructurePage />
        ) : (
          <ImportWizardPage />
        )}
      </main>
    </div>
  );
}

export default App;