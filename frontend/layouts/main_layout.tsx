// src/layouts/MainLayout.tsx
import React from "react";

export const MainLayout: React.FC = ({ children }) => {
  return (
    <div className="layout">
      <aside className="sidebar">Sidebar</aside>
      <main className="content">{children}</main>
    </div>
  );
};
