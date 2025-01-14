// src/components/FolderList.tsx
import React, { useEffect, useState } from "react";
import { fetchMockFolders } from "../mock_data/mockApi";
import { Folder } from "../types";

export const FolderList: React.FC = () => {
  const [folders, setFolders] = useState<Folder[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      const data = await fetchMockFolders(); // Use mock API
      setFolders(data);
    };
    fetchData();
  }, []);

  return (
    <div>
      <h1>Folders</h1>
      <ul>
        {folders.map((folder) => (
          <li key={folder.id}>
            {folder.folderName} ({folder.categories.length} categories)
          </li>
        ))}
      </ul>
    </div>
  );
};
