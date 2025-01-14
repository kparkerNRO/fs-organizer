// src/api.ts

import { fetchMockFolders } from "./mock_data/mockApi";
import { useMockMode } from "./mock_data/MockModeContext";

export const fetchFolders = async () => {
  const useMockData = useMockMode();

  if (useMockData) {
    return fetchMockFolders();
  }

  const response = await fetch(`/api/folders`);
  return response.json();
};
