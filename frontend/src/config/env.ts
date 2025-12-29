// src/config/env.ts

interface EnvConfig {
  apiUrl: string;
  nodeEnv: string;
}

export const env: EnvConfig = {
  apiUrl: import.meta.env.VITE_API_URL || "http://localhost:8000",
  nodeEnv: import.meta.env.NODE_ENV || "development",
};
