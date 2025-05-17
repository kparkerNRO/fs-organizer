import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { MockModeProvider } from './mock_data/MockModeContext'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <MockModeProvider>
      <App />
    </MockModeProvider>
  </StrictMode>,
)
