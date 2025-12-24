# Frontend Context (Electron + React)

Context for working with the Electron desktop app frontend.

## Project Structure

```
frontend/
├── electron/
│   ├── main.ts      # Main process: window lifecycle, native APIs
│   └── preload.ts   # Preload: secure IPC bridge
├── src/
│   ├── components/  # React components (NavBar, CategoryDetails, etc.)
│   ├── types/       # TypeScript types (types.ts, enums.ts)
│   ├── mock_data/   # Mock data system (mockApi.ts, MockModeContext.tsx)
│   ├── utils/       # Utilities (folderTreeOperations.ts, folderSelection.ts)
│   ├── test/        # Test setup and utilities
│   ├── api.ts       # Backend API client (isMockMode flag)
│   └── App.tsx      # Main app component
└── .nvmrc           # Node 20 LTS
```

## Key Patterns

### Mock Mode System
- **Toggle**: `isMockMode` flag in `api.ts` (currently `true`)
- **Context**: `MockModeContext.tsx` provides mock data to entire app
- **Data**: `mockData.ts` contains static test data
- **Usage**: Allows UI development without backend running

### Component Architecture
- **Functional components** with React hooks (no class components)
- **Styled-components** for all styling (co-located with components)
- **Props**: Explicitly typed interfaces (e.g., `interface CategoryDetailsProps`)
- **State**: Local useState, no global state management library

### Electron IPC
- **Main → Renderer**: Use preload script to expose safe APIs
- **Security**: Never expose full Node.js to renderer
- **Pattern**: Define IPC handlers in main.ts, expose via preload.ts

### Testing
- **Framework**: Vitest + @testing-library/react
- **Location**: `src/test/` for setup, co-located `*.test.ts*` for component tests
- **Pattern**: See `folderTreeOperations.test.ts` for examples
- **Mocks**: `src/test/mocks.ts` for shared test data

## Code Conventions

### TypeScript
- **Strict mode**: Enabled, avoid `any` types
- **Types location**: `src/types/types.ts` for shared types
- **Enums**: `src/types/enums.ts` for constants
- **Naming**: PascalCase for types/components, camelCase for functions/variables

### Styling
- **Method**: styled-components (no CSS files)
- **Pattern**: Define styled components at bottom of file
- **Theme**: No theme system currently, styles are component-local
- **Responsive**: Design for desktop, not mobile

### Imports
```typescript
// 1. External libraries
import React, { useState } from 'react';
import styled from 'styled-components';

// 2. Internal components
import { CategoryDetails } from './components/CategoryDetails';

// 3. Types and utils
import { Category } from './types/types';
import { formatDate } from './utils/helpers';
```

## Common Tasks

### Add New Component
1. Create in `src/components/ComponentName.tsx`
2. Define props interface: `interface ComponentNameProps`
3. Use functional component with hooks
4. Add styled-components at bottom
5. Export as named export

### Modify Mock Data
- Edit `src/mock_data/mockData.ts`
- Update `mockApi.ts` if adding new endpoints
- Context will auto-update all consumers

### Add Electron Native Feature
1. Add handler in `electron/main.ts`
2. Expose via `electron/preload.ts`
3. Use in renderer via `window.electron` (if exposed)

### Run Tests
```bash
npm run test          # Watch mode
npm run test:run      # Single run
npm run test:ui       # Visual test UI
```

## Critical Files

- `api.ts` - Backend API client, toggle mock mode here
- `mock_data/MockModeContext.tsx` - Mock data provider
- `components/App.tsx` - Main app router/layout
- `electron/main.ts` - Electron app lifecycle
- `types/types.ts` - Shared TypeScript types