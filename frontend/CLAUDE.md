# FS-Organizer Frontend Guide

## Build & Development Commands
```
npm run dev          # Start development server
npm run build        # Build for production
npm run lint         # Run ESLint
npm run preview      # Preview production build
```

## Code Style Guidelines
- **TypeScript**: Use strict typing; avoid `any` types
- **Imports**: Group imports by external libraries, then internal components, then types/utils
- **Components**: Use functional components with React hooks
- **Styling**: Use styled-components following the component pattern
- **State Management**: Use React hooks (useState, useEffect, custom hooks)
- **Error Handling**: Use try/catch blocks for async operations, log errors with console.error
- **Naming**: 
  - PascalCase for components and types
  - camelCase for variables, functions, and instances
  - Descriptive, specific naming

## File Structure
- Organize code by feature/domain
- Keep components focused on a single responsibility
- Place reusable hooks in `/hooks` directory
- Define types in `/types` directory