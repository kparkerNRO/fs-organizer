# Dual Representation Feature Flag

## Overview

The dual representation feature is an experimental new version of the Import Wizard that uses the v2 API with dual representation data structures. This provides a unified view of both the original folder structure and the categorized organization side-by-side.

## Current Status

🚧 **EXPERIMENTAL** - This feature is behind a feature flag and not enabled by default.

## How to Enable

### Option 1: Environment Variable (Recommended for Development)

Add to your `.env` file in the `frontend` directory:

```bash
VITE_USE_DUAL_REPRESENTATION=true
```

Then restart your development server:

```bash
npm run dev
```

### Option 2: Browser Console (Runtime Toggle)

You can toggle the feature flag at runtime using the browser console:

```javascript
// Enable the feature
setFeatureFlag("useDualRepresentation", true)

// Disable the feature
setFeatureFlag("useDualRepresentation", false)

// Reset all flags to defaults
resetFeatureFlags()
```

After changing flags in the console, **reload the page** to see the changes.

## Features

When the dual representation feature is enabled, you'll see:

### 1. Simplified Workflow
- **Step 1: Gather & Group** - Processes the folder and runs both gather and group operations automatically
- **Step 2: View Results** - Displays the dual representation with synchronized highlighting

### 2. Side-by-Side Views
- **File System Structure** - Shows the original folder hierarchy
- **Category Structure** - Shows the categorized organization
- Both views share the same items and highlight synchronously when you hover

### 3. New Data Structure
- Uses the `/api/v2/folder-structure` endpoint
- Flattened item store for efficient lookups
- Separate hierarchies for nodes and categories
- Real-time synchronized highlighting

## Differences from Original Import Wizard

| Feature | Original | Dual Representation |
|---------|----------|---------------------|
| API Version | v1 | v2 |
| Data Structure | Nested tree | Flattened items + hierarchies |
| Steps | 4 (Import, Group, Organize, Review) | 2 (Gather & Group, View) |
| Views | Sequential comparison | Side-by-side dual view |
| Highlighting | Within view only | Synchronized across views |
| Status | Stable | Experimental |

## Architecture

### Components

1. **`DualRepresentationImportWizard.tsx`** - Main wizard component
2. **`HierarchyBrowser.tsx`** - Renders a single hierarchy (node or category)
3. **`useDualRepresentation.ts`** - React hook for state management

### Data Flow

```
User Input (folder path)
  ↓
gatherFiles() → /api/gather
  ↓
groupFolders() → /api/group
  ↓
getDualRepresentation() → /api/v2/folder-structure
  ↓
DualRepresentation {
  items: { "node-1": {...}, "category-1": {...} },
  node_hierarchy: { "node-root": ["node-1", ...] },
  category_hierarchy: { "category-root": ["category-1", ...] }
}
  ↓
HierarchyBrowser (x2) - Render both hierarchies
```

## Known Limitations

1. **No Edit Functionality** - Currently view-only; the diff application is not yet implemented
2. **No Apply/Export** - Missing the final "apply organization" step
3. **Experimental API** - The v2 API is still being developed
4. **Limited Testing** - Not extensively tested in production scenarios

## Testing

To test the dual representation feature:

1. Enable the feature flag (see "How to Enable" above)
2. Navigate to the Import Wizard
3. You should see an "EXPERIMENTAL" badge in the header
4. Select a folder and click "Process"
5. After processing, click "View Dual Representation →"
6. Hover over items in either view to see synchronized highlighting

## Debugging

The feature flag system logs to the console in development mode:

```
Feature Flags: { useDualRepresentation: true, ... }
To toggle flags at runtime, use:
  setFeatureFlag("useDualRepresentation", true)
  resetFeatureFlags()
```

You can also inspect the current state:

```javascript
// Check current feature flags
console.log(featureFlags)

// Check localStorage overrides
console.log(JSON.parse(localStorage.getItem('featureFlags')))
```

## Future Enhancements

- [ ] Add diff application functionality
- [ ] Implement drag-and-drop for moving items between categories
- [ ] Add export/apply organization step
- [ ] Expand/collapse controls for hierarchies
- [ ] Search and filter capabilities
- [ ] Performance optimizations for large hierarchies

## Feedback

Since this is experimental, we'd love your feedback! Please report:
- Any bugs or unexpected behavior
- Performance issues with large folder structures
- UI/UX suggestions
- Missing features you'd like to see

## Related Files

- `frontend/src/config/featureFlags.ts` - Feature flag configuration
- `frontend/src/pages/DualRepresentationImportWizard.tsx` - New wizard
- `frontend/src/components/HierarchyBrowser.tsx` - Hierarchy rendering
- `frontend/src/hooks/useDualRepresentation.ts` - State management hook
- `frontend/src/App.tsx` - Main app with feature flag integration
