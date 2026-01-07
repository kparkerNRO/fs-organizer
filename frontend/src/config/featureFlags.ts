/**
 * Feature flags configuration.
 *
 * This file controls which experimental features are enabled.
 * Features behind flags can be developed and tested without affecting production users.
 */

export interface FeatureFlags {
  // Use dual representation API for import wizard
  useDualRepresentation: boolean;

  // Enable advanced debugging tools
  enableDebugTools: boolean;
}

// Default feature flags (can be overridden by environment variables)
const defaultFlags: FeatureFlags = {
  useDualRepresentation: false, // Default to false (use existing implementation)
  enableDebugTools: false,
};

/**
 * Get feature flags with environment variable overrides.
 */
function getFeatureFlags(): FeatureFlags {
  const flags = { ...defaultFlags };

  // Check for environment variable overrides
  // In Vite, env variables are accessed via import.meta.env
  if (typeof import.meta !== 'undefined' && import.meta.env) {
    // VITE_USE_DUAL_REPRESENTATION=true will enable the dual representation feature
    if (import.meta.env.VITE_USE_DUAL_REPRESENTATION === 'true') {
      flags.useDualRepresentation = true;
    }

    if (import.meta.env.VITE_ENABLE_DEBUG_TOOLS === 'true') {
      flags.enableDebugTools = true;
    }
  }

  // Also check localStorage for runtime overrides (useful for testing)
  if (typeof window !== 'undefined' && window.localStorage) {
    const storedFlags = window.localStorage.getItem('featureFlags');
    if (storedFlags) {
      try {
        const parsed = JSON.parse(storedFlags);
        Object.assign(flags, parsed);
      } catch (e) {
        console.warn('Failed to parse feature flags from localStorage:', e);
      }
    }
  }

  return flags;
}

// Export the current feature flags
export const featureFlags = getFeatureFlags();

/**
 * Update a feature flag at runtime (useful for development/testing).
 * Changes are persisted to localStorage.
 */
export function setFeatureFlag(key: keyof FeatureFlags, value: boolean): void {
  if (typeof window === 'undefined' || !window.localStorage) {
    console.warn('Cannot set feature flag: localStorage not available');
    return;
  }

  featureFlags[key] = value;

  // Save to localStorage
  window.localStorage.setItem('featureFlags', JSON.stringify(featureFlags));

  console.log(`Feature flag '${key}' set to:`, value);
  console.log('Reload the page to apply changes.');
}

/**
 * Reset all feature flags to defaults.
 */
export function resetFeatureFlags(): void {
  if (typeof window !== 'undefined' && window.localStorage) {
    window.localStorage.removeItem('featureFlags');
  }

  console.log('Feature flags reset to defaults. Reload the page to apply.');
}

// Log current feature flags in development
if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.DEV) {
  console.log('Feature Flags:', featureFlags);
  console.log('To toggle flags at runtime, use:');
  console.log('  setFeatureFlag("useDualRepresentation", true)');
  console.log('  resetFeatureFlags()');

  // Make functions available globally in development
  if (typeof window !== 'undefined') {
    (window as any).setFeatureFlag = setFeatureFlag;
    (window as any).resetFeatureFlags = resetFeatureFlags;
  }
}
