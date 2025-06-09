import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ImportWizardPage } from './ImportWizardPage';

// Mock the API functions
vi.mock('../mock_data/mockApi', () => ({
  importFolder: vi.fn().mockResolvedValue({
    name: 'test-folder',
    count: 10,
    confidence: 0.9,
    children: []
  }),
  groupFolders: vi.fn().mockResolvedValue({
    name: 'grouped-folder',
    count: 10,
    confidence: 0.9,
    children: []
  }),
  organizeFolders: vi.fn().mockResolvedValue({
    name: 'organized-folder',
    count: 10,
    confidence: 0.9,
    children: []
  }),
  applyOrganization: vi.fn().mockResolvedValue({
    message: 'Organization applied successfully'
  })
}));

describe('ImportWizardPage Layout Consistency', () => {
  const getContainerDimensions = () => {
    const wizardContainer = document.querySelector('[data-testid="wizard-container"]') || 
                           document.querySelector('div'); // fallback
    const wizardContent = document.querySelector('[data-testid="wizard-content"]') ||
                         wizardContainer?.children[1]; // WizardContent is usually second child
    
    return {
      container: wizardContainer ? {
        width: getComputedStyle(wizardContainer).width,
        height: getComputedStyle(wizardContainer).height,
        padding: getComputedStyle(wizardContainer).padding,
        margin: getComputedStyle(wizardContainer).margin,
      } : null,
      content: wizardContent ? {
        width: getComputedStyle(wizardContent).width,
        height: getComputedStyle(wizardContent).height,
        padding: getComputedStyle(wizardContent).padding,
        margin: getComputedStyle(wizardContent).margin,
      } : null
    };
  };

  it('should maintain consistent container dimensions across all wizard steps', async () => {
    const { rerender } = render(<ImportWizardPage />);
    
    // Step 1: Initial dimensions
    const step1Dimensions = getContainerDimensions();
    expect(step1Dimensions.container).toBeTruthy();
    expect(step1Dimensions.content).toBeTruthy();
    
    // Navigate to step 2
    const sourceInput = screen.getByPlaceholderText('Enter folder path or click Browse...');
    fireEvent.change(sourceInput, { target: { value: '/test/path' } });
    
    const importButton = screen.getByText('Import Folder');
    fireEvent.click(importButton);
    
    // Wait for import to complete and navigate to step 2
    await vi.waitFor(() => {
      const nextButton = screen.queryByText('Next: Group Similar Items');
      if (nextButton) {
        fireEvent.click(nextButton);
      }
    });
    
    // Step 2: Compare dimensions
    const step2Dimensions = getContainerDimensions();
    expect(step2Dimensions.container?.width).toBe(step1Dimensions.container?.width);
    expect(step2Dimensions.container?.height).toBe(step1Dimensions.container?.height);
    expect(step2Dimensions.content?.width).toBe(step1Dimensions.content?.width);
    expect(step2Dimensions.content?.height).toBe(step1Dimensions.content?.height);
    
    // Navigate to step 3
    await vi.waitFor(() => {
      const nextButton = screen.queryByText('Next: Organize Structure');
      if (nextButton) {
        fireEvent.click(nextButton);
      }
    });
    
    // Step 3: Compare dimensions
    const step3Dimensions = getContainerDimensions();
    expect(step3Dimensions.container?.width).toBe(step1Dimensions.container?.width);
    expect(step3Dimensions.container?.height).toBe(step1Dimensions.container?.height);
    expect(step3Dimensions.content?.width).toBe(step1Dimensions.content?.width);
    expect(step3Dimensions.content?.height).toBe(step1Dimensions.content?.height);
    
    // Navigate to step 4
    await vi.waitFor(() => {
      const nextButton = screen.queryByText('Next: Review & Apply');
      if (nextButton) {
        fireEvent.click(nextButton);
      }
    });
    
    // Step 4: Compare dimensions
    const step4Dimensions = getContainerDimensions();
    expect(step4Dimensions.container?.width).toBe(step1Dimensions.container?.width);
    expect(step4Dimensions.container?.height).toBe(step1Dimensions.container?.height);
    expect(step4Dimensions.content?.width).toBe(step1Dimensions.content?.width);
    expect(step4Dimensions.content?.height).toBe(step1Dimensions.content?.height);
  });
  
  it('should maintain consistent dimensions during loading states', async () => {
    render(<ImportWizardPage />);
    
    // Get initial dimensions
    const initialDimensions = getContainerDimensions();
    
    // Trigger loading state
    const sourceInput = screen.getByPlaceholderText('Enter folder path or click Browse...');
    fireEvent.change(sourceInput, { target: { value: '/test/path' } });
    
    const importButton = screen.getByText('Import Folder');
    fireEvent.click(importButton);
    
    // Check dimensions during loading (should be same)
    const loadingDimensions = getContainerDimensions();
    expect(loadingDimensions.container?.width).toBe(initialDimensions.container?.width);
    expect(loadingDimensions.container?.height).toBe(initialDimensions.container?.height);
    expect(loadingDimensions.content?.width).toBe(initialDimensions.content?.width);
    expect(loadingDimensions.content?.height).toBe(initialDimensions.content?.height);
  });
  
  it('should log container dimensions for debugging', () => {
    render(<ImportWizardPage />);
    const dimensions = getContainerDimensions();
    
    console.log('Container Dimensions:', {
      container: dimensions.container,
      content: dimensions.content
    });
    
    // This test always passes but logs dimensions for debugging
    expect(dimensions).toBeDefined();
  });
});