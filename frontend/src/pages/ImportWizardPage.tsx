import React, { useState } from "react";
import styled from "styled-components";
import { FolderV2 } from "../types/types";
import { importFolder, groupFolders, organizeFolders, applyOrganization } from "../mock_data/mockApi";
import { selectFolder } from "../utils/folderSelection";
import { FolderBrowser, FolderBrowserViewType } from "../components/FolderBrowser";

interface ImportWizardState {
  currentStep: number;
  sourcePath: string;
  originalStructure?: FolderV2;
  groupedStructure?: FolderV2;
  organizedStructure?: FolderV2;
  targetPath: string;
  duplicateHandling: 'newest' | 'largest' | 'both' | 'both-if-different';
  isLoading: boolean;
  loadingMessage: string;
}

export const ImportWizardPage: React.FC = () => {
  const [state, setState] = useState<ImportWizardState>({
    currentStep: 1,
    sourcePath: '',
    targetPath: '',
    duplicateHandling: 'newest',
    isLoading: false,
    loadingMessage: ''
  });

  const updateState = (updates: Partial<ImportWizardState>) => {
    setState(prev => ({ ...prev, ...updates }));
  };

  const nextStep = () => {
    if (state.currentStep < 4) {
      updateState({ currentStep: state.currentStep + 1 });
    }
  };

  const prevStep = () => {
    if (state.currentStep > 1) {
      updateState({ currentStep: state.currentStep - 1 });
    }
  };

  const goToStep = (stepNumber: number) => {
    // Only allow going to previous steps or current step
    if (stepNumber <= state.currentStep && stepNumber >= 1) {
      updateState({ currentStep: stepNumber });
    }
  };

  const renderStep = () => {
    switch (state.currentStep) {
      case 1:
        return <ImportStep state={state} updateState={updateState} onNext={nextStep} />;
      case 2:
        return <GroupStep state={state} updateState={updateState} onNext={nextStep} onPrev={prevStep} />;
      case 3:
        return <OrganizeStep state={state} updateState={updateState} onNext={nextStep} onPrev={prevStep} />;
      case 4:
        return <ReviewStep state={state} updateState={updateState} onPrev={prevStep} />;
      default:
        return <ImportStep state={state} updateState={updateState} onNext={nextStep} />;
    }
  };

  return (
    <WizardContainer data-testid="wizard-container">
      <WizardHeader>
        <Title>Import Wizard</Title>
        <StepIndicator>
          <StepItem 
            active={state.currentStep === 1} 
            completed={state.currentStep > 1}
            clickable={1 <= state.currentStep}
            onClick={() => goToStep(1)}
          >
            1. Import
          </StepItem>
          <StepItem 
            active={state.currentStep === 2} 
            completed={state.currentStep > 2}
            clickable={2 <= state.currentStep}
            onClick={() => goToStep(2)}
          >
            2. Group
          </StepItem>
          <StepItem 
            active={state.currentStep === 3} 
            completed={state.currentStep > 3}
            clickable={3 <= state.currentStep}
            onClick={() => goToStep(3)}
          >
            3. Organize
          </StepItem>
          <StepItem 
            active={state.currentStep === 4} 
            completed={false}
            clickable={4 <= state.currentStep}
            onClick={() => goToStep(4)}
          >
            4. Review
          </StepItem>
        </StepIndicator>
      </WizardHeader>
      <WizardContent data-testid="wizard-content">
        {renderStep()}
      </WizardContent>
    </WizardContainer>
  );
};

interface StepProps {
  state: ImportWizardState;
  updateState: (updates: Partial<ImportWizardState>) => void;
  onNext?: () => void;
  onPrev?: () => void;
}

const ImportStep: React.FC<StepProps> = ({ state, updateState, onNext }) => {
  const [abortController, setAbortController] = React.useState<AbortController | null>(null);

  const handleBrowseFolder = async () => {
    const result = await selectFolder();
    
    if (!result.success) {
      if (result.error && !result.error.includes('cancelled')) {
        alert(result.error);
      }
      return;
    }
    
    if (result.path) {
      updateState({ sourcePath: result.path });
    }
  };

  const handleFolderSelect = async (folderPath: string) => {
    if (!folderPath.trim()) return;
    
    // Create abort controller for this import operation
    const controller = new AbortController();
    setAbortController(controller);
    
    updateState({ 
      sourcePath: folderPath,
      isLoading: true, 
      loadingMessage: 'Processing folder structure...',
      // Clear previous results when processing a new folder
      originalStructure: undefined,
      groupedStructure: undefined,
      organizedStructure: undefined
    });
    
    try {
      const importedStructure = await importFolder(folderPath);
      
      // Check if operation was cancelled
      if (controller.signal.aborted) {
        return;
      }
      
      updateState({ 
        originalStructure: importedStructure,
        isLoading: false,
        loadingMessage: ''
      });
    } catch (error) {
      if (!controller.signal.aborted) {
        updateState({ isLoading: false, loadingMessage: '' });
        console.error('Error importing folder:', error);
      }
    } finally {
      setAbortController(null);
    }
  };

  const handleCancelImport = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
    }
    updateState({ 
      isLoading: false, 
      loadingMessage: ''
    });
  };

  return (
    <StepContainer>      
      <FolderSelectSection>
        <FolderInput
          type="text"
          placeholder="Select folder or enter path..."
          value={state.sourcePath}
          onChange={(e) => updateState({ sourcePath: e.target.value })}
          disabled={state.isLoading}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && state.sourcePath.trim()) {
              handleFolderSelect(state.sourcePath);
            }
          }}
        />
        <BrowseButton 
          onClick={handleBrowseFolder}
          disabled={state.isLoading}
        >
          Browse
        </BrowseButton>
        <ProcessButton 
          onClick={state.isLoading ? handleCancelImport : () => handleFolderSelect(state.sourcePath)}
          disabled={!state.sourcePath.trim() && !state.isLoading}
          $isCancel={state.isLoading}
          title={
            state.isLoading ? "Cancel folder processing" : 
            !state.sourcePath.trim() ? "Select a folder first" : 
            "Process selected folder"
          }
        >
          {state.isLoading ? 'Cancel' : 'Process'}
        </ProcessButton>
      </FolderSelectSection>


      <ContentContainer isLoading={state.isLoading}>
        {state.originalStructure ? (
          <FolderBrowser
            folderViewResponse={{ original: state.originalStructure, new: state.originalStructure }}
            onSelectItem={() => {}}
            viewType={FolderBrowserViewType.ORIGINAL}
            externalSelectedFile={null}
            shouldSync={false}
            showConfidence={false}
          />
        ) : (
          <div style={{ color: '#64748b', textAlign: 'center', padding: '3rem' }}>
            Folder structure will appear here after import...
          </div>
        )}
        {state.isLoading && (
          <LoadingOverlay>
            <LoadingModal>
              <LoadingSpinner />
              <LoadingText>{state.loadingMessage}</LoadingText>
            </LoadingModal>
          </LoadingOverlay>
        )}
      </ContentContainer>
      
      {state.originalStructure && (
        <ButtonRow>
          <SuccessButton onClick={onNext}>Next</SuccessButton>
        </ButtonRow>
      )}
    </StepContainer>
  );
};

const GroupStep: React.FC<StepProps> = ({ state, updateState, onNext, onPrev }) => {
  React.useEffect(() => {
    if (!state.groupedStructure && state.originalStructure) {
      updateState({ isLoading: true, loadingMessage: 'Analyzing folder similarities...' });
      
      groupFolders(state.originalStructure)
        .then(groupedStructure => {
          updateState({ 
            groupedStructure,
            isLoading: false,
            loadingMessage: ''
          });
        })
        .catch(error => {
          console.error('Error grouping folders:', error);
          updateState({ isLoading: false, loadingMessage: '' });
        });
    }
  }, [state.groupedStructure, state.originalStructure, updateState]);

  return (
    <StepContainer>
      <ComparisonView isLoading={state.isLoading}>
        <Panel>
          <SectionTitle>Original Structure</SectionTitle>
          {state.originalStructure ? (
            <FolderBrowser
              folderViewResponse={{ original: state.originalStructure, new: state.originalStructure }}
              onSelectItem={() => {}}
              viewType={FolderBrowserViewType.ORIGINAL}
              externalSelectedFile={null}
              shouldSync={false}
              showConfidence={false}
            />
          ) : (
            <div style={{ color: '#64748b', textAlign: 'center', padding: '3rem' }}>
              Original folder structure will appear here...
            </div>
          )}
        </Panel>
        <Panel>
          <SectionTitle>Grouped Structure</SectionTitle>
          {state.groupedStructure ? (
            <FolderBrowser
              folderViewResponse={{ original: state.groupedStructure, new: state.groupedStructure }}
              onSelectItem={() => {}}
              viewType={FolderBrowserViewType.ORIGINAL}
              externalSelectedFile={null}
              shouldSync={false}
              showConfidence={true}
            />
          ) : (
            <div style={{ color: '#64748b', textAlign: 'center', padding: '3rem' }}>
              Grouped structure will appear here after processing...
            </div>
          )}
        </Panel>
      </ComparisonView>
      {state.isLoading && (
        <LoadingOverlay>
          <LoadingModal>
            <LoadingSpinner />
            <LoadingText>{state.loadingMessage}</LoadingText>
          </LoadingModal>
        </LoadingOverlay>
      )}

      <ButtonRow>
        <SecondaryButton onClick={onPrev}>Previous</SecondaryButton>
        <WarningButton onClick={() => alert('Grouping saved!')}>Save Changes</WarningButton>
        <SuccessButton onClick={onNext} disabled={state.isLoading}>
          Next
        </SuccessButton>
      </ButtonRow>
    </StepContainer>
  );
};

const OrganizeStep: React.FC<StepProps> = ({ state, updateState, onNext, onPrev }) => {
  React.useEffect(() => {
    if (!state.organizedStructure && state.groupedStructure) {
      updateState({ isLoading: true, loadingMessage: 'Generating final organization...' });
      
      organizeFolders(state.groupedStructure)
        .then(organizedStructure => {
          updateState({ 
            organizedStructure,
            isLoading: false,
            loadingMessage: ''
          });
        })
        .catch(error => {
          console.error('Error organizing folders:', error);
          updateState({ isLoading: false, loadingMessage: '' });
        });
    }
  }, [state.organizedStructure, state.groupedStructure, updateState]);

  return (
    <StepContainer>
      <ComparisonView isLoading={state.isLoading}>
        <Panel>
          <SectionTitle>Grouped Structure</SectionTitle>
          {state.groupedStructure ? (
            <FolderBrowser
              folderViewResponse={{ original: state.groupedStructure, new: state.groupedStructure }}
              onSelectItem={() => {}}
              viewType={FolderBrowserViewType.ORIGINAL}
              externalSelectedFile={null}
              shouldSync={false}
              showConfidence={true}
            />
          ) : (
            <div style={{ color: '#64748b', textAlign: 'center', padding: '3rem' }}>
              Grouped structure will appear here...
            </div>
          )}
        </Panel>
        <Panel>
          <SectionTitle>Final Organization</SectionTitle>
          {state.organizedStructure ? (
            <FolderBrowser
              folderViewResponse={{ original: state.organizedStructure, new: state.organizedStructure }}
              onSelectItem={() => {}}
              viewType={FolderBrowserViewType.ORIGINAL}
              externalSelectedFile={null}
              shouldSync={false}
              showConfidence={false}
            />
          ) : (
            <div style={{ color: '#64748b', textAlign: 'center', padding: '3rem' }}>
              Final organization will appear here after processing...
            </div>
          )}
        </Panel>
      </ComparisonView>
      {state.isLoading && (
        <LoadingOverlay>
          <LoadingModal>
            <LoadingSpinner />
            <LoadingText>{state.loadingMessage}</LoadingText>
          </LoadingModal>
        </LoadingOverlay>
      )}

      <ButtonRow>
        <SecondaryButton onClick={onPrev}>Previous</SecondaryButton>
        <WarningButton onClick={() => alert('Organization saved!')}>Save Changes</WarningButton>
        <SuccessButton onClick={onNext} disabled={state.isLoading}>
          Next
        </SuccessButton>
      </ButtonRow>
    </StepContainer>
  );
};

const ReviewStep: React.FC<StepProps> = ({ state, updateState, onPrev }) => {
  const [isApplying, setIsApplying] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleBrowseTargetFolder = async () => {
    const result = await selectFolder();
    
    if (!result.success) {
      if (result.error && !result.error.includes('cancelled')) {
        alert(result.error);
      }
      return;
    }
    
    if (result.path) {
      updateState({ targetPath: result.path });
    }
  };

  const handleApply = async () => {
    if (!state.organizedStructure || !state.targetPath) return;
    
    setIsApplying(true);
    setProgress(0);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 300);

      const result = await applyOrganization(
        state.organizedStructure, 
        state.targetPath, 
        state.duplicateHandling
      );

      clearInterval(progressInterval);
      setProgress(100);
      
      setTimeout(() => {
        alert(result.message);
        setIsApplying(false);
        setProgress(0);
      }, 500);
    } catch (error) {
      console.error('Error applying organization:', error);
      setIsApplying(false);
      setProgress(0);
      alert('Error applying organization. Please try again.');
    }
  };

  return (
    <StepContainer>
      <ComparisonView>
        <Panel>
          <SectionTitle>Configuration Settings</SectionTitle>
          <SettingGroup>
            <SettingLabel>Target Folder:</SettingLabel>
            <FolderSelectSection>
              <FolderInput
                type="text"
                placeholder="Choose destination folder..."
                value={state.targetPath}
                onChange={(e) => updateState({ targetPath: e.target.value })}
              />
              <BrowseButton onClick={handleBrowseTargetFolder}>
                Browse
              </BrowseButton>
            </FolderSelectSection>
          </SettingGroup>

          <SettingGroup>
            <SettingLabel>Duplicate Handling:</SettingLabel>
            <Select
              value={state.duplicateHandling}
              onChange={(e) => updateState({ duplicateHandling: e.target.value as any })}
            >
              <option value="newest">Keep Newest</option>
              <option value="largest">Keep Largest</option>
              <option value="both">Keep Both</option>
              <option value="both-if-different">Keep Both if Not Identical</option>
            </Select>
          </SettingGroup>
        </Panel>
        <Panel>
          {state.organizedStructure && (
            <FolderBrowser
              folderViewResponse={{ original: state.organizedStructure, new: state.organizedStructure }}
              onSelectItem={() => {}}
              viewType={FolderBrowserViewType.ORIGINAL}
              externalSelectedFile={null}
              shouldSync={false}
              showConfidence={false}
            />
          )}
        </Panel>
      </ComparisonView>

      {isApplying && (
        <ProgressSection>
          <ProgressBar>
            <ProgressFill style={{ width: `${progress}%` }} />
          </ProgressBar>
          <ProgressText>Applying organization... {progress}%</ProgressText>
        </ProgressSection>
      )}

      <ButtonRow>
        <SecondaryButton onClick={onPrev} disabled={isApplying}>
          Previous
        </SecondaryButton>
        <DangerButton onClick={handleApply} disabled={isApplying || !state.targetPath}>
          {isApplying ? 'Applying...' : 'Apply Organization'}
        </DangerButton>
      </ButtonRow>
    </StepContainer>
  );
};


// Styled Components
const WizardContainer = styled.div`
  width: 100vw;
  height: 100vh;
  margin: 0;
  padding: 1.5rem;
  background-color: #f8fafc;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
`;

const WizardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  flex-shrink: 0;
  height: 60px;
`;

const Title = styled.h1`
  font-size: 2rem;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
`;

const StepIndicator = styled.div`
  display: flex;
  gap: 1rem;
  align-items: center;
`;

const StepItem = styled.div<{ active: boolean; completed: boolean; clickable: boolean }>`
  padding: 0.75rem 1.5rem;
  border-radius: 0.75rem;
  font-weight: 600;
  font-size: 0.95rem;
  transition: all 0.2s ease;
  border: 2px solid transparent;
  cursor: ${props => props.clickable ? 'pointer' : 'default'};
  user-select: none;
  background-color: ${props => 
    props.active ? '#2563eb' : 
    props.completed ? '#10b981' : '#e2e8f0'
  };
  color: ${props => 
    props.active || props.completed ? 'white' : '#64748b'
  };
  opacity: ${props => props.clickable ? 1 : 0.6};
  
  ${props => props.active && `
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
    border-color: #1d4ed8;
  `}
  
  ${props => props.clickable && !props.active && `
    &:hover {
      transform: translateY(-1px);
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      ${props.completed ? `
        background-color: #059669;
      ` : `
        background-color: #cbd5e1;
      `}
    }
  `}
  
  ${props => !props.clickable && `
    cursor: not-allowed;
  `}
`;

const WizardContent = styled.div`
  background: white;
  border-radius: 0.5rem;
  padding: 1.5rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  position: relative;
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-height: 0;
`;

const StepContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
  min-height: 0;
`;



const FolderSelectSection = styled.div`
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  align-items: stretch;
`;

const FolderInput = styled.input`
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 1rem;
  font-family: inherit;
  
  &:focus {
    outline: none;
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
  }
`;

const BrowseButton = styled.button`
  padding: 0.75rem 1.5rem;
  background-color: #6b7280;
  color: white;
  border: none;
  border-radius: 0.375rem;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  white-space: nowrap;
  
  &:hover:not(:disabled) {
    background-color: #4b5563;
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const ProcessButton = styled.button.withConfig({
  shouldForwardProp: (prop) => !['$isCancel'].includes(prop)
})<{ $isCancel?: boolean }>`
  padding: 0.75rem 1.5rem;
  background-color: ${props => props.$isCancel ? '#ef4444' : '#2563eb'};
  color: white;
  border: none;
  border-radius: 0.375rem;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  white-space: nowrap;
  transition: background-color 0.2s ease;
  
  &:hover:not(:disabled) {
    background-color: ${props => props.$isCancel ? '#dc2626' : '#1d4ed8'};
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;




// New loading overlay components
const LoadingOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  border-radius: 0.75rem;
`;

const LoadingModal = styled.div`
  background: white;
  padding: 2rem 3rem;
  border-radius: 1rem;
  box-shadow: 
    0 0 0 1px rgba(0, 0, 0, 0.05),
    0 20px 25px -5px rgba(0, 0, 0, 0.2),
    0 10px 10px -5px rgba(0, 0, 0, 0.1);
  border: 2px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
  min-width: 300px;
  max-width: 400px;
  text-align: center;
`;


const LoadingSpinner = styled.div`
  width: 40px;
  height: 40px;
  border: 4px solid #e5e7eb;
  border-top: 4px solid #2563eb;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const LoadingText = styled.div`
  color: #374151;
  font-weight: 500;
  font-size: 1.1rem;
  line-height: 1.5;
`;

// Standardized content containers
const ContentContainer = styled.div.withConfig({
  shouldForwardProp: (prop) => !['isLoading'].includes(prop)
})<{ isLoading?: boolean }>`
  padding: 1.5rem;
  border: 2px solid #e2e8f0;
  border-radius: 0.75rem;
  background: linear-gradient(to bottom, #ffffff, #f8fafc);
  box-shadow: 
    0 0 0 1px rgba(0, 0, 0, 0.05),
    0 4px 6px -1px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease;
  position: relative;
  opacity: ${props => props.isLoading ? 0.5 : 1};
  pointer-events: ${props => props.isLoading ? 'none' : 'auto'};
  overflow-y: auto;
  flex: 1;
  min-height: 0;
  
  &:hover {
    border-color: #cbd5e1;
    box-shadow: 
      0 0 0 1px rgba(0, 0, 0, 0.05),
      0 8px 25px -5px rgba(0, 0, 0, 0.1);
  }
`;

// Remove PreviewTitle - will use SectionTitle instead

const SuccessButton = styled.button`
  padding: 0.75rem 2rem;
  background-color: #10b981;
  color: white;
  border: none;
  border-radius: 0.375rem;
  font-weight: 500;
  cursor: pointer;
  font-size: 1rem;
  
  &:hover:not(:disabled) {
    background-color: #059669;
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const ComparisonView = styled.div.withConfig({
  shouldForwardProp: (prop) => !['isLoading'].includes(prop)
})<{ isLoading?: boolean }>`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  position: relative;
  opacity: ${props => props.isLoading ? 0.5 : 1};
  pointer-events: ${props => props.isLoading ? 'none' : 'auto'};
  transition: opacity 0.2s ease;
  flex: 1;
  min-height: 0;
`;

const Panel = styled.div`
  padding: 1.5rem;
  border: 2px solid #e2e8f0;
  border-radius: 0.75rem;
  background: linear-gradient(to bottom, #ffffff, #f8fafc);
  box-shadow: 
    0 0 0 1px rgba(0, 0, 0, 0.05),
    0 4px 6px -1px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  min-height: 0;
  
  &:hover {
    border-color: #cbd5e1;
    box-shadow: 
      0 0 0 1px rgba(0, 0, 0, 0.05),
      0 8px 25px -5px rgba(0, 0, 0, 0.1);
  }
`;

const SectionTitle = styled.h4`
  font-weight: 700;
  margin-bottom: 1rem;
  color: #0f172a;
  font-size: 1.25rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #e2e8f0;
`;

const ButtonRow = styled.div`
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
  padding-top: 1rem;
  border-top: 1px solid #e2e8f0;
  flex-wrap: wrap;
  flex-shrink: 0;
  height: 60px;
  align-items: center;
`;

const SecondaryButton = styled.button`
  padding: 0.75rem 2rem;
  background-color: #6b7280;
  color: white;
  border: none;
  border-radius: 0.375rem;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  
  &:hover:not(:disabled) {
    background-color: #4b5563;
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const WarningButton = styled.button`
  padding: 0.75rem 2rem;
  background-color: #f59e0b;
  color: white;
  border: none;
  border-radius: 0.375rem;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  
  &:hover:not(:disabled) {
    background-color: #d97706;
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;



const SettingGroup = styled.div`
  margin-bottom: 1rem;
`;

const SettingLabel = styled.label`
  display: block;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #1f2937;
  font-size: 1rem;
`;

const Select = styled.select`
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 1rem;
  font-family: inherit;
  background-color: white;
  
  &:focus {
    outline: none;
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
  }
`;

// Remove FinalPreview - will use ContentContainer instead

const ProgressSection = styled.div`
  margin: 1.5rem 0;
  padding: 1.5rem;
  background: linear-gradient(to bottom, #ffffff, #f8fafc);
  border-radius: 0.75rem;
  border: 2px solid #e2e8f0;
  box-shadow: 
    0 0 0 1px rgba(0, 0, 0, 0.05),
    0 4px 6px -1px rgba(0, 0, 0, 0.1);
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 8px;
  background-color: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
`;

const ProgressFill = styled.div`
  height: 100%;
  background-color: #10b981;
  transition: width 0.3s ease;
`;

const ProgressText = styled.div`
  text-align: center;
  margin-top: 0.5rem;
  color: #6b7280;
  font-weight: 500;
`;

const DangerButton = styled.button`
  padding: 0.75rem 2rem;
  background-color: #dc2626;
  color: white;
  border: none;
  border-radius: 0.375rem;
  font-weight: 500;
  cursor: pointer;
  font-size: 1rem;
  
  &:hover:not(:disabled) {
    background-color: #b91c1c;
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

