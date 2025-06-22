import React, { useState, useCallback } from "react";
import styled from "styled-components";
import { FolderV2 } from "../types/types";
import {
  gatherFiles,
  groupFolders as apiGroupFolders,
  generateFolders,
  getGatherStructure,
  getGroupStructure,
  getFoldersStructure,
  saveGraph,
} from "../api";
import { applyOrganization } from "../mock_data/mockApi";
import { selectFolder } from "../utils/folderSelection";
import { FolderBrowser } from "../components/FolderBrowser";

interface ImportWizardState {
  currentStep: number;
  sourcePath: string;
  originalStructure?: FolderV2;
  groupedStructure?: FolderV2;
  organizedStructure?: FolderV2;
  targetPath: string;
  duplicateHandling: "newest" | "largest" | "both" | "both-if-different";
  isLoading: boolean;
  loadingMessage: string;
  progress: number;
  hasTriggeredGather: boolean;
  hasTriggeredGroup: boolean;
  hasTriggeredFolders: boolean;
  selectedFileId: number | null;
}

export const ImportWizardPage: React.FC = () => {
  const [state, setState] = useState<ImportWizardState>({
    currentStep: 1,
    sourcePath: "",
    targetPath: "",
    duplicateHandling: "newest",
    isLoading: false,
    loadingMessage: "",
    progress: 0,
    hasTriggeredGather: false,
    hasTriggeredGroup: false,
    hasTriggeredFolders: false,
    selectedFileId: null,
  });

  // Load existing structures on component mount
  React.useEffect(() => {
    const loadExistingStructures = async () => {
      try {
        // Check for existing gather structure
        const gatherStructure = await getGatherStructure();
        if (gatherStructure) {
          updateState({ originalStructure: gatherStructure });
        }

        // Check for existing group structure
        const groupStructure = await getGroupStructure();
        if (groupStructure) {
          updateState({ groupedStructure: groupStructure });
        }

        // Check for existing organized structure
        const foldersStructure = await getFoldersStructure();
        if (foldersStructure) {
          updateState({ organizedStructure: foldersStructure });
        }
      } catch (error) {
        console.error("Error loading existing structures:", error);
      }
    };

    loadExistingStructures();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const updateState = useCallback((updates: Partial<ImportWizardState>) => {
    setState((prev) => ({ ...prev, ...updates }));
  }, []);

  const nextStep = () => {
    if (state.currentStep < 4) {
      updateState({ currentStep: state.currentStep + 1, selectedFileId: null });
    }
  };

  const prevStep = () => {
    if (state.currentStep > 1) {
      updateState({ currentStep: state.currentStep - 1, selectedFileId: null });
    }
  };

  const goToStep = (stepNumber: number) => {
    // Only allow going to previous steps or current step
    if (stepNumber <= state.currentStep && stepNumber >= 1) {
      updateState({ currentStep: stepNumber, selectedFileId: null });
    }
  };

  const renderStep = () => {
    switch (state.currentStep) {
      case 1:
        return (
          <ImportStep
            state={state}
            updateState={updateState}
            onNext={nextStep}
          />
        );
      case 2:
        return (
          <GroupStep
            state={state}
            updateState={updateState}
            onNext={nextStep}
            onPrev={prevStep}
          />
        );
      case 3:
        return (
          <OrganizeStep
            state={state}
            updateState={updateState}
            onNext={nextStep}
            onPrev={prevStep}
          />
        );
      case 4:
        return (
          <ReviewStep
            state={state}
            updateState={updateState}
            onPrev={prevStep}
          />
        );
      default:
        return (
          <ImportStep
            state={state}
            updateState={updateState}
            onNext={nextStep}
          />
        );
    }
  };

  return (
    <WizardContainer data-testid="wizard-container">
      <WizardHeader>
        <div style={{ width: "120px", flexShrink: 0 }} />{" "}
        {/* Spacer for centering */}
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
      <WizardContent data-testid="wizard-content">{renderStep()}</WizardContent>
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
  const [abortController, setAbortController] =
    React.useState<AbortController | null>(null);

  React.useEffect(() => {
    return () => {
      if (abortController) {
        abortController.abort();
      }
    };
  }, []);

  const handleBrowseFolder = async () => {
    const result = await selectFolder();

    if (!result.success) {
      if (result.error && !result.error.includes("cancelled")) {
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
      loadingMessage: "Processing folder structure...",
      hasTriggeredGather: true,
      // Clear previous results when processing a new folder
      originalStructure: undefined,
      groupedStructure: undefined,
      organizedStructure: undefined,
      selectedFileId: null,
    });

    try {
      const taskResult = await gatherFiles(
        folderPath,
        (progress) => {
          updateState({ progress: Math.round(progress * 100) });
        },
        controller.signal
      );

      // Check if operation was cancelled
      if (controller.signal.aborted) {
        return;
      }

      // Extract folder structure from task result if available
      const folderStructure = taskResult.result?.folder_structure as
        | FolderV2
        | undefined;

      updateState({
        originalStructure: folderStructure || undefined,
        isLoading: false,
        loadingMessage: "",
        progress: 0,
      });
    } catch (error) {
      if (!controller.signal.aborted) {
        updateState({ isLoading: false, loadingMessage: "" });
        console.error("Error importing folder:", error);
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
      loadingMessage: "",
      progress: 0,
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
            if (e.key === "Enter" && state.sourcePath.trim()) {
              handleFolderSelect(state.sourcePath);
            }
          }}
        />
        <BrowseButton onClick={handleBrowseFolder} disabled={state.isLoading}>
          Browse
        </BrowseButton>
        <ProcessButton
          onClick={
            state.isLoading
              ? handleCancelImport
              : () => handleFolderSelect(state.sourcePath)
          }
          disabled={!state.sourcePath.trim() && !state.isLoading}
          $isCancel={state.isLoading}
          title={
            state.isLoading
              ? "Cancel folder processing"
              : !state.sourcePath.trim()
                ? "Select a folder first"
                : "Process selected folder"
          }
        >
          {state.isLoading ? "Cancel" : "Process"}
        </ProcessButton>
      </FolderSelectSection>

      <ContentContainer isLoading={state.isLoading}>
        {state.originalStructure ? (
          <FolderBrowser
            folderTree={state.originalStructure}
            onSelectItem={() => {}}
            externalSelectedFile={null}
            shouldSync={false}
            showConfidence={false}
          />
        ) : (
          <div
            style={{
              color: "#64748b",
              textAlign: "center",
              padding: "1.5rem 1rem",
            }}
          >
            Folder structure will appear here after import...
          </div>
        )}
        {state.isLoading && (
          <LoadingOverlay>
            <LoadingModal>
              <LoadingSpinner />
              <LoadingText>{state.loadingMessage}</LoadingText>
              {state.progress > 0 && (
                <ProgressSection>
                  <ProgressBar>
                    <ProgressFill style={{ width: `${state.progress}%` }} />
                  </ProgressBar>
                  <ProgressText>{state.progress}% complete</ProgressText>
                </ProgressSection>
              )}
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

const GroupStep: React.FC<StepProps> = ({
  state,
  updateState,
  onNext,
  onPrev,
}) => {
  const [abortController, setAbortController] =
    React.useState<AbortController | null>(null);

  React.useEffect(() => {
    return () => {
      if (abortController) {
        abortController.abort();
      }
    };
  }, []);

  const handleFileSelect = (fileId: number | null) => {
    updateState({ selectedFileId: fileId });
  };

  React.useEffect(() => {
    // Only trigger grouping if we don't have a grouped structure and haven't triggered it yet
    if (
      !state.groupedStructure &&
      state.originalStructure &&
      !state.hasTriggeredGroup
    ) {
      const controller = new AbortController();
      setAbortController(controller);

      updateState({
        isLoading: true,
        loadingMessage: "Analyzing folder similarities...",
        progress: 0,
        hasTriggeredGroup: true,
      });

      apiGroupFolders((progress) => {
        updateState({ progress: Math.round(progress * 100) });
      }, controller.signal)
        .then((taskResult) => {
          if (controller.signal.aborted) return;

          const folderStructure = taskResult.result?.folder_structure as
            | FolderV2
            | undefined;
          updateState({
            groupedStructure: folderStructure || undefined,
            isLoading: false,
            loadingMessage: "",
            progress: 0,
            selectedFileId: null,
          });
        })
        .catch((error) => {
          if (!controller.signal.aborted) {
            console.error("Error grouping folders:", error);
            updateState({ isLoading: false, loadingMessage: "", progress: 0 });
          }
        })
        .finally(() => {
          setAbortController(null);
        });
    }
  }, [
    state.groupedStructure,
    state.originalStructure,
    state.hasTriggeredGroup,
    updateState,
  ]);

  return (
    <StepContainer>
      <ComparisonView isLoading={state.isLoading}>
        <Panel>
          <SectionTitle>
            <span>Original Structure</span>
          </SectionTitle>
          {state.originalStructure ? (
            <FolderBrowser
              folderTree={state.originalStructure}
              onSelectItem={handleFileSelect}
              externalSelectedFile={state.selectedFileId}
              shouldSync={true}
              showConfidence={false}
            />
          ) : (
            <div
              style={{
                color: "#64748b",
                textAlign: "center",
                padding: "1.5rem 1rem",
              }}
            >
              Original folder structure will appear here...
            </div>
          )}
        </Panel>
        <Panel>
          <SectionTitle>
            <span>Grouped Structure</span>
          </SectionTitle>
          {state.groupedStructure ? (
            <FolderBrowser
              folderTree={state.groupedStructure}
              onSelectItem={handleFileSelect}
              externalSelectedFile={state.selectedFileId}
              shouldSync={true}
              showConfidence={true}
            />
          ) : (
            <div
              style={{
                color: "#64748b",
                textAlign: "center",
                padding: "1.5rem 1rem",
              }}
            >
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
        <WarningButton
          onClick={async () => {
            if (state.groupedStructure) {
              try {
                updateState({
                  isLoading: true,
                  loadingMessage: "Saving graph...",
                });
                await saveGraph(state.groupedStructure);
                updateState({ isLoading: false });
                // Optional: Show success message or notification
              } catch (error) {
                console.error("Failed to save graph:", error);
                updateState({ isLoading: false });
                // Optional: Show error message to user
              }
            }
          }}
          disabled={state.isLoading || !state.groupedStructure}
        >
          Save
        </WarningButton>
        <SuccessButton
          onClick={onNext}
          disabled={state.isLoading || !state.groupedStructure}
        >
          Next
        </SuccessButton>
      </ButtonRow>
    </StepContainer>
  );
};

const OrganizeStep: React.FC<StepProps> = ({
  state,
  updateState,
  onNext,
  onPrev,
}) => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [abortController, setAbortController] =
    React.useState<AbortController | null>(null);

  React.useEffect(() => {
    return () => {
      if (abortController) {
        abortController.abort();
      }
    };
  }, []);

  const handleFileSelect = (fileId: number | null) => {
    updateState({ selectedFileId: fileId });
  };

  React.useEffect(() => {
    // Only trigger folder generation if we don't have an organized structure and haven't triggered it yet
    if (
      !state.organizedStructure &&
      state.groupedStructure &&
      !state.hasTriggeredFolders
    ) {
      const controller = new AbortController();
      setAbortController(controller);

      updateState({
        isLoading: true,
        loadingMessage: "Generating final organization...",
        progress: 0,
        hasTriggeredFolders: true,
      });

      generateFolders((progress) => {
        updateState({ progress: Math.round(progress * 100) });
      }, controller.signal)
        .then((taskResult) => {
          if (controller.signal.aborted) return;

          const folderStructure = taskResult.result?.folder_structure as
            | FolderV2
            | undefined;
          updateState({
            organizedStructure: folderStructure || undefined,
            isLoading: false,
            loadingMessage: "",
            progress: 0,
            selectedFileId: null,
          });
        })
        .catch((error) => {
          if (!controller.signal.aborted) {
            console.error("Error organizing folders:", error);
            updateState({ isLoading: false, loadingMessage: "", progress: 0 });
          }
        })
        .finally(() => {
          setAbortController(null);
        });
    }
  }, [
    state.organizedStructure,
    state.groupedStructure,
    state.hasTriggeredFolders,
    updateState,
  ]);

  return (
    <StepContainer>
      <ComparisonView isLoading={state.isLoading}>
        <Panel>
          <SectionTitle>
            <span>Grouped Structure</span>
          </SectionTitle>
          {state.groupedStructure ? (
            <FolderBrowser
              folderTree={state.groupedStructure}
              onSelectItem={handleFileSelect}
              externalSelectedFile={state.selectedFileId}
              shouldSync={true}
              showConfidence={true}
            />
          ) : (
            <div
              style={{
                color: "#64748b",
                textAlign: "center",
                padding: "1.5rem 1rem",
              }}
            >
              Grouped structure will appear here...
            </div>
          )}
        </Panel>
        <Panel>
          <SectionTitle>
            <span>Final Organization</span>
          </SectionTitle>
          {state.organizedStructure ? (
            <FolderBrowser
              folderTree={state.organizedStructure}
              onSelectItem={handleFileSelect}
              externalSelectedFile={state.selectedFileId}
              shouldSync={true}
              showConfidence={false}
            />
          ) : (
            <div
              style={{
                color: "#64748b",
                textAlign: "center",
                padding: "1.5rem 1rem",
              }}
            >
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
        <WarningButton
          onClick={() => {
            updateState({
              hasTriggeredFolders: false,
              organizedStructure: undefined,
              selectedFileId: null,
            });
          }}
          disabled={state.isLoading}
        >
          Re-run Organization
        </WarningButton>
        <SuccessButton
          onClick={onNext}
          disabled={state.isLoading || !state.organizedStructure}
        >
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
      if (result.error && !result.error.includes("cancelled")) {
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
        setProgress((prev) => {
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
      console.error("Error applying organization:", error);
      setIsApplying(false);
      setProgress(0);
      alert("Error applying organization. Please try again.");
    }
  };

  return (
    <StepContainer>
      <ComparisonView>
        <Panel>
          <SectionTitle>
            <span>Configuration Settings</span>
          </SectionTitle>
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
              onChange={(e) =>
                updateState({
                  duplicateHandling: e.target
                    .value as ImportWizardState["duplicateHandling"],
                })
              }
            >
              <option value="newest">Keep Newest</option>
              <option value="largest">Keep Largest</option>
              <option value="both">Keep Both</option>
              <option value="both-if-different">
                Keep Both if Not Identical
              </option>
            </Select>
          </SettingGroup>
        </Panel>
        <Panel>
          {state.organizedStructure && (
            <FolderBrowser
              folderTree={state.organizedStructure}
              onSelectItem={() => {}}
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
        <DangerButton
          onClick={handleApply}
          disabled={isApplying || !state.targetPath}
        >
          {isApplying ? "Applying..." : "Apply Organization"}
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
  padding: 0.25rem;
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
  margin-bottom: 0.5rem;
  flex-shrink: 0;
  height: 48px;
  position: relative;
  min-width: 0;
`;

const Title = styled.h1`
  font-size: 1.5rem;
  font-weight: 700;
  color: #1f2937;
  margin: 0;
  padding-top: 0;
  line-height: 1.1;
  flex: 1;
  text-align: center;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const StepIndicator = styled.div`
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-shrink: 0;
`;

const StepItem = styled.div.withConfig({
  shouldForwardProp: (prop) =>
    !["active", "completed", "clickable"].includes(prop),
})<{ active: boolean; completed: boolean; clickable: boolean }>`
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  transition: all 0.2s ease;
  border: 1px solid transparent;
  cursor: ${(props) =>
    props.clickable
      ? "pointer"
      : props.clickable === false
        ? "not-allowed"
        : "default"};
  user-select: none;
  opacity: ${(props) => (props.clickable ? 1 : 0.6)};

  // Background and text colors
  ${(props) => {
    if (props.active) {
      return `
        background-color: #2563eb;
        color: white;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
        border-color: #1d4ed8;
      `;
    } else if (props.completed) {
      return `
        background-color: #10b981;
        color: white;
      `;
    } else {
      return `
        background-color: #e2e8f0;
        color: #64748b;
      `;
    }
  }}

  // Hover effects for clickable items
  ${(props) =>
    props.clickable &&
    !props.active &&
    `
    &:hover {
      transform: translateY(-1px);
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
      background-color: ${props.completed ? "#059669" : "#cbd5e1"};
    }
  `}
`;

const WizardContent = styled.div`
  background: white;
  border-radius: 0.375rem;
  padding: 0.75rem;
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
  gap: 0.75rem;
  margin-bottom: 1rem;
  align-items: stretch;
`;

const FolderInput = styled.input`
  flex: 1;
  padding: 0.5rem;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-family: inherit;

  &:focus {
    outline: none;
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
  }
`;

// Base button styles
const BaseButton = styled.button`
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 0.375rem;
  cursor: pointer;
  font-size: 0.875rem;
  font-weight: 500;
  white-space: nowrap;
  transition: background-color 0.2s ease;

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const BrowseButton = styled(BaseButton)`
  background-color: #6b7280;
  color: white;

  &:hover:not(:disabled) {
    background-color: #4b5563;
  }
`;

const ProcessButton = styled(BaseButton).withConfig({
  shouldForwardProp: (prop) => !["$isCancel"].includes(prop),
})<{ $isCancel?: boolean }>`
  background-color: ${(props) => (props.$isCancel ? "#ef4444" : "#2563eb")};
  color: white;

  &:hover:not(:disabled) {
    background-color: ${(props) => (props.$isCancel ? "#dc2626" : "#1d4ed8")};
  }
`;

// Loading components
const LoadingOverlay = styled.div`
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  border-radius: 0.75rem;
`;

const LoadingModal = styled.div`
  background: white;
  padding: 1.5rem 2rem;
  border-radius: 0.75rem;
  box-shadow:
    0 0 0 1px rgba(0, 0, 0, 0.05),
    0 10px 15px -3px rgba(0, 0, 0, 0.2);
  border: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  min-width: 280px;
  max-width: 350px;
  text-align: center;
`;

const LoadingSpinner = styled.div`
  width: 32px;
  height: 32px;
  border: 3px solid #e5e7eb;
  border-top: 3px solid #2563eb;
  border-radius: 50%;
  animation: spin 1s linear infinite;

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

const LoadingText = styled.div`
  color: #374151;
  font-weight: 500;
  font-size: 1rem;
`;

// Base container with loading state
const BaseContainer = styled.div.withConfig({
  shouldForwardProp: (prop) => !["isLoading"].includes(prop),
})<{ isLoading?: boolean }>`
  padding: 0.5rem;
  border: 1px solid #e2e8f0;
  border-radius: 0.375rem;
  background: linear-gradient(to bottom, #ffffff, #f8fafc);
  box-shadow:
    0 0 0 1px rgba(0, 0, 0, 0.05),
    0 2px 4px -1px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease;
  position: relative;
  opacity: ${(props) => (props.isLoading ? 0.5 : 1)};
  pointer-events: ${(props) => (props.isLoading ? "none" : "auto")};
  overflow-y: auto;
  flex: 1;
  min-height: 0;

  &:hover {
    border-color: #cbd5e1;
    box-shadow:
      0 0 0 1px rgba(0, 0, 0, 0.05),
      0 4px 12px -2px rgba(0, 0, 0, 0.1);
  }
`;

const ContentContainer = styled(BaseContainer)``;

const ComparisonView = styled.div.withConfig({
  shouldForwardProp: (prop) => !["isLoading"].includes(prop),
})<{ isLoading?: boolean }>`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  position: relative;
  opacity: ${(props) => (props.isLoading ? 0.5 : 1)};
  pointer-events: ${(props) => (props.isLoading ? "none" : "auto")};
  transition: opacity 0.2s ease;
  flex: 1;
  min-height: 0;
`;

const SectionTitle = styled.h4`
  font-weight: 700;
  margin: 0;
  color: #0f172a;
  font-size: 0.875rem;
  line-height: 1.2;
  position: relative;
  text-align: center;
  margin-bottom: 0.5rem;

  &::before,
  &::after {
    content: "";
    position: absolute;
    top: 50%;
    width: calc(50% - 0.75rem);
    height: 1px;
    background-color: #f1f5f9;
    opacity: 0.6;
  }

  &::before {
    left: 0;
  }

  &::after {
    right: 0;
  }

  span {
    background: #fafbfc;
    padding: 0 0.5rem;
    position: relative;
    z-index: 1;
    transition: background-color 0.2s ease;
  }
`;

const Panel = styled(BaseContainer)`
  background: #fafbfc;
  display: flex;
  flex-direction: column;
`;

const ButtonRow = styled.div`
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
  padding: 0.75rem 0;
  border-top: 1px solid #e2e8f0;
  flex-wrap: wrap;
  flex-shrink: 0;
  min-height: 56px;
  align-items: center;
`;

// Large button base for wizard navigation
const LargeButton = styled(BaseButton)`
  padding: 0.75rem 2rem;
  font-size: 1rem;
`;

const SuccessButton = styled(LargeButton)`
  background-color: #10b981;
  color: white;

  &:hover:not(:disabled) {
    background-color: #059669;
  }
`;

const SecondaryButton = styled(LargeButton)`
  background-color: #6b7280;
  color: white;

  &:hover:not(:disabled) {
    background-color: #4b5563;
  }
`;

const WarningButton = styled(LargeButton)`
  background-color: #f59e0b;
  color: white;

  &:hover:not(:disabled) {
    background-color: #d97706;
  }
`;

const DangerButton = styled(LargeButton)`
  background-color: #dc2626;
  color: white;

  &:hover:not(:disabled) {
    background-color: #b91c1c;
  }
`;

const SettingGroup = styled.div`
  margin-bottom: 0.75rem;
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

const ProgressSection = styled(BaseContainer)`
  margin: 1rem 0;
  padding: 1rem;
  border-radius: 0.5rem;
  flex: none;
`;

// Progress components
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
