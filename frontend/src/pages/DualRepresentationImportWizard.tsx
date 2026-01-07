/**
 * DualRepresentationImportWizard - New version of import wizard using dual representation.
 *
 * This is the feature-flagged version that uses the v2 API with dual representation.
 * It provides a simplified view showing both node and category hierarchies side-by-side.
 */

import React, { useState, useCallback, useEffect } from "react";
import styled from "styled-components";
import { gatherFiles, groupFolders as apiGroupFolders } from "../api";
import { selectFolder } from "../utils/folderSelection";
import { HierarchyBrowser } from "../components/HierarchyBrowser";
import { useDualRepresentation } from "../hooks/useDualRepresentation";
import { DualRepresentation, HierarchyItem } from "../types/types";

interface WizardState {
  currentStep: number;
  sourcePath: string;
  isLoading: boolean;
  loadingMessage: string;
  progress: number;
  hasGathered: boolean;
  hasGrouped: boolean;
}

export const DualRepresentationImportWizard: React.FC = () => {
  const [state, setState] = useState<WizardState>({
    currentStep: 1,
    sourcePath: "",
    isLoading: false,
    loadingMessage: "",
    progress: 0,
    hasGathered: false,
    hasGrouped: false,
  });

  const {
    dualRep,
    isLoading: isDualRepLoading,
    error: dualRepError,
    highlightedItemId,
    fetchDualRepresentation,
    highlightItem,
  } = useDualRepresentation();

  const updateState = useCallback((updates: Partial<WizardState>) => {
    setState((prev) => ({ ...prev, ...updates }));
  }, []);

  const nextStep = () => {
    if (state.currentStep < 2) {
      updateState({ currentStep: state.currentStep + 1 });
    }
  };

  const prevStep = () => {
    if (state.currentStep > 1) {
      updateState({ currentStep: state.currentStep - 1 });
    }
  };

  const renderStep = () => {
    switch (state.currentStep) {
      case 1:
        return (
          <GatherStep
            state={state}
            updateState={updateState}
            onNext={nextStep}
            fetchDualRep={fetchDualRepresentation}
          />
        );
      case 2:
        return (
          <ViewStep
            dualRep={dualRep}
            isDualRepLoading={isDualRepLoading}
            dualRepError={dualRepError}
            highlightedItemId={highlightedItemId}
            onHighlight={highlightItem}
            onPrev={prevStep}
          />
        );
      default:
        return null;
    }
  };

  return (
    <WizardContainer>
      <WizardHeader>
        <div style={{ width: "120px", flexShrink: 0 }} />
        <Title>Import Wizard (Dual Representation)</Title>
        <FeatureBadge>EXPERIMENTAL</FeatureBadge>
        <StepIndicator>
          <StepItem
            active={state.currentStep === 1}
            completed={state.currentStep > 1}
          >
            1. Gather & Group
          </StepItem>
          <StepItem active={state.currentStep === 2} completed={false}>
            2. View Results
          </StepItem>
        </StepIndicator>
      </WizardHeader>
      <WizardContent>{renderStep()}</WizardContent>
    </WizardContainer>
  );
};

interface GatherStepProps {
  state: WizardState;
  updateState: (updates: Partial<WizardState>) => void;
  onNext: () => void;
  fetchDualRep: () => Promise<void>;
}

const GatherStep: React.FC<GatherStepProps> = ({
  state,
  updateState,
  onNext,
  fetchDualRep,
}) => {
  const [abortController, setAbortController] =
    useState<AbortController | null>(null);

  useEffect(() => {
    return () => {
      if (abortController) {
        abortController.abort();
      }
    };
  }, [abortController]);

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

  const handleProcess = async () => {
    if (!state.sourcePath.trim()) return;

    const controller = new AbortController();
    setAbortController(controller);

    try {
      // Step 1: Gather
      updateState({
        isLoading: true,
        loadingMessage: "Processing folder structure...",
        progress: 0,
      });

      await gatherFiles(
        state.sourcePath,
        (progress) => updateState({ progress: Math.round(progress * 100) }),
        controller.signal,
      );

      if (controller.signal.aborted) return;

      updateState({
        hasGathered: true,
        loadingMessage: "Analyzing and grouping folders...",
        progress: 0,
      });

      // Step 2: Group
      await apiGroupFolders(
        (progress) => updateState({ progress: Math.round(progress * 100) }),
        controller.signal,
      );

      if (controller.signal.aborted) return;

      updateState({
        hasGrouped: true,
        loadingMessage: "Loading dual representation...",
        progress: 100,
      });

      // Step 3: Fetch dual representation
      await fetchDualRep();

      updateState({
        isLoading: false,
        loadingMessage: "",
        progress: 0,
      });
    } catch (error) {
      if (!controller.signal.aborted) {
        updateState({ isLoading: false, loadingMessage: "", progress: 0 });
        console.error("Error processing:", error);
        alert("Error processing folder. See console for details.");
      }
    } finally {
      setAbortController(null);
    }
  };

  const handleCancel = () => {
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
      <InfoBox>
        <InfoTitle>🚀 Dual Representation Mode</InfoTitle>
        <InfoText>
          This experimental version uses the new dual representation API, which
          provides a unified view of both your original folder structure and the
          categorized organization.
        </InfoText>
      </InfoBox>

      <FolderSelectSection>
        <FolderInput
          type="text"
          placeholder="Select folder or enter path..."
          value={state.sourcePath}
          onChange={(e) => updateState({ sourcePath: e.target.value })}
          disabled={state.isLoading}
          onKeyDown={(e) => {
            if (e.key === "Enter" && state.sourcePath.trim()) {
              handleProcess();
            }
          }}
        />
        <BrowseButton onClick={handleBrowseFolder} disabled={state.isLoading}>
          Browse
        </BrowseButton>
        <ProcessButton
          onClick={state.isLoading ? handleCancel : handleProcess}
          disabled={!state.sourcePath.trim() && !state.isLoading}
          $isCancel={state.isLoading}
        >
          {state.isLoading ? "Cancel" : "Process"}
        </ProcessButton>
      </FolderSelectSection>

      {state.isLoading && (
        <LoadingSection>
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
        </LoadingSection>
      )}

      {state.hasGrouped && !state.isLoading && (
        <ButtonRow>
          <SuccessButton onClick={onNext}>
            View Dual Representation →
          </SuccessButton>
        </ButtonRow>
      )}
    </StepContainer>
  );
};

interface ViewStepProps {
  dualRep: DualRepresentation | null;
  isDualRepLoading: boolean;
  dualRepError: Error | null;
  highlightedItemId: string | null;
  onHighlight: (itemId: string | null) => void;
  onPrev: () => void;
}

const ViewStep: React.FC<ViewStepProps> = ({
  dualRep,
  isDualRepLoading,
  dualRepError,
  highlightedItemId,
  onHighlight,
  onPrev,
}) => {
  if (isDualRepLoading) {
    return (
      <StepContainer>
        <LoadingSection>
          <LoadingSpinner />
          <LoadingText>Loading dual representation...</LoadingText>
        </LoadingSection>
      </StepContainer>
    );
  }

  if (dualRepError) {
    return (
      <StepContainer>
        <ErrorBox>
          <ErrorTitle>Error Loading Data</ErrorTitle>
          <ErrorText>{dualRepError.message}</ErrorText>
        </ErrorBox>
        <ButtonRow>
          <SecondaryButton onClick={onPrev}>← Back</SecondaryButton>
        </ButtonRow>
      </StepContainer>
    );
  }

  if (!dualRep) {
    return (
      <StepContainer>
        <InfoBox>
          <InfoText>No data available. Please process a folder first.</InfoText>
        </InfoBox>
        <ButtonRow>
          <SecondaryButton onClick={onPrev}>← Back</SecondaryButton>
        </ButtonRow>
      </StepContainer>
    );
  }

  return (
    <StepContainer>
      <InfoBox>
        <InfoTitle>📊 Dual Representation View</InfoTitle>
        <InfoText>
          Hover over items to see synchronized highlighting across both views.
        </InfoText>
        <StatsRow>
          <Stat>
            <StatLabel>Total Items:</StatLabel>
            <StatValue>{Object.keys(dualRep.items).length}</StatValue>
          </Stat>
          <Stat>
            <StatLabel>Nodes:</StatLabel>
            <StatValue>
              {
                Object.values(dualRep.items).filter(
                  (item: HierarchyItem) => item.type === "node",
                ).length
              }
            </StatValue>
          </Stat>
          <Stat>
            <StatLabel>Categories:</StatLabel>
            <StatValue>
              {
                Object.values(dualRep.items).filter(
                  (item: HierarchyItem) => item.type === "category",
                ).length
              }
            </StatValue>
          </Stat>
        </StatsRow>
      </InfoBox>

      <ComparisonView>
        <Panel>
          <SectionTitle>
            <span>File System Structure</span>
          </SectionTitle>
          <HierarchyBrowser
            items={dualRep.items}
            hierarchy={dualRep.node_hierarchy}
            rootId="node-root"
            highlightedItemId={highlightedItemId}
            onItemClick={onHighlight}
            onItemHover={onHighlight}
            showPath={false}
          />
        </Panel>
        <Panel>
          <SectionTitle>
            <span>Category Structure</span>
          </SectionTitle>
          <HierarchyBrowser
            items={dualRep.items}
            hierarchy={dualRep.category_hierarchy}
            rootId="category-root"
            highlightedItemId={highlightedItemId}
            onItemClick={onHighlight}
            onItemHover={onHighlight}
            showPath={false}
          />
        </Panel>
      </ComparisonView>

      <ButtonRow>
        <SecondaryButton onClick={onPrev}>← Back</SecondaryButton>
      </ButtonRow>
    </StepContainer>
  );
};

// Styled Components (reusing styles from original ImportWizardPage)

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

const FeatureBadge = styled.div`
  position: absolute;
  top: -4px;
  right: 140px;
  background-color: #fbbf24;
  color: #78350f;
  font-size: 0.625rem;
  font-weight: 700;
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  text-transform: uppercase;
  letter-spacing: 0.025em;
`;

const StepIndicator = styled.div`
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-shrink: 0;
`;

const StepItem = styled.div<{ active: boolean; completed: boolean }>`
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  transition: all 0.2s ease;
  border: 1px solid transparent;
  user-select: none;

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
  gap: 1rem;
`;

const InfoBox = styled.div`
  background-color: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 0.5rem;
  padding: 1rem;
  flex-shrink: 0;
`;

const InfoTitle = styled.div`
  font-weight: 600;
  color: #1e40af;
  margin-bottom: 0.5rem;
  font-size: 1rem;
`;

const InfoText = styled.div`
  color: #1e40af;
  font-size: 0.875rem;
  line-height: 1.5;
`;

const StatsRow = styled.div`
  display: flex;
  gap: 1.5rem;
  margin-top: 0.75rem;
`;

const Stat = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const StatLabel = styled.span`
  font-size: 0.75rem;
  color: #1e40af;
  font-weight: 500;
`;

const StatValue = styled.span`
  font-size: 0.875rem;
  color: #1e40af;
  font-weight: 700;
`;

const ErrorBox = styled.div`
  background-color: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 0.5rem;
  padding: 1rem;
  flex-shrink: 0;
`;

const ErrorTitle = styled.div`
  font-weight: 600;
  color: #991b1b;
  margin-bottom: 0.5rem;
  font-size: 1rem;
`;

const ErrorText = styled.div`
  color: #991b1b;
  font-size: 0.875rem;
`;

const FolderSelectSection = styled.div`
  display: flex;
  gap: 0.75rem;
  align-items: stretch;
  flex-shrink: 0;
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

  &:disabled {
    background-color: #f3f4f6;
    cursor: not-allowed;
  }
`;

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

const ProcessButton = styled(BaseButton)<{ $isCancel?: boolean }>`
  background-color: ${(props) => (props.$isCancel ? "#ef4444" : "#2563eb")};
  color: white;

  &:hover:not(:disabled) {
    background-color: ${(props) => (props.$isCancel ? "#dc2626" : "#1d4ed8")};
  }
`;

const LoadingSection = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  padding: 2rem;
  flex: 1;
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

const ProgressSection = styled.div`
  width: 100%;
  max-width: 300px;
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
  font-size: 0.875rem;
`;

const ComparisonView = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  flex: 1;
  min-height: 0;
  overflow: hidden;
`;

const Panel = styled.div`
  background: #fafbfc;
  padding: 0.5rem;
  border: 1px solid #e2e8f0;
  border-radius: 0.375rem;
  display: flex;
  flex-direction: column;
  overflow: hidden;
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
  padding-bottom: 0.5rem;
  border-bottom: 1px solid #e2e8f0;

  span {
    background: #fafbfc;
    padding: 0 0.5rem;
    position: relative;
    z-index: 1;
  }
`;

const ButtonRow = styled.div`
  display: flex;
  gap: 0.75rem;
  justify-content: flex-end;
  padding: 0.75rem 0 0 0;
  border-top: 1px solid #e2e8f0;
  flex-shrink: 0;
`;

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
