import React, { useState, useCallback } from "react";
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
    <div
      className="w-screen h-screen m-0 p-1 bg-slate-50 overflow-hidden flex flex-col box-border"
      data-testid="wizard-container"
    >
      <div className="flex justify-between items-center mb-2 flex-shrink-0 h-12 relative min-w-0">
        <div style={{ width: "120px", flexShrink: 0 }} />{" "}
        {/* Spacer for centering */}
        <h1 className="text-2xl font-bold text-gray-800 m-0 pt-0 leading-tight flex-1 text-center min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">
          Import Wizard
        </h1>
        <div className="flex gap-2 items-center flex-shrink-0">
          <div
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all duration-200 border border-transparent select-none ${
              1 <= state.currentStep
                ? "cursor-pointer opacity-100"
                : "cursor-not-allowed opacity-60"
            } ${
              state.currentStep === 1
                ? "bg-blue-600 text-white shadow-[0_0_0_2px_rgba(37,99,235,0.2)] border-blue-700"
                : ""
            } ${
              state.currentStep > 1 && state.currentStep !== 1
                ? "bg-emerald-500 text-white"
                : ""
            } ${
              state.currentStep < 1 && state.currentStep !== 1
                ? "bg-slate-200 text-slate-500"
                : ""
            } ${
              1 <= state.currentStep &&
              state.currentStep !== 1 &&
              state.currentStep > 1
                ? "hover:-translate-y-0.5 hover:shadow-sm hover:bg-emerald-600"
                : ""
            } ${
              1 <= state.currentStep &&
              state.currentStep !== 1 &&
              state.currentStep < 1
                ? "hover:-translate-y-0.5 hover:shadow-sm hover:bg-slate-300"
                : ""
            }`}
            onClick={() => goToStep(1)}
          >
            1. Import
          </div>
          <div
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all duration-200 border border-transparent select-none ${
              2 <= state.currentStep
                ? "cursor-pointer opacity-100"
                : "cursor-not-allowed opacity-60"
            } ${
              state.currentStep === 2
                ? "bg-blue-600 text-white shadow-[0_0_0_2px_rgba(37,99,235,0.2)] border-blue-700"
                : ""
            } ${
              state.currentStep > 2 && state.currentStep !== 2
                ? "bg-emerald-500 text-white"
                : ""
            } ${
              state.currentStep < 2 && state.currentStep !== 2
                ? "bg-slate-200 text-slate-500"
                : ""
            } ${
              2 <= state.currentStep &&
              state.currentStep !== 2 &&
              state.currentStep > 2
                ? "hover:-translate-y-0.5 hover:shadow-sm hover:bg-emerald-600"
                : ""
            } ${
              2 <= state.currentStep &&
              state.currentStep !== 2 &&
              state.currentStep < 2
                ? "hover:-translate-y-0.5 hover:shadow-sm hover:bg-slate-300"
                : ""
            }`}
            onClick={() => goToStep(2)}
          >
            2. Group
          </div>
          <div
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all duration-200 border border-transparent select-none ${
              3 <= state.currentStep
                ? "cursor-pointer opacity-100"
                : "cursor-not-allowed opacity-60"
            } ${
              state.currentStep === 3
                ? "bg-blue-600 text-white shadow-[0_0_0_2px_rgba(37,99,235,0.2)] border-blue-700"
                : ""
            } ${
              state.currentStep > 3 && state.currentStep !== 3
                ? "bg-emerald-500 text-white"
                : ""
            } ${
              state.currentStep < 3 && state.currentStep !== 3
                ? "bg-slate-200 text-slate-500"
                : ""
            } ${
              3 <= state.currentStep &&
              state.currentStep !== 3 &&
              state.currentStep > 3
                ? "hover:-translate-y-0.5 hover:shadow-sm hover:bg-emerald-600"
                : ""
            } ${
              3 <= state.currentStep &&
              state.currentStep !== 3 &&
              state.currentStep < 3
                ? "hover:-translate-y-0.5 hover:shadow-sm hover:bg-slate-300"
                : ""
            }`}
            onClick={() => goToStep(3)}
          >
            3. Organize
          </div>
          <div
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all duration-200 border border-transparent select-none ${
              4 <= state.currentStep
                ? "cursor-pointer opacity-100"
                : "cursor-not-allowed opacity-60"
            } ${
              state.currentStep === 4
                ? "bg-blue-600 text-white shadow-[0_0_0_2px_rgba(37,99,235,0.2)] border-blue-700"
                : ""
            } ${
              state.currentStep !== 4 ? "bg-slate-200 text-slate-500" : ""
            } ${
              4 <= state.currentStep && state.currentStep !== 4
                ? "hover:-translate-y-0.5 hover:shadow-sm hover:bg-slate-300"
                : ""
            }`}
            onClick={() => goToStep(4)}
          >
            4. Review
          </div>
        </div>
      </div>
      <div
        className="bg-white rounded-md p-3 shadow-sm relative flex-1 flex flex-col overflow-hidden min-h-0"
        data-testid="wizard-content"
      >
        {renderStep()}
      </div>
    </div>
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
    <div className="flex-1 flex flex-col relative overflow-hidden min-h-0">
      <div className="flex gap-3 mb-4 items-stretch">
        <input
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
          className="flex-1 px-2 py-2 border border-gray-300 rounded-md text-sm font-inherit focus:outline-none focus:border-blue-600 focus:ring-4 focus:ring-blue-100"
        />
        <button
          onClick={handleBrowseFolder}
          disabled={state.isLoading}
          className="px-4 py-2 border-0 rounded-md cursor-pointer text-sm font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-gray-500 text-white hover:bg-gray-600 disabled:hover:bg-gray-500"
        >
          Browse
        </button>
        <button
          onClick={
            state.isLoading
              ? handleCancelImport
              : () => handleFolderSelect(state.sourcePath)
          }
          disabled={!state.sourcePath.trim() && !state.isLoading}
          title={
            state.isLoading
              ? "Cancel folder processing"
              : !state.sourcePath.trim()
                ? "Select a folder first"
                : "Process selected folder"
          }
          className={`px-4 py-2 border-0 rounded-md cursor-pointer text-sm font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed text-white ${
            state.isLoading
              ? "bg-red-500 hover:bg-red-600 disabled:hover:bg-red-500"
              : "bg-blue-600 hover:bg-blue-700 disabled:hover:bg-blue-600"
          }`}
        >
          {state.isLoading ? "Cancel" : "Process"}
        </button>
      </div>

      <div
        className={`p-2 border border-slate-200 rounded-md bg-gradient-to-b from-white to-slate-50 shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_2px_4px_-1px_rgba(0,0,0,0.1)] transition-all duration-200 relative overflow-y-auto flex-1 min-h-0 hover:border-slate-300 hover:shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_4px_12px_-2px_rgba(0,0,0,0.1)] ${
          state.isLoading ? "opacity-50 pointer-events-none" : "opacity-100"
        }`}
      >
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
          <div className="absolute inset-0 bg-black/30 flex items-center justify-center z-10 rounded-xl">
            <div className="bg-white px-8 py-6 rounded-xl shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_10px_15px_-3px_rgba(0,0,0,0.2)] border border-slate-200 flex flex-col items-center gap-4 min-w-[280px] max-w-[350px] text-center">
              <div className="w-8 h-8 border-[3px] border-gray-200 border-t-blue-600 rounded-full animate-spin" />
              <div className="text-gray-700 font-medium text-base">
                {state.loadingMessage}
              </div>
              {state.progress > 0 && (
                <div className="p-2 border border-slate-200 rounded-md bg-gradient-to-b from-white to-slate-50 shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_2px_4px_-1px_rgba(0,0,0,0.1)] transition-all duration-200 relative overflow-y-auto flex-1 min-h-0 hover:border-slate-300 hover:shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_4px_12px_-2px_rgba(0,0,0,0.1)] my-4 p-4 rounded-lg flex-none">
                  <div className="w-full h-2 bg-gray-200 rounded overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 transition-[width] duration-300"
                      style={{ width: `${state.progress}%` }}
                    />
                  </div>
                  <div className="text-center mt-2 text-gray-500 font-medium">
                    {state.progress}% complete
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {state.originalStructure && (
        <div className="flex gap-3 justify-end pt-3 pb-3 border-t border-slate-200 flex-wrap flex-shrink-0 min-h-[56px] items-center">
          <button
            onClick={onNext}
            className="px-8 py-3 text-base border-0 rounded-md cursor-pointer font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-emerald-500 text-white hover:bg-emerald-600 disabled:hover:bg-emerald-500"
          >
            Next
          </button>
        </div>
      )}
    </div>
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
  }, [abortController]);

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
    <div className="flex-1 flex flex-col relative overflow-hidden min-h-0">
      <div
        className={`grid grid-cols-2 gap-3 relative transition-opacity duration-200 flex-1 min-h-0 ${
          state.isLoading ? "opacity-50 pointer-events-none" : "opacity-100"
        }`}
      >
        <div className="p-2 border border-slate-200 rounded-md bg-gradient-to-b from-white to-slate-50 shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_2px_4px_-1px_rgba(0,0,0,0.1)] transition-all duration-200 relative overflow-y-auto flex-1 min-h-0 hover:border-slate-300 hover:shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_4px_12px_-2px_rgba(0,0,0,0.1)] bg-[#fafbfc] flex flex-col">
          <h4 className="font-bold m-0 text-slate-900 text-sm leading-tight relative text-center mb-2 before:content-[''] before:absolute before:top-1/2 before:left-0 before:w-[calc(50%-0.75rem)] before:h-[1px] before:bg-slate-100 before:opacity-60 after:content-[''] after:absolute after:top-1/2 after:right-0 after:w-[calc(50%-0.75rem)] after:h-[1px] after:bg-slate-100 after:opacity-60">
            <span className="bg-[#fafbfc] px-2 relative z-[1] transition-colors duration-200">
              Original Structure
            </span>
          </h4>
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
        </div>
        <div className="p-2 border border-slate-200 rounded-md bg-gradient-to-b from-white to-slate-50 shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_2px_4px_-1px_rgba(0,0,0,0.1)] transition-all duration-200 relative overflow-y-auto flex-1 min-h-0 hover:border-slate-300 hover:shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_4px_12px_-2px_rgba(0,0,0,0.1)] bg-[#fafbfc] flex flex-col">
          <h4 className="font-bold m-0 text-slate-900 text-sm leading-tight relative text-center mb-2 before:content-[''] before:absolute before:top-1/2 before:left-0 before:w-[calc(50%-0.75rem)] before:h-[1px] before:bg-slate-100 before:opacity-60 after:content-[''] after:absolute after:top-1/2 after:right-0 after:w-[calc(50%-0.75rem)] after:h-[1px] after:bg-slate-100 after:opacity-60">
            <span className="bg-[#fafbfc] px-2 relative z-[1] transition-colors duration-200">
              Grouped Structure
            </span>
          </h4>
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
        </div>
      </div>
      {state.isLoading && (
        <div className="absolute inset-0 bg-black/30 flex items-center justify-center z-10 rounded-xl">
          <div className="bg-white px-8 py-6 rounded-xl shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_10px_15px_-3px_rgba(0,0,0,0.2)] border border-slate-200 flex flex-col items-center gap-4 min-w-[280px] max-w-[350px] text-center">
            <div className="w-8 h-8 border-[3px] border-gray-200 border-t-blue-600 rounded-full animate-spin" />
            <div className="text-gray-700 font-medium text-base">
              {state.loadingMessage}
            </div>
          </div>
        </div>
      )}

      <div className="flex gap-3 justify-end pt-3 pb-3 border-t border-slate-200 flex-wrap flex-shrink-0 min-h-[56px] items-center">
        <button
          onClick={onPrev}
          className="px-8 py-3 text-base border-0 rounded-md cursor-pointer font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-gray-500 text-white hover:bg-gray-600 disabled:hover:bg-gray-500"
        >
          Previous
        </button>
        <button
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
          className="px-8 py-3 text-base border-0 rounded-md cursor-pointer font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-amber-500 text-white hover:bg-amber-600 disabled:hover:bg-amber-500"
        >
          Save
        </button>
        <button
          onClick={onNext}
          disabled={state.isLoading || !state.groupedStructure}
          className="px-8 py-3 text-base border-0 rounded-md cursor-pointer font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-emerald-500 text-white hover:bg-emerald-600 disabled:hover:bg-emerald-500"
        >
          Next
        </button>
      </div>
    </div>
  );
};

const OrganizeStep: React.FC<StepProps> = ({
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
  }, [abortController]);

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
    <div className="flex-1 flex flex-col relative overflow-hidden min-h-0">
      <div
        className={`grid grid-cols-2 gap-3 relative transition-opacity duration-200 flex-1 min-h-0 ${
          state.isLoading ? "opacity-50 pointer-events-none" : "opacity-100"
        }`}
      >
        <div className="p-2 border border-slate-200 rounded-md bg-gradient-to-b from-white to-slate-50 shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_2px_4px_-1px_rgba(0,0,0,0.1)] transition-all duration-200 relative overflow-y-auto flex-1 min-h-0 hover:border-slate-300 hover:shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_4px_12px_-2px_rgba(0,0,0,0.1)] bg-[#fafbfc] flex flex-col">
          <h4 className="font-bold m-0 text-slate-900 text-sm leading-tight relative text-center mb-2 before:content-[''] before:absolute before:top-1/2 before:left-0 before:w-[calc(50%-0.75rem)] before:h-[1px] before:bg-slate-100 before:opacity-60 after:content-[''] after:absolute after:top-1/2 after:right-0 after:w-[calc(50%-0.75rem)] after:h-[1px] after:bg-slate-100 after:opacity-60">
            <span className="bg-[#fafbfc] px-2 relative z-[1] transition-colors duration-200">
              Grouped Structure
            </span>
          </h4>
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
        </div>
        <div className="p-2 border border-slate-200 rounded-md bg-gradient-to-b from-white to-slate-50 shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_2px_4px_-1px_rgba(0,0,0,0.1)] transition-all duration-200 relative overflow-y-auto flex-1 min-h-0 hover:border-slate-300 hover:shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_4px_12px_-2px_rgba(0,0,0,0.1)] bg-[#fafbfc] flex flex-col">
          <h4 className="font-bold m-0 text-slate-900 text-sm leading-tight relative text-center mb-2 before:content-[''] before:absolute before:top-1/2 before:left-0 before:w-[calc(50%-0.75rem)] before:h-[1px] before:bg-slate-100 before:opacity-60 after:content-[''] after:absolute after:top-1/2 after:right-0 after:w-[calc(50%-0.75rem)] after:h-[1px] after:bg-slate-100 after:opacity-60">
            <span className="bg-[#fafbfc] px-2 relative z-[1] transition-colors duration-200">
              Final Organization
            </span>
          </h4>
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
        </div>
      </div>
      {state.isLoading && (
        <div className="absolute inset-0 bg-black/30 flex items-center justify-center z-10 rounded-xl">
          <div className="bg-white px-8 py-6 rounded-xl shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_10px_15px_-3px_rgba(0,0,0,0.2)] border border-slate-200 flex flex-col items-center gap-4 min-w-[280px] max-w-[350px] text-center">
            <div className="w-8 h-8 border-[3px] border-gray-200 border-t-blue-600 rounded-full animate-spin" />
            <div className="text-gray-700 font-medium text-base">
              {state.loadingMessage}
            </div>
          </div>
        </div>
      )}

      <div className="flex gap-3 justify-end pt-3 pb-3 border-t border-slate-200 flex-wrap flex-shrink-0 min-h-[56px] items-center">
        <button
          onClick={onPrev}
          className="px-8 py-3 text-base border-0 rounded-md cursor-pointer font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-gray-500 text-white hover:bg-gray-600 disabled:hover:bg-gray-500"
        >
          Previous
        </button>
        <button
          onClick={() => {
            updateState({
              hasTriggeredFolders: false,
              organizedStructure: undefined,
              selectedFileId: null,
            });
          }}
          disabled={state.isLoading}
          className="px-8 py-3 text-base border-0 rounded-md cursor-pointer font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-amber-500 text-white hover:bg-amber-600 disabled:hover:bg-amber-500"
        >
          Re-run Organization
        </button>
        <button
          onClick={onNext}
          disabled={state.isLoading || !state.organizedStructure}
          className="px-8 py-3 text-base border-0 rounded-md cursor-pointer font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-emerald-500 text-white hover:bg-emerald-600 disabled:hover:bg-emerald-500"
        >
          Next
        </button>
      </div>
    </div>
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
    <div className="flex-1 flex flex-col relative overflow-hidden min-h-0">
      <div className="grid grid-cols-2 gap-3 relative transition-opacity duration-200 flex-1 min-h-0 opacity-100">
        <div className="p-2 border border-slate-200 rounded-md bg-gradient-to-b from-white to-slate-50 shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_2px_4px_-1px_rgba(0,0,0,0.1)] transition-all duration-200 relative overflow-y-auto flex-1 min-h-0 hover:border-slate-300 hover:shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_4px_12px_-2px_rgba(0,0,0,0.1)] bg-[#fafbfc] flex flex-col">
          <h4 className="font-bold m-0 text-slate-900 text-sm leading-tight relative text-center mb-2 before:content-[''] before:absolute before:top-1/2 before:left-0 before:w-[calc(50%-0.75rem)] before:h-[1px] before:bg-slate-100 before:opacity-60 after:content-[''] after:absolute after:top-1/2 after:right-0 after:w-[calc(50%-0.75rem)] after:h-[1px] after:bg-slate-100 after:opacity-60">
            <span className="bg-[#fafbfc] px-2 relative z-[1] transition-colors duration-200">
              Configuration Settings
            </span>
          </h4>
          <div className="mb-3">
            <label className="block font-semibold mb-2 text-gray-800 text-base">
              Target Folder:
            </label>
            <div className="flex gap-3 mb-4 items-stretch">
              <input
                type="text"
                placeholder="Choose destination folder..."
                value={state.targetPath}
                onChange={(e) => updateState({ targetPath: e.target.value })}
                className="flex-1 px-2 py-2 border border-gray-300 rounded-md text-sm font-inherit focus:outline-none focus:border-blue-600 focus:ring-4 focus:ring-blue-100"
              />
              <button
                onClick={handleBrowseTargetFolder}
                className="px-4 py-2 border-0 rounded-md cursor-pointer text-sm font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-gray-500 text-white hover:bg-gray-600 disabled:hover:bg-gray-500"
              >
                Browse
              </button>
            </div>
          </div>

          <div className="mb-3">
            <label className="block font-semibold mb-2 text-gray-800 text-base">
              Duplicate Handling:
            </label>
            <select
              value={state.duplicateHandling}
              onChange={(e) =>
                updateState({
                  duplicateHandling: e.target
                    .value as ImportWizardState["duplicateHandling"],
                })
              }
              className="w-full px-3 py-3 border border-gray-300 rounded-md text-base font-inherit bg-white focus:outline-none focus:border-blue-600 focus:ring-4 focus:ring-blue-100"
            >
              <option value="newest">Keep Newest</option>
              <option value="largest">Keep Largest</option>
              <option value="both">Keep Both</option>
              <option value="both-if-different">
                Keep Both if Not Identical
              </option>
            </select>
          </div>
        </div>
        <div className="p-2 border border-slate-200 rounded-md bg-gradient-to-b from-white to-slate-50 shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_2px_4px_-1px_rgba(0,0,0,0.1)] transition-all duration-200 relative overflow-y-auto flex-1 min-h-0 hover:border-slate-300 hover:shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_4px_12px_-2px_rgba(0,0,0,0.1)] bg-[#fafbfc] flex flex-col">
          {state.organizedStructure && (
            <FolderBrowser
              folderTree={state.organizedStructure}
              onSelectItem={() => {}}
              externalSelectedFile={null}
              shouldSync={false}
              showConfidence={false}
            />
          )}
        </div>
      </div>

      {isApplying && (
        <div className="p-2 border border-slate-200 rounded-md bg-gradient-to-b from-white to-slate-50 shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_2px_4px_-1px_rgba(0,0,0,0.1)] transition-all duration-200 relative overflow-y-auto flex-1 min-h-0 hover:border-slate-300 hover:shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_4px_12px_-2px_rgba(0,0,0,0.1)] my-4 p-4 rounded-lg flex-none">
          <div className="w-full h-2 bg-gray-200 rounded overflow-hidden">
            <div
              className="h-full bg-emerald-500 transition-[width] duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="text-center mt-2 text-gray-500 font-medium">
            Applying organization... {progress}%
          </div>
        </div>
      )}

      <div className="flex gap-3 justify-end pt-3 pb-3 border-t border-slate-200 flex-wrap flex-shrink-0 min-h-[56px] items-center">
        <button
          onClick={onPrev}
          disabled={isApplying}
          className="px-8 py-3 text-base border-0 rounded-md cursor-pointer font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-gray-500 text-white hover:bg-gray-600 disabled:hover:bg-gray-500"
        >
          Previous
        </button>
        <button
          onClick={handleApply}
          disabled={isApplying || !state.targetPath}
          className="px-8 py-3 text-base border-0 rounded-md cursor-pointer font-medium whitespace-nowrap transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed bg-red-600 text-white hover:bg-red-700 disabled:hover:bg-red-600"
        >
          {isApplying ? "Applying..." : "Apply Organization"}
        </button>
      </div>
    </div>
  );
};
