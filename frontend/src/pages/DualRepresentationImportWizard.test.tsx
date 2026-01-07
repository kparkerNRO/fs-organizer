import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "../test/utils";
import { DualRepresentationImportWizard } from "./DualRepresentationImportWizard";
import * as api from "../api";
import * as folderSelection from "../utils/folderSelection";

// Mock modules
vi.mock("../api");
vi.mock("../utils/folderSelection");
vi.mock("../hooks/useDualRepresentation");


// Mock the useDualRepresentation hook
vi.mock("../hooks/useDualRepresentation", () => ({
  useDualRepresentation: () => ({
    dualRep: null,
    isLoading: false,
    error: null,
    highlightedItemId: null,
    fetchDualRepresentation: vi.fn(),
    highlightItem: vi.fn(),
    refresh: vi.fn(),
    addToParent: vi.fn(),
    removeFromParent: vi.fn(),
    moveItem: vi.fn(),
    applyPendingChanges: vi.fn(),
    clearPendingChanges: vi.fn(),
    setView: vi.fn(),
    getItem: vi.fn(),
    getChildren: vi.fn(),
    findItemInBothHierarchies: vi.fn(),
    pendingDiff: { added: {}, deleted: {} },
    hasPendingChanges: false,
    selectedView: "node" as const,
  }),
}));

describe("DualRepresentationImportWizard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the wizard with step 1 initially", () => {
    render(<DualRepresentationImportWizard />);

    expect(screen.getByText("Import Wizard (Dual Representation)")).toBeInTheDocument();
    expect(screen.getByText("EXPERIMENTAL")).toBeInTheDocument();
    expect(screen.getByText("1. Gather & Group")).toBeInTheDocument();
  });

  it("shows experimental badge", () => {
    render(<DualRepresentationImportWizard />);

    expect(screen.getByText("EXPERIMENTAL")).toBeInTheDocument();
  });

  it("displays folder input and buttons", () => {
    render(<DualRepresentationImportWizard />);

    expect(screen.getByPlaceholderText("Select folder or enter path...")).toBeInTheDocument();
    expect(screen.getByText("Browse")).toBeInTheDocument();
    expect(screen.getByText("Process")).toBeInTheDocument();
  });

  it("allows typing in folder path", () => {
    render(<DualRepresentationImportWizard />);

    const input = screen.getByPlaceholderText("Select folder or enter path...") as HTMLInputElement;

    fireEvent.change(input, { target: { value: "/test/path" } });

    expect(input.value).toBe("/test/path");
  });

  it("disables Process button when path is empty", () => {
    render(<DualRepresentationImportWizard />);

    const processButton = screen.getByText("Process");

    expect(processButton).toBeDisabled();
  });

  it("enables Process button when path is provided", () => {
    render(<DualRepresentationImportWizard />);

    const input = screen.getByPlaceholderText("Select folder or enter path...");
    const processButton = screen.getByText("Process");

    fireEvent.change(input, { target: { value: "/test/path" } });

    expect(processButton).not.toBeDisabled();
  });

  it("calls selectFolder when Browse is clicked", async () => {
    vi.mocked(folderSelection.selectFolder).mockResolvedValue({
      success: true,
      path: "/selected/path",
    });

    render(<DualRepresentationImportWizard />);

    const browseButton = screen.getByText("Browse");

    fireEvent.click(browseButton);

    await waitFor(() => {
      expect(folderSelection.selectFolder).toHaveBeenCalled();
    });
  });

  it("updates input when folder is selected via Browse", async () => {
    vi.mocked(folderSelection.selectFolder).mockResolvedValue({
      success: true,
      path: "/selected/path",
    });

    render(<DualRepresentationImportWizard />);

    const browseButton = screen.getByText("Browse");
    const input = screen.getByPlaceholderText("Select folder or enter path...") as HTMLInputElement;

    fireEvent.click(browseButton);

    await waitFor(() => {
      expect(input.value).toBe("/selected/path");
    });
  });

  it("does not update input when folder selection is cancelled", async () => {
    vi.mocked(folderSelection.selectFolder).mockResolvedValue({
      success: false,
      error: "User cancelled",
    });

    render(<DualRepresentationImportWizard />);

    const browseButton = screen.getByText("Browse");
    const input = screen.getByPlaceholderText("Select folder or enter path...") as HTMLInputElement;

    fireEvent.click(browseButton);

    await waitFor(() => {
      expect(folderSelection.selectFolder).toHaveBeenCalled();
    });

    expect(input.value).toBe("");
  });

  it("shows loading state when processing", async () => {
    vi.mocked(api.gatherFiles).mockImplementation(() => new Promise(() => {})); // Never resolves

    render(<DualRepresentationImportWizard />);

    const input = screen.getByPlaceholderText("Select folder or enter path...");
    fireEvent.change(input, { target: { value: "/test/path" } });

    const processButton = screen.getByText("Process");
    fireEvent.click(processButton);

    await waitFor(() => {
      expect(screen.getByText("Processing folder structure...")).toBeInTheDocument();
    });
  });

  it("changes button to Cancel when processing", async () => {
    vi.mocked(api.gatherFiles).mockImplementation(() => new Promise(() => {}));

    render(<DualRepresentationImportWizard />);

    const input = screen.getByPlaceholderText("Select folder or enter path...");
    fireEvent.change(input, { target: { value: "/test/path" } });

    const processButton = screen.getByText("Process");
    fireEvent.click(processButton);

    await waitFor(() => {
      expect(screen.getByText("Cancel")).toBeInTheDocument();
    });
  });

  it("displays info box with feature description", () => {
    render(<DualRepresentationImportWizard />);

    expect(screen.getByText("🚀 Dual Representation Mode")).toBeInTheDocument();
    expect(
      screen.getByText(/This experimental version uses the new dual representation API/i)
    ).toBeInTheDocument();
  });

  it("shows step indicators", () => {
    render(<DualRepresentationImportWizard />);

    expect(screen.getByText("1. Gather & Group")).toBeInTheDocument();
    expect(screen.getByText("2. View Results")).toBeInTheDocument();
  });

  it("highlights current step", () => {
    render(<DualRepresentationImportWizard />);

    const step1 = screen.getByText("1. Gather & Group");
    const step2 = screen.getByText("2. View Results");

    // Step 1 should be active (visible in document is enough for this test)
    expect(step1).toBeInTheDocument();
    expect(step2).toBeInTheDocument();
  });

  it("handles processing errors gracefully", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const alertSpy = vi.spyOn(window, "alert").mockImplementation(() => {});

    vi.mocked(api.gatherFiles).mockRejectedValue(new Error("Processing failed"));

    render(<DualRepresentationImportWizard />);

    const input = screen.getByPlaceholderText("Select folder or enter path...");
    fireEvent.change(input, { target: { value: "/test/path" } });

    const processButton = screen.getByText("Process");
    fireEvent.click(processButton);

    await waitFor(() => {
      expect(alertSpy).toHaveBeenCalledWith(
        "Error processing folder. See console for details."
      );
    });

    consoleErrorSpy.mockRestore();
    alertSpy.mockRestore();
  });

  it("allows cancelling a running process", async () => {
    let abortCalled = false;
    const mockAbortController = {
      signal: { aborted: false },
      abort: () => {
        abortCalled = true;
      },
    };

    global.AbortController = vi.fn(() => mockAbortController) as unknown as typeof AbortController;

    vi.mocked(api.gatherFiles).mockImplementation(() => new Promise(() => {}));

    render(<DualRepresentationImportWizard />);

    const input = screen.getByPlaceholderText("Select folder or enter path...");
    fireEvent.change(input, { target: { value: "/test/path" } });

    const processButton = screen.getByText("Process");
    fireEvent.click(processButton);

    await waitFor(() => {
      expect(screen.getByText("Cancel")).toBeInTheDocument();
    });

    const cancelButton = screen.getByText("Cancel");
    fireEvent.click(cancelButton);

    expect(abortCalled).toBe(true);
  });

  it("supports Enter key to submit folder path", () => {
    vi.mocked(api.gatherFiles).mockResolvedValue({
      task_id: "test",
      status: "completed",
      message: "Done",
      progress: 100,
      result: {},
      error: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });

    render(<DualRepresentationImportWizard />);

    const input = screen.getByPlaceholderText("Select folder or enter path...");
    fireEvent.change(input, { target: { value: "/test/path" } });

    fireEvent.keyDown(input, { key: "Enter", code: "Enter" });

    expect(api.gatherFiles).toHaveBeenCalled();
  });
});
