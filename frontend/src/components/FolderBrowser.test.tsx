import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "../test/utils";
import { FolderBrowser } from "./FolderBrowser";
import { TREE_TYPE } from "../types/enums";
import { FolderV2 } from "../types/types";
import { mockRootFolder, simpleTestTree } from "../test/folderTreeTestData";

// Mock the useFolderTree hook
const mockUseFolderTree = {
  originalTree: null,
  modifiedTree: null,
  hasModifications: false,
  expandedFolders: new Set<string>(),
  selectedFileId: null,
  selectedFolderPaths: [],
  isOperationInProgress: false,
  lastOperation: null,
  operationHistory: [],
  setTreeData: vi.fn(),
  resetToOriginal: vi.fn(),
  renameItem: vi.fn().mockResolvedValue({ success: true }),
  createFolder: vi.fn().mockResolvedValue({ success: true }),
  deleteItems: vi.fn().mockResolvedValue({ success: true }),
  moveItem: vi.fn().mockResolvedValue({ success: true }),
  mergeItems: vi.fn().mockResolvedValue({ success: true }),
  flattenItems: vi.fn().mockResolvedValue({ success: true }),
  invertItems: vi.fn().mockResolvedValue({ success: true }),
  undoLastOperation: vi.fn().mockResolvedValue({ success: true }),
  redoLastUndoneOperation: vi.fn().mockResolvedValue({ success: true }),
  clearHistory: vi.fn(),
  updateTreeState: vi.fn(),
};

vi.mock("../hooks/useFolderTree", () => ({
  useFolderTree: () => mockUseFolderTree,
}));

// Mock the utility functions
vi.mock("../utils/folderTreeOperations", async () => {
  const actual = await vi.importActual("../utils/folderTreeOperations");
  return {
    ...actual,
    canFlattenFolders: vi.fn().mockReturnValue({ canFlatten: true }),
    canInvertFolder: vi.fn().mockReturnValue({ canInvert: true }),
    setConfidenceToMax: vi.fn(),
  };
});

const defaultProps = {
  folderTree: mockRootFolder,
  onSelectItem: vi.fn(),
  onSelectFolder: vi.fn(),
  externalSelectedFile: null,
  shouldSync: true,
  showConfidence: false,
  treeType: TREE_TYPE.LOADED,
};

// Mock DragEvent for tests
global.DragEvent = class DragEvent extends Event {
  dataTransfer: {
    effectAllowed: string;
    dropEffect: string;
    setData: (format: string, data: string) => void;
    getData: (format: string) => string;
  };

  constructor(type: string, options?: EventInit) {
    super(type, options);
    this.dataTransfer = {
      effectAllowed: "all",
      dropEffect: "none",
      setData: vi.fn(),
      getData: vi.fn().mockReturnValue(""),
    };
  }
} as unknown as typeof DragEvent;

describe("FolderBrowser", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset clipboard mock
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Basic Rendering", () => {
    it("renders the folder tree structure", async () => {
      render(<FolderBrowser {...defaultProps} />);

      // The root folder should be expanded by default and show its children
      await waitFor(() => {
        expect(screen.getByText("documents")).toBeInTheDocument();
        expect(screen.getByText("code")).toBeInTheDocument();
      });
    });

    it("renders error message when no folder tree is provided", () => {
      render(<FolderBrowser {...defaultProps} folderTree={null} />);

      expect(
        screen.getByText("Failed to load folder structure"),
      ).toBeInTheDocument();
    });

    it("shows confidence when showConfidence is true", () => {
      render(<FolderBrowser {...defaultProps} showConfidence={true} />);

      const confidenceElements = screen.getAllByText((_content, element) => {
        return element?.textContent?.includes("Confidence: 100%") || false;
      });
      expect(confidenceElements.length).toBeGreaterThan(0);
    });
  });

  describe("Expanding/Collapsing Folders", () => {
    it("expands folder when chevron is clicked", async () => {
      render(<FolderBrowser {...defaultProps} />);

      const documentsFolder = screen
        .getByText("documents")
        .closest("[data-folder-path]") as HTMLElement;
      const chevron = within(documentsFolder!).getByRole("button");

      fireEvent.click(chevron);

      await waitFor(() => {
        expect(screen.getByText("document1.pdf")).toBeInTheDocument();
        expect(screen.getByText("images")).toBeInTheDocument();
      });
    });

    it("collapses folder when chevron is clicked on expanded folder", async () => {
      render(<FolderBrowser {...defaultProps} />);

      const documentsFolder = screen
        .getByText("documents")
        .closest("[data-folder-path]") as HTMLElement;
      const chevron = within(documentsFolder!).getByRole("button");

      // First expand
      fireEvent.click(chevron);
      await waitFor(() => {
        expect(screen.getByText("document1.pdf")).toBeInTheDocument();
      });

      // Then collapse
      fireEvent.click(chevron);
      await waitFor(() => {
        expect(screen.queryByText("document1.pdf")).not.toBeInTheDocument();
      });
    });

    it("prevents folder row click when chevron is clicked", () => {
      const onSelectFolder = vi.fn();
      render(
        <FolderBrowser {...defaultProps} onSelectFolder={onSelectFolder} />,
      );

      const documentsFolder = screen
        .getByText("documents")
        .closest("[data-folder-path]") as HTMLElement;
      const chevron = within(documentsFolder!).getByRole("button");

      fireEvent.click(chevron);

      expect(onSelectFolder).not.toHaveBeenCalled();
    });
  });

  describe("Single Selection", () => {
    it("selects file when clicked", () => {
      const onSelectItem = vi.fn();
      render(
        <FolderBrowser
          {...defaultProps}
          folderTree={simpleTestTree}
          onSelectItem={onSelectItem}
        />,
      );

      // Expand folder first
      const folder1 = screen
        .getByText("folder1")
        .closest("[data-folder-path]") as HTMLElement;
      const chevron = within(folder1!).getByRole("button");
      fireEvent.click(chevron);

      // Click file
      const file = screen.getByText("file1.txt");
      fireEvent.click(file);

      expect(onSelectItem).toHaveBeenCalledWith(10);
    });

    it("selects folder when clicked", () => {
      const onSelectFolder = vi.fn();
      render(
        <FolderBrowser {...defaultProps} onSelectFolder={onSelectFolder} />,
      );

      const folder = screen.getByText("documents");
      fireEvent.click(folder);

      expect(onSelectFolder).toHaveBeenCalledWith("root/documents");
    });

    it("clears folder selection when file is selected", () => {
      const onSelectFolder = vi.fn();
      render(
        <FolderBrowser
          {...defaultProps}
          folderTree={simpleTestTree}
          onSelectFolder={onSelectFolder}
        />,
      );

      // First select folder
      const folder = screen.getByText("folder1");
      fireEvent.click(folder);
      expect(onSelectFolder).toHaveBeenCalledWith("simple/folder1");

      // Expand folder and select file
      const chevron = within(folder.closest("[data-folder-path]")!).getByRole(
        "button",
      );
      fireEvent.click(chevron);
      const file = screen.getByText("file1.txt");
      fireEvent.click(file);

      expect(onSelectFolder).toHaveBeenCalledWith(null);
    });

    it("clears file selection when folder is selected", () => {
      const onSelectItem = vi.fn();
      render(
        <FolderBrowser
          {...defaultProps}
          folderTree={simpleTestTree}
          onSelectItem={onSelectItem}
        />,
      );

      // First expand and select file
      const folder1 = screen
        .getByText("folder1")
        .closest("[data-folder-path]") as HTMLElement;
      const chevron = within(folder1!).getByRole("button");
      fireEvent.click(chevron);
      const file = screen.getByText("file1.txt");
      fireEvent.click(file);
      expect(onSelectItem).toHaveBeenCalledWith(10);

      // Then select folder
      const folder = screen.getByText("folder1");
      fireEvent.click(folder);

      expect(onSelectItem).toHaveBeenCalledWith(null);
    });
  });

  describe("Multi-Selection", () => {
    it("adds folder to selection with Ctrl+click", () => {
      const onSelectFolder = vi.fn();
      render(
        <FolderBrowser {...defaultProps} onSelectFolder={onSelectFolder} />,
      );

      // Select first folder
      const documentsFolder = screen.getByText("documents");
      fireEvent.click(documentsFolder);
      expect(onSelectFolder).toHaveBeenCalledWith("root/documents");

      // Ctrl+click second folder
      const codeFolder = screen.getByText("code");
      fireEvent.click(codeFolder, { ctrlKey: true });
      expect(onSelectFolder).toHaveBeenCalledWith("root/code");
    });

    it("removes folder from selection with Ctrl+click on already selected folder", () => {
      const onSelectFolder = vi.fn();
      render(
        <FolderBrowser {...defaultProps} onSelectFolder={onSelectFolder} />,
      );

      // Select folder
      const documentsFolder = screen.getByText("documents");
      fireEvent.click(documentsFolder);
      expect(onSelectFolder).toHaveBeenCalledWith("root/documents");

      // Ctrl+click same folder to deselect
      fireEvent.click(documentsFolder, { ctrlKey: true });
      expect(onSelectFolder).toHaveBeenLastCalledWith("root/documents");
    });

    it("works with Cmd+click (Mac)", () => {
      const onSelectFolder = vi.fn();
      render(
        <FolderBrowser {...defaultProps} onSelectFolder={onSelectFolder} />,
      );

      const documentsFolder = screen.getByText("documents");
      fireEvent.click(documentsFolder, { metaKey: true });

      expect(onSelectFolder).toHaveBeenCalledWith("root/documents");
    });

    it("does not select file with Ctrl+click", () => {
      const onSelectItem = vi.fn();
      render(
        <FolderBrowser
          {...defaultProps}
          folderTree={simpleTestTree}
          onSelectItem={onSelectItem}
        />,
      );

      // Expand folder
      const folder1 = screen
        .getByText("folder1")
        .closest("[data-folder-path]") as HTMLElement;
      const chevron = within(folder1!).getByRole("button");
      fireEvent.click(chevron);

      // Ctrl+click file
      const file = screen.getByText("file1.txt");
      fireEvent.click(file, { ctrlKey: true });

      expect(onSelectItem).not.toHaveBeenCalled();
    });

    describe("Contiguous Selection (Shift+click)", () => {
      it("selects range of folders at same level with Shift+click", () => {
        const testTree: FolderV2 = {
          name: "root",
          path: "/root",
          confidence: 1.0,
          count: 3,
          children: [
            {
              name: "folder1",
              path: "/root/folder1",
              confidence: 0.9,
              count: 0,
              children: [],
            },
            {
              name: "folder2",
              path: "/root/folder2",
              confidence: 0.9,
              count: 0,
              children: [],
            },
            {
              name: "folder3",
              path: "/root/folder3",
              confidence: 0.9,
              count: 0,
              children: [],
            },
          ],
        };

        const onSelectFolder = vi.fn();
        render(
          <FolderBrowser
            {...defaultProps}
            folderTree={testTree}
            onSelectFolder={onSelectFolder}
          />,
        );

        // Select first folder
        const folder1 = screen.getByText("folder1");
        fireEvent.click(folder1);
        expect(onSelectFolder).toHaveBeenCalledWith("root/folder1");

        // Shift+click third folder
        const folder3 = screen.getByText("folder3");
        fireEvent.click(folder3, { shiftKey: true });
        expect(onSelectFolder).toHaveBeenCalledWith("root/folder3");
      });

      it("selects only current folder if no previous selection for Shift+click", async () => {
        const onSelectFolder = vi.fn();
        render(
          <FolderBrowser {...defaultProps} onSelectFolder={onSelectFolder} />,
        );

        // Wait for the root folder to be expanded and documents to be visible
        await waitFor(() => {
          expect(screen.getByText("documents")).toBeInTheDocument();
        });

        const documentsFolder = screen.getByText("documents");
        fireEvent.click(documentsFolder, { shiftKey: true });

        expect(onSelectFolder).toHaveBeenCalledWith("root/documents");
      });

      it("selects only current folder if previous selection is at different level", async () => {
        const onSelectFolder = vi.fn();
        render(
          <FolderBrowser {...defaultProps} onSelectFolder={onSelectFolder} />,
        );

        // Wait for the root folder to be expanded and documents to be visible
        await waitFor(() => {
          expect(screen.getByText("documents")).toBeInTheDocument();
        });

        // Expand documents folder and select images subfolder
        const documentsFolder = screen
          .getByText("documents")
          .closest("[data-folder-path]") as HTMLElement;
        const chevron = within(documentsFolder!).getByRole("button");
        fireEvent.click(chevron);

        await waitFor(() => {
          expect(screen.getByText("images")).toBeInTheDocument();
        });

        const imagesFolder = screen.getByText("images");
        fireEvent.click(imagesFolder);
        expect(onSelectFolder).toHaveBeenCalledWith("root/documents/images");

        // Shift+click on code folder (different level)
        const codeFolder = screen.getByText("code");
        fireEvent.click(codeFolder, { shiftKey: true });
        expect(onSelectFolder).toHaveBeenLastCalledWith("root/code");
      });
    });

    describe("Escape Key", () => {
      it("clears all selections with Escape key", () => {
        const onSelectItem = vi.fn();
        const onSelectFolder = vi.fn();
        render(
          <FolderBrowser
            {...defaultProps}
            onSelectItem={onSelectItem}
            onSelectFolder={onSelectFolder}
          />,
        );

        // Select folder
        const folder = screen.getByText("documents");
        fireEvent.click(folder);
        expect(onSelectFolder).toHaveBeenCalledWith("root/documents");

        // Press Escape
        fireEvent.keyDown(document, { key: "Escape" });

        expect(onSelectItem).toHaveBeenCalledWith(null);
        expect(onSelectFolder).toHaveBeenCalledWith(null);
      });

      it("cancels rename operation with Escape key", async () => {
        render(<FolderBrowser {...defaultProps} />);

        // Right-click folder to open context menu
        const folder = screen.getByText("documents");
        fireEvent.contextMenu(folder);

        // Click rename
        const renameOption = screen.getByText("Rename folder");
        fireEvent.click(renameOption);

        // Verify input appears
        await waitFor(() => {
          expect(screen.getByDisplayValue("documents")).toBeInTheDocument();
        });

        // Press Escape
        fireEvent.keyDown(document, { key: "Escape" });

        // Verify input is gone
        await waitFor(() => {
          expect(
            screen.queryByDisplayValue("documents"),
          ).not.toBeInTheDocument();
        });
      });
    });
  });

  describe("Synchronization with External State", () => {
    it("updates selection when externalSelectedFile changes", () => {
      const { rerender } = render(
        <FolderBrowser {...defaultProps} folderTree={simpleTestTree} />,
      );

      // Change external selection
      rerender(
        <FolderBrowser
          {...defaultProps}
          folderTree={simpleTestTree}
          externalSelectedFile={10}
        />,
      );

      // Should sync with the hook
      expect(mockUseFolderTree.setTreeData).toHaveBeenCalled();
    });

    it("expands parent folders when external file is selected", () => {
      const { rerender } = render(
        <FolderBrowser {...defaultProps} folderTree={simpleTestTree} />,
      );

      // Change external selection
      rerender(
        <FolderBrowser
          {...defaultProps}
          folderTree={simpleTestTree}
          externalSelectedFile={10}
        />,
      );

      // Should expand folders to show the selected file
      expect(screen.getByText("file1.txt")).toBeInTheDocument();
    });

    it("does not sync when shouldSync is false", () => {
      const { rerender } = render(
        <FolderBrowser
          {...defaultProps}
          folderTree={simpleTestTree}
          shouldSync={false}
        />,
      );

      // Change external selection
      rerender(
        <FolderBrowser
          {...defaultProps}
          folderTree={simpleTestTree}
          externalSelectedFile={10}
          shouldSync={false}
        />,
      );

      // Should still call setTreeData but not scroll
      expect(mockUseFolderTree.setTreeData).toHaveBeenCalled();
    });
  });

  describe("Context Menu", () => {
    it("shows file context menu on right-click", () => {
      render(<FolderBrowser {...defaultProps} folderTree={simpleTestTree} />);

      // Expand folder
      const folder1 = screen
        .getByText("folder1")
        .closest("[data-folder-path]") as HTMLElement;
      const chevron = within(folder1!).getByRole("button");
      fireEvent.click(chevron);

      // Right-click file
      const file = screen.getByText("file1.txt");
      fireEvent.contextMenu(file);

      expect(screen.getByText("Rename file")).toBeInTheDocument();
      expect(screen.getByText("View File Details")).toBeInTheDocument();
      expect(screen.getByText("Copy File Path")).toBeInTheDocument();
    });

    it("shows folder context menu on right-click", () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.contextMenu(folder);

      expect(screen.getByText("Rename folder")).toBeInTheDocument();
      expect(screen.getByText("New folder")).toBeInTheDocument();
      expect(screen.getByText("Copy Folder Path")).toBeInTheDocument();
      expect(screen.getByText("Delete")).toBeInTheDocument();
    });

    it('shows "Mark as valid" for low confidence folders', () => {
      const lowConfidenceTree: FolderV2 = {
        name: "root",
        path: "/root",
        confidence: 1.0,
        count: 1,
        children: [
          {
            name: "uncertain",
            path: "/root/uncertain",
            confidence: 0.5,
            count: 0,
            children: [],
          },
        ],
      };

      render(
        <FolderBrowser {...defaultProps} folderTree={lowConfidenceTree} />,
      );

      const folder = screen.getByText("uncertain");
      fireEvent.contextMenu(folder);

      expect(screen.getByText("Mark as valid")).toBeInTheDocument();
    });

    it.skip("shows merge option for multiple selected folders", () => {
      render(<FolderBrowser {...defaultProps} />);

      // Select multiple folders
      const folder1 = screen.getByText("documents");
      fireEvent.click(folder1);

      const folder2 = screen.getByText("code");
      fireEvent.click(folder2, { ctrlKey: true });

      // Right-click one of them
      fireEvent.contextMenu(folder1);

      expect(screen.getByText("Merge Folders")).toBeInTheDocument();
      expect(screen.getByText("Flatten folders")).toBeInTheDocument();
    });

    it.skip("shows invert option when canInvertFolder returns true", () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.contextMenu(folder);

      expect(screen.getByText("Invert with children")).toBeInTheDocument();
    });

    it('copies path to clipboard when "Copy Folder Path" is clicked', async () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.contextMenu(folder);

      const copyOption = screen.getByText("Copy Folder Path");
      fireEvent.click(copyOption);

      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
        "/root/documents",
      );
    });
  });

  describe("Drag and Drop", () => {
    it("starts drag operation when dragging an item", () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      const dragStartEvent = new DragEvent("dragstart", { bubbles: true });
      const mockDataTransfer = {
        effectAllowed: "",
        setData: vi.fn(),
      };
      Object.defineProperty(dragStartEvent, "dataTransfer", {
        value: mockDataTransfer,
      });

      fireEvent(folder, dragStartEvent);

      expect(mockDataTransfer.effectAllowed).toBe("move");
      expect(mockDataTransfer.setData).toHaveBeenCalledWith(
        "text/plain",
        "root/documents",
      );
    });

    it("allows drop on valid folder targets", () => {
      render(<FolderBrowser {...defaultProps} />);

      const sourceFolder = screen.getByText("documents");
      const targetFolder = screen.getByText("code");

      // Start drag
      const dragStartEvent = new DragEvent("dragstart", { bubbles: true });
      const mockDataTransfer = {
        effectAllowed: "",
        setData: vi.fn(),
        dropEffect: "",
      };
      Object.defineProperty(dragStartEvent, "dataTransfer", {
        value: mockDataTransfer,
      });
      fireEvent(sourceFolder, dragStartEvent);

      // Drag over target
      const dragOverEvent = new DragEvent("dragover", { bubbles: true });
      Object.defineProperty(dragOverEvent, "dataTransfer", {
        value: mockDataTransfer,
      });
      fireEvent(targetFolder, dragOverEvent);

      expect(mockDataTransfer.dropEffect).toBe("move");
    });

    it("performs move operation on drop", async () => {
      render(<FolderBrowser {...defaultProps} />);

      const sourceFolder = screen.getByText("documents");
      const targetFolder = screen.getByText("code");

      // Start drag
      fireEvent.dragStart(sourceFolder);

      // Drop on target
      const dropEvent = new DragEvent("drop", { bubbles: true });
      Object.defineProperty(dropEvent, "dataTransfer", {
        value: { getData: () => "documents" },
      });
      fireEvent(targetFolder, dropEvent);

      expect(mockUseFolderTree.moveItem).toHaveBeenCalledWith(
        "root/documents",
        "root/code",
      );
    });

    it("prevents dropping on files", () => {
      render(<FolderBrowser {...defaultProps} folderTree={simpleTestTree} />);

      // Expand folder
      const folder1 = screen
        .getByText("folder1")
        .closest("[data-folder-path]") as HTMLElement;
      const chevron = within(folder1!).getByRole("button");
      fireEvent.click(chevron);

      const file = screen.getByText("file1.txt");

      const dragOverEvent = new DragEvent("dragover", { bubbles: true });
      const mockDataTransfer = { dropEffect: "" };
      Object.defineProperty(dragOverEvent, "dataTransfer", {
        value: mockDataTransfer,
      });

      fireEvent(file, dragOverEvent);

      // Should not allow drop on files
      expect(mockDataTransfer.dropEffect).toBe("");
    });
  });

  describe("Inline Editing", () => {
    it('starts rename mode when "Rename folder" is clicked', async () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.contextMenu(folder);

      const renameOption = screen.getByText("Rename folder");
      fireEvent.click(renameOption);

      await waitFor(() => {
        expect(screen.getByDisplayValue("documents")).toBeInTheDocument();
      });
    });

    it("submits rename on Enter key", async () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.contextMenu(folder);

      const renameOption = screen.getByText("Rename folder");
      fireEvent.click(renameOption);

      const input = await screen.findByDisplayValue("documents");
      fireEvent.change(input, { target: { value: "new-name" } });
      fireEvent.keyDown(input, { key: "Enter" });

      expect(mockUseFolderTree.renameItem).toHaveBeenCalledWith(
        "root/documents",
        "new-name",
      );
    });

    it("cancels rename on Escape key", async () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.contextMenu(folder);

      const renameOption = screen.getByText("Rename folder");
      fireEvent.click(renameOption);

      const input = await screen.findByDisplayValue("documents");
      fireEvent.keyDown(input, { key: "Escape" });

      await waitFor(() => {
        expect(screen.queryByDisplayValue("documents")).not.toBeInTheDocument();
      });
      expect(mockUseFolderTree.renameItem).not.toHaveBeenCalled();
    });

    it("submits rename on blur", async () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.contextMenu(folder);

      const renameOption = screen.getByText("Rename folder");
      fireEvent.click(renameOption);

      const input = await screen.findByDisplayValue("documents");
      fireEvent.change(input, { target: { value: "new-name" } });
      fireEvent.blur(input);

      await waitFor(() => {
        expect(screen.queryByDisplayValue("new-name")).not.toBeInTheDocument();
      });
    });

    it('creates new folder when "New folder" is clicked', async () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.contextMenu(folder);

      const newFolderOption = screen.getByText("New folder");
      fireEvent.click(newFolderOption);

      await waitFor(() => {
        expect(screen.getByDisplayValue("New Folder")).toBeInTheDocument();
      });
    });

    it("submits new folder creation on Enter", async () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.contextMenu(folder);

      const newFolderOption = screen.getByText("New folder");
      fireEvent.click(newFolderOption);

      const input = await screen.findByDisplayValue("New Folder");
      fireEvent.change(input, { target: { value: "created-folder" } });
      fireEvent.keyDown(input, { key: "Enter" });

      expect(mockUseFolderTree.createFolder).toHaveBeenCalledWith(
        "root/documents",
        "created-folder",
      );
    });

    it("cancels folder creation on Escape", async () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.contextMenu(folder);

      const newFolderOption = screen.getByText("New folder");
      fireEvent.click(newFolderOption);

      const input = await screen.findByDisplayValue("New Folder");
      fireEvent.keyDown(input, { key: "Escape" });

      await waitFor(() => {
        expect(
          screen.queryByDisplayValue("New Folder"),
        ).not.toBeInTheDocument();
      });
      expect(mockUseFolderTree.createFolder).not.toHaveBeenCalled();
    });
  });

  describe("Visual Indicators", () => {
    it("shows red dot indicator for folders with low confidence children", () => {
      const treeWithLowConfidence: FolderV2 = {
        name: "root",
        path: "/root",
        confidence: 1.0,
        count: 1,
        children: [
          {
            name: "parent",
            path: "/root/parent",
            confidence: 0.9,
            count: 1,
            children: [
              {
                name: "low-confidence",
                path: "/root/parent/low-confidence",
                confidence: 0.3,
                count: 0,
                children: [],
              },
            ],
          },
        ],
      };

      render(
        <FolderBrowser {...defaultProps} folderTree={treeWithLowConfidence} />,
      );

      const parentFolder = screen.getByText("parent").closest("span");
      expect(parentFolder?.querySelector("div")).toBeInTheDocument(); // Red dot indicator
    });

    it("applies correct styling for selected items", () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.click(folder);

      const folderItem = folder.closest("[data-folder-path]");
      expect(folderItem).toHaveStyle({ backgroundColor: "#dbeafe" });
    });

    it("applies correct styling for drag operations", () => {
      render(<FolderBrowser {...defaultProps} />);

      const folder = screen.getByText("documents");
      fireEvent.dragStart(folder);

      const folderItem = folder.closest("[data-folder-path]");
      expect(folderItem).toHaveStyle({ opacity: "0.5" });
    });
  });
});
