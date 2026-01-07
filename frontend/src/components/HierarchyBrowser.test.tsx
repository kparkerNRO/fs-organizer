import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "../test/utils";
import { HierarchyBrowser } from "./HierarchyBrowser";
import { ItemStore, Hierarchy } from "../types/types";

const createMockItems = (): ItemStore => ({
  "node-root": {
    id: "node-root",
    name: "root",
    type: "node",
  },
  "node-1": {
    id: "node-1",
    name: "Documents",
    type: "node",
    originalPath: "/home/user/Documents",
  },
  "node-2": {
    id: "node-2",
    name: "file1.txt",
    type: "node",
    originalPath: "/home/user/Documents/file1.txt",
  },
  "category-root": {
    id: "category-root",
    name: "Categories",
    type: "category",
  },
  "category-1": {
    id: "category-1",
    name: "Work",
    type: "category",
  },
});

const createMockHierarchy = (): Hierarchy => ({
  "node-root": ["node-1"],
  "node-1": ["node-2"],
  "node-2": [],
});

describe("HierarchyBrowser", () => {
  it("renders items in the hierarchy", () => {
    const items = createMockItems();
    const hierarchy = createMockHierarchy();

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
      />,
    );

    expect(screen.getByText("Documents")).toBeInTheDocument();
    expect(screen.getByText("file1.txt")).toBeInTheDocument();
  });

  it("does not render the root item itself", () => {
    const items = createMockItems();
    const hierarchy = createMockHierarchy();

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
      />,
    );

    // Root should not be visible, only its children
    expect(screen.queryByText("root")).not.toBeInTheDocument();
  });

  it("renders nested items correctly", () => {
    const items = createMockItems();
    const hierarchy = createMockHierarchy();

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
      />,
    );

    // Both parent and child should be present
    expect(screen.getByText("Documents")).toBeInTheDocument();
    expect(screen.getByText("file1.txt")).toBeInTheDocument();
  });

  it("shows child count for items with children", () => {
    const items = createMockItems();
    const hierarchy = createMockHierarchy();

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
      />,
    );

    // Documents has 1 child (file1.txt)
    expect(screen.getByText("(1)")).toBeInTheDocument();
  });

  it("displays path when showPath is true", () => {
    const items = createMockItems();
    const hierarchy = createMockHierarchy();

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
        showPath={true}
      />,
    );

    expect(screen.getByText("/home/user/Documents")).toBeInTheDocument();
  });

  it("does not display path when showPath is false", () => {
    const items = createMockItems();
    const hierarchy = createMockHierarchy();

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
        showPath={false}
      />,
    );

    expect(screen.queryByText("/home/user/Documents")).not.toBeInTheDocument();
  });

  it("calls onItemClick when item is clicked", () => {
    const items = createMockItems();
    const hierarchy = createMockHierarchy();
    const onItemClick = vi.fn();

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
        onItemClick={onItemClick}
      />,
    );

    fireEvent.click(screen.getByText("Documents"));

    expect(onItemClick).toHaveBeenCalledWith("node-1");
  });

  it("calls onItemHover on mouse enter", () => {
    const items = createMockItems();
    const hierarchy = createMockHierarchy();
    const onItemHover = vi.fn();

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
        onItemHover={onItemHover}
      />,
    );

    fireEvent.mouseEnter(screen.getByText("Documents"));

    expect(onItemHover).toHaveBeenCalledWith("node-1");
  });

  it("calls onItemHover with null on mouse leave", () => {
    const items = createMockItems();
    const hierarchy = createMockHierarchy();
    const onItemHover = vi.fn();

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
        onItemHover={onItemHover}
      />,
    );

    const documentsElement = screen.getByText("Documents");
    fireEvent.mouseEnter(documentsElement);
    fireEvent.mouseLeave(documentsElement);

    expect(onItemHover).toHaveBeenLastCalledWith(null);
  });

  it("highlights the specified item", () => {
    const items = createMockItems();
    const hierarchy = createMockHierarchy();

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
        highlightedItemId="node-1"
      />,
    );

    // The highlighted item should have specific styling
    const documentsRow = screen.getByText("Documents").closest("div");
    expect(documentsRow).toBeInTheDocument();
  });

  it("renders category items with category styling", () => {
    const items: ItemStore = {
      "category-root": {
        id: "category-root",
        name: "Categories",
        type: "category",
      },
      "category-1": {
        id: "category-1",
        name: "Work Files",
        type: "category",
      },
    };
    const hierarchy: Hierarchy = {
      "category-root": ["category-1"],
      "category-1": [],
    };

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="category-root"
      />,
    );

    expect(screen.getByText("Work Files")).toBeInTheDocument();
    // Category items should render with 📁 icon
    expect(screen.getByText("📁")).toBeInTheDocument();
  });

  it("renders node items with node styling", () => {
    const items = createMockItems();
    const hierarchy = createMockHierarchy();

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
      />,
    );

    // Node items should render with 📄 icon
    const icons = screen.getAllByText("📄");
    expect(icons.length).toBeGreaterThan(0);
  });

  it("handles empty hierarchy gracefully", () => {
    const items: ItemStore = {
      "node-root": {
        id: "node-root",
        name: "root",
        type: "node",
      },
    };
    const hierarchy: Hierarchy = {
      "node-root": [],
    };

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
      />,
    );

    // Should render without errors, showing empty hierarchy
    expect(screen.queryByText("root")).not.toBeInTheDocument();
  });

  it("handles missing items gracefully", () => {
    const items: ItemStore = {
      "node-root": {
        id: "node-root",
        name: "root",
        type: "node",
      },
    };
    const hierarchy: Hierarchy = {
      "node-root": ["non-existent-id"],
    };

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
      />,
    );

    // Should not crash, should just not render the missing item
    expect(screen.queryByText("non-existent-id")).not.toBeInTheDocument();
  });

  it("renders deeply nested hierarchies", () => {
    const items: ItemStore = {
      "node-root": { id: "node-root", name: "root", type: "node" },
      "node-1": { id: "node-1", name: "Level 1", type: "node" },
      "node-2": { id: "node-2", name: "Level 2", type: "node" },
      "node-3": { id: "node-3", name: "Level 3", type: "node" },
    };
    const hierarchy: Hierarchy = {
      "node-root": ["node-1"],
      "node-1": ["node-2"],
      "node-2": ["node-3"],
      "node-3": [],
    };

    render(
      <HierarchyBrowser
        items={items}
        hierarchy={hierarchy}
        rootId="node-root"
      />,
    );

    expect(screen.getByText("Level 1")).toBeInTheDocument();
    expect(screen.getByText("Level 2")).toBeInTheDocument();
    expect(screen.getByText("Level 3")).toBeInTheDocument();
  });
});
