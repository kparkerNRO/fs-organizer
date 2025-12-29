import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "../test/utils";
import { Pagination } from "./Pagination";

describe("Pagination", () => {
  const defaultProps = {
    currentPage: 1,
    totalPages: 5,
    pageSize: 10,
    totalItems: 50,
    onPageChange: vi.fn(),
    onPageSizeChange: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders pagination info correctly", () => {
    render(<Pagination {...defaultProps} />);

    expect(screen.getByText("Showing 1 to 10 of 50 items")).toBeInTheDocument();
  });

  it("renders page size selector", () => {
    render(<Pagination {...defaultProps} />);

    const select = screen.getByDisplayValue("10");
    expect(select).toBeInTheDocument();
    expect(screen.getByText("Show:")).toBeInTheDocument();
    expect(screen.getByText("items")).toBeInTheDocument();
  });

  it("calls onPageSizeChange when page size is changed", () => {
    render(<Pagination {...defaultProps} />);

    const select = screen.getByDisplayValue("10");
    fireEvent.change(select, { target: { value: "20" } });

    expect(defaultProps.onPageSizeChange).toHaveBeenCalledWith(20);
  });

  it("renders page numbers correctly", () => {
    render(<Pagination {...defaultProps} />);

    // Get all page number buttons specifically by role and aria-label or text content
    const pageButtons = screen
      .getAllByRole("button")
      .filter((button) => /^[1-5]$/.test(button.textContent || ""));

    expect(pageButtons).toHaveLength(5);
    expect(pageButtons[0]).toHaveTextContent("1");
    expect(pageButtons[1]).toHaveTextContent("2");
    expect(pageButtons[2]).toHaveTextContent("3");
    expect(pageButtons[3]).toHaveTextContent("4");
    expect(pageButtons[4]).toHaveTextContent("5");
  });

  it("calls onPageChange when page number is clicked", () => {
    render(<Pagination {...defaultProps} />);

    const pageButtons = screen
      .getAllByRole("button")
      .filter((button) => button.textContent === "3");
    fireEvent.click(pageButtons[0]);

    expect(defaultProps.onPageChange).toHaveBeenCalledWith(3);
  });

  it("disables previous button on first page", () => {
    render(<Pagination {...defaultProps} currentPage={1} />);

    const prevButton = screen.getAllByRole("button")[0]; // First button is previous
    expect(prevButton).toBeDisabled();
  });

  it("disables next button on last page", () => {
    render(<Pagination {...defaultProps} currentPage={5} />);

    const buttons = screen.getAllByRole("button");
    const nextButton = buttons[buttons.length - 1]; // Last button is next
    expect(nextButton).toBeDisabled();
  });

  it("calls onPageChange when navigation buttons are clicked", () => {
    render(<Pagination {...defaultProps} currentPage={3} />);

    const buttons = screen.getAllByRole("button");
    const prevButton = buttons[0];
    const nextButton = buttons[buttons.length - 1];

    fireEvent.click(prevButton);
    expect(defaultProps.onPageChange).toHaveBeenCalledWith(2);

    fireEvent.click(nextButton);
    expect(defaultProps.onPageChange).toHaveBeenCalledWith(4);
  });

  it("shows correct pagination info for different pages", () => {
    render(
      <Pagination
        {...defaultProps}
        currentPage={3}
        pageSize={5}
        totalItems={23}
      />,
    );

    expect(
      screen.getByText("Showing 11 to 15 of 23 items"),
    ).toBeInTheDocument();
  });

  it("handles last page correctly when items dont fill page", () => {
    render(
      <Pagination
        {...defaultProps}
        currentPage={5}
        pageSize={10}
        totalItems={47}
      />,
    );

    expect(
      screen.getByText("Showing 41 to 47 of 47 items"),
    ).toBeInTheDocument();
  });
});
