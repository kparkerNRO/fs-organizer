import React from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  pageSize: number;
  totalItems: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
}

export const Pagination: React.FC<PaginationProps> = ({
  currentPage,
  totalPages,
  pageSize,
  totalItems,
  onPageChange,
  onPageSizeChange,
}) => {
  const pageSizeOptions = [5, 10, 20];

  // Generate array of page numbers to show
  const getPageNumbers = () => {
    const pages = [];
    const showPages = 5; // Number of page buttons to show
    let start = Math.max(1, currentPage - 2);
    const end = Math.min(totalPages, start + showPages - 1);

    if (end - start + 1 < showPages) {
      start = Math.max(1, end - showPages + 1);
    }

    for (let i = start; i <= end; i++) {
      pages.push(i);
    }
    return pages;
  };

  return (
    <div className="flex items-center justify-between p-4 border-t border-gray-200">
      <div className="flex items-center gap-2 text-gray-500 text-sm">
        <span>Show:</span>
        <select
          value={pageSize}
          onChange={(e) => onPageSizeChange(Number(e.target.value))}
          className="py-1 px-2 border border-gray-200 rounded outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
        >
          {pageSizeOptions.map((size) => (
            <option key={size} value={size}>
              {size}
            </option>
          ))}
        </select>
        <span>items</span>
      </div>

      <div className="text-gray-500 text-sm">
        Showing {Math.min((currentPage - 1) * pageSize + 1, totalItems)} to{' '}
        {Math.min(currentPage * pageSize, totalItems)} of {totalItems} items
      </div>

      <div className="flex items-center gap-1">
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          className={`p-1.5 border border-gray-200 rounded bg-white ${
            currentPage === 1
              ? 'text-gray-300 cursor-not-allowed'
              : 'text-gray-700 cursor-pointer hover:bg-gray-100'
          }`}
        >
          <ChevronLeft size={16} />
        </button>

        {getPageNumbers().map((page) => (
          <button
            key={page}
            onClick={() => onPageChange(page)}
            className={`py-1.5 px-3 border rounded cursor-pointer ${
              page === currentPage
                ? 'border-blue-500 bg-blue-500 text-white hover:bg-blue-600'
                : 'border-gray-200 bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            {page}
          </button>
        ))}

        <button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          className={`p-1.5 border border-gray-200 rounded bg-white ${
            currentPage === totalPages
              ? 'text-gray-300 cursor-not-allowed'
              : 'text-gray-700 cursor-pointer hover:bg-gray-100'
          }`}
        >
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  );
};