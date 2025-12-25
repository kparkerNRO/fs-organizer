import React from "react";
import { CategoryDetailsProps } from "../types/types";
import { FileIcon, FolderIcon } from "lucide-react";

const detailsContainerClass = "p-6 bg-white rounded-lg shadow-sm w-full box-border overflow-x-hidden overflow-y-auto [.folder-structure-page_&]:flex-1 [.folder-structure-page_&]:flex [.folder-structure-page_&]:flex-col [.folder-structure-page_&]:min-h-0 [.folder-structure-page_&]:overflow-y-auto [.folder-structure-page_&]:w-[450px] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-gray-100 [&::-webkit-scrollbar-track]:rounded [&::-webkit-scrollbar-thumb]:bg-gray-400 [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb:hover]:bg-gray-500";

export const CategoryDetails: React.FC<CategoryDetailsProps> = ({
  category,
  folder,
  file
}) => {
  // If no item is selected, return the placeholder
  if (!category && !folder && !file) {
    return (
      <div className={detailsContainerClass}>
        <div className="flex items-center mb-5 pb-3 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900 m-0">Details</h3>
        </div>
        <div className="flex flex-col items-center justify-center flex-1 min-h-[200px]">
          <p className="text-gray-500 text-base">Select a folder or file to view details</p>
        </div>
      </div>
    );
  }

  // Render file details if a file is selected
  if (file) {
    return (
      <div className={detailsContainerClass}>
        <div className="flex items-center mb-5 pb-3 border-b border-gray-200">
          <FileIcon size={18} style={{ marginRight: '0.5rem', opacity: 0.7 }} />
          <h3 className="text-lg font-medium text-gray-900 m-0">File Details</h3>
        </div>

        <div className="grid grid-cols-2 gap-6 mb-6 [.folder-structure-page_&]:grid-cols-1">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">File Name</label>
            <input
              type="text"
              value={file.name || ""}
              readOnly
              className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">File Type</label>
            <input
              type="text"
              value={file.fileType || ""}
              readOnly
              className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </div>

          {file.size && (
            <div className="flex flex-col gap-1">
              <label className="text-sm font-medium text-gray-700">Size</label>
              <input
                type="text"
                value={file.size}
                readOnly
                className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
              />
            </div>
          )}
        </div>

        <div className="mt-4 flex flex-col gap-1 w-full">
          <label className="text-sm font-medium text-gray-700">Original Path</label>
          <input
            type="text"
            value={file.original_path || ""}
            readOnly
            className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
        </div>

        {file.categories && file.categories.length > 0 && (
          <div className="mt-4 flex flex-col gap-1 w-full">
            <label className="text-sm font-medium text-gray-700">Categories</label>
            <input
              type="text"
              value={file.categories.join(", ")}
              readOnly
              className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </div>
        )}

        <div className="mt-4 flex flex-col gap-1 w-full">
          <label className="text-sm font-medium text-gray-700">Confidence</label>
          <div className="w-full h-10 bg-gray-300 rounded-lg overflow-hidden">
            <div
              style={{ width: `${file.confidence}%` }}
              className="h-full bg-gray-500 rounded-lg transition-all duration-300 flex items-center justify-center text-white text-sm"
            >
              {file.confidence}%
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Render folder details if a folder is selected
  if (folder != null) {
    const hasProcessedNames = folder.processed_names && folder.processed_names.length > 0;

    return (
      <div className={detailsContainerClass}>
        <div className="flex items-center mb-5 pb-3 border-b border-gray-200">
          <FolderIcon size={18} style={{ marginRight: '0.5rem', opacity: 0.7 }} />
          <h3 className="text-lg font-medium text-gray-900 m-0">Folder Details</h3>
        </div>

        <div className="grid grid-cols-2 gap-6 mb-6 [.folder-structure-page_&]:grid-cols-1">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">Category Name</label>
            <input
              type="text"
              value={category?.name || ""}
              readOnly
              className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">Name:</label>
            <input
              type="text"
              value={folder.name || ""}
              readOnly
              className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">Classification</label>
            <select
              className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            >
              <option value={folder.classification}>
                {folder.classification}
              </option>
            </select>
          </div>
        </div>

        <div className="mt-4 flex flex-col gap-1 w-full">
          <label className="text-sm font-medium text-gray-700">Original Filename</label>
          <input
            type="text"
            value={folder.original_filename || ""}
            readOnly
            className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
        </div>

        <div className="mt-4 flex flex-col gap-1 w-full">
          <label className="text-sm font-medium text-gray-700">Original Path</label>
          <input
            type="text"
            value={folder.original_path || ""}
            readOnly
            className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
        </div>

        {hasProcessedNames && (
          <div className="mt-4 flex flex-col gap-1 w-full">
            <label className="text-sm font-medium text-gray-700">Categories</label>
            <input
              type="text"
              value={folder.processed_names!.join(", ")}
              readOnly
              className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </div>
        )}
        {!hasProcessedNames && <div className="h-20 w-full" />}

        <div className="mt-4 flex flex-col gap-1 w-full">
          <label className="text-sm font-medium text-gray-700">Confidence</label>
          <div className="w-full h-10 bg-gray-300 rounded-lg overflow-hidden">
            <div
              style={{ width: `${folder.confidence}%` }}
              className="h-full bg-gray-500 rounded-lg transition-all duration-300 flex items-center justify-center text-white text-sm"
            >
              {folder.confidence}%
            </div>
          </div>
        </div>

        {hasProcessedNames && (
          <div className="mt-4">
            <label className="text-sm font-medium text-gray-700">Processed Names</label>
            <textarea
              value={folder.processed_names!.join("\n")}
              readOnly
              className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100 min-h-[100px] resize-y"
            />
          </div>
        )}
        {!hasProcessedNames && <div className="h-20 w-full" />}
      </div>
    );
  }

  if (category != null) {
    return (
      <div className={detailsContainerClass}>
        <div className="flex items-center mb-5 pb-3 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900 m-0">Category Details</h3>
        </div>

        <div className="grid grid-cols-2 gap-6 mb-6 [.folder-structure-page_&]:grid-cols-1">
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">Category Name</label>
            <input
              type="text"
              value={category.name || ""}
              readOnly
              className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            />
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">Classification</label>
            <select
              className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
            >
              <option value={category.classification}>
                {category.classification}
              </option>
            </select>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700">Confidence</label>
            <div className="w-full h-10 bg-gray-300 rounded-lg overflow-hidden">
              <div
                style={{ width: `${category.confidence}%` }}
                className="h-full bg-gray-500 rounded-lg transition-all duration-300 flex items-center justify-center text-white text-sm"
              >
                {category.confidence}%
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 flex flex-col gap-1 w-full">
          <label className="text-sm font-medium text-gray-700">Possible Classifications</label>
          <input
            type="text"
            value={category.possibleClassifications?.join(", ") || ""}
            readOnly
            className="w-full py-2 px-3 border border-gray-200 rounded-lg outline-none box-border focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
        </div>
      </div>
    );
  }

  if (!category && !folder) {
    return (
      <div className={detailsContainerClass}>
        <p className="text-gray-500 text-base">Select a category to view details</p>
      </div>
    );
  }
};
