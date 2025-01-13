import React from "react";

interface DetailPanelProps {
  rowData: any;
}

export const DetailPanel: React.FC<DetailPanelProps> = ({ rowData }) => {
  return (
    <div>
      <h3>{rowData.name}</h3>
      <p>Classification: {rowData.classification}</p>
      <p>Confidence: {rowData.confidence}%</p>
      <p>Original Path: {rowData.original_path}</p>
      <input
        type="text"
        defaultValue={rowData.processed_name}
        onBlur={(e) => {
          // Call API to update processed name
          console.log("Update processed name:", e.target.value);
        }}
      />
    </div>
  );
};
