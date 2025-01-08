import React, { useState } from "react";

interface CategoryTableProps {
  data: any[];
  onRowSelect: (row: any) => void;
}

export const CategoryTable: React.FC<CategoryTableProps> = ({ data, onRowSelect }) => {
  const [sortBy, setSortBy] = useState<string>("name");
  const [filter, setFilter] = useState<string>("");

  const handleSort = (column: string) => setSortBy(column);

  const filteredData = data.filter((item) =>
    item.name.toLowerCase().includes(filter.toLowerCase())
  );

  const sortedData = [...filteredData].sort((a, b) =>
    a[sortBy].localeCompare(b[sortBy])
  );

  return (
    <div>
      <input
        type="text"
        placeholder="Filter by name"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
      />
      <table>
        <thead>
          <tr>
            <th onClick={() => handleSort("name")}>Name</th>
            <th onClick={() => handleSort("classification")}>Classification</th>
            <th onClick={() => handleSort("count")}>Count</th>
            <th onClick={() => handleSort("confidence")}>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {sortedData.map((item) => (
            <tr key={item.id} onClick={() => onRowSelect(item)}>
              <td>{item.name}</td>
              <td>{item.classification}</td>
              <td>{item.count}</td>
              <td>{item.confidence}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
