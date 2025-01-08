import React, { useState, useEffect } from "react";
import { CategoryTable } from "./category_table";
import { DetailPanel } from "./detail_panel";

const App: React.FC = () => {
  const [data, setData] = useState<any[]>([]); // Fetch data from FastAPI
  const [selectedRow, setSelectedRow] = useState(null);

  useEffect(() => {
    // Fetch data from FastAPI
    const fetchData = async () => {
      const response = await fetch("/api/folders");
      const result = await response.json();
      setData(result);
    };
    fetchData();
  }, []);

  return (
    <div>
      <CategoryTable data={data} onRowSelect={setSelectedRow} />
      {selectedRow && <DetailPanel rowData={selectedRow} />}
    </div>
  );
};

export default App;
