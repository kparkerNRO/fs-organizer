
import React, { useState, useEffect } from "react";
import { CategoryTable } from "./components/CategoryTable";
import { DetailPanel } from "./components/DetailPanel";
import { fetchCategories } from "./api";

const App: React.FC = () => {
  const [data, setData] = useState<any[]>([]); // Fetch data from FastAPI
  const [selectedRow, setSelectedRow] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      // Fetch data from FastAPI
      setData(await fetchCategories());
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
