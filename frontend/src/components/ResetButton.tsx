import React from 'react';

interface ResetButtonProps {
  onReset: () => void;
}

export const ResetButton: React.FC<ResetButtonProps> = ({ onReset }) => (
  <button
    onClick={onReset}
    className="bg-red-500 text-white py-2 px-4 rounded-md border-none cursor-pointer text-sm transition-colors hover:bg-red-600 focus:outline-2 focus:outline-red-600 focus:outline-offset-2"
  >
    Reset to Initial Data
  </button>
);