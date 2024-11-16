// src/components/Suggestions.js
import React from 'react';

const Suggestions = () => {
  return (
    <div className="nochat">
      <div className="s1">
        <h1>Suggestions</h1>
      </div>
      <div className="s2">
        <div className="suggestioncard">
          <h2>How do I start a new chat?</h2>
          <p>Click on the new message icon to start a new conversation.</p>
        </div>
        <div className="suggestioncard">
          <h2>What can you help me with?</h2>
          <p>I can assist you with various tasks like answering questions and providing information.</p>
        </div>
      </div>
    </div>
  );
};

export default Suggestions;
