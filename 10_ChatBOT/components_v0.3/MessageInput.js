// src/components/MessageInput.js
import React, { useState } from 'react';

const MessageInput = () => {
  const [input, setInput] = useState('');

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSendMessage = () => {
    console.log('Sending message:', input);
    setInput('');
  };

  return (
    <div className="messagebar">
      <input
        type="text"
        placeholder="Type a message..."
        value={input}
        onChange={handleInputChange}
        onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
      />
      <svg onClick={handleSendMessage}>{/* Send Icon SVG */}</svg>
    </div>
  );
};

export default MessageInput;
