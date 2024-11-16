// src/components/Message.js
import React from 'react';

const Message = ({ user, text }) => {
  return (
    <div className="message">
      <img src={`https://ui-avatars.com/api/?name=${user}`} alt={`${user} avatar`} />
      <div>
        <h2>{user}</h2>
        <p>{text}</p>
      </div>
    </div>
  );
};

export default Message;
