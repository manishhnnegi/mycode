// src/components/ChatApp.js
import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import './ChatApp.css';

// Import logos
import userAvatar from '../assets/user-avatar.png';
import chatgptAvatar from '../assets/chatgpt-avatar.png';

// Check if the browser supports Speech Recognition
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;

function ChatApp() {
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [sessionId, setSessionId] = useState(null);
    const messagesEndRef = useRef(null);
    const [isListening, setIsListening] = useState(false);

    useEffect(() => {
        axios.get('http://127.0.0.1:8000/session')
            .then((response) => setSessionId(response.data))
            .catch((error) => console.error("Error creating session:", error));
    }, []);

    useEffect(() => {
        if (sessionId) {
            axios.get(`http://127.0.0.1:8000/messages/${sessionId}`)
                .then((response) => setMessages(response.data))
                .catch((error) => console.error("Error fetching messages:", error));
        }
    }, [sessionId]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSendMessage = () => {
        if (inputValue.trim() === '' || !sessionId) return;

        const newMessage = {
            session_id: sessionId,
            message: {
                sender: 'User',
                content: inputValue,
            },
        };

        axios.post('http://127.0.0.1:8000/messages', newMessage)
            .then((response) => {
                setMessages((prevMessages) => [...prevMessages, newMessage.message, response.data]);
                setInputValue('');
            })
            .catch((error) => console.error("Error sending message:", error));
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            handleSendMessage();
        }
    };

    const startSpeechRecognition = () => {
        if (recognition && !isListening) {
            setIsListening(true);
            recognition.start();
        }
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInputValue(transcript);
        handleSendMessage();
    };

    recognition.onend = () => setIsListening(false);

    recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
    };

    return (
        <div className="rightSection">
            <h1 className="chatHeading">How Can I Help You?</h1>
            <div className="messages">
                {messages.map((message, index) => (
                    <div key={index} className={`message ${message.sender === 'User' ? 'user-message' : 'gpt-message'}`}>
                        <img
                            src={message.sender === 'User' ? userAvatar : chatgptAvatar}
                            alt={`${message.sender} avatar`}
                            className="avatar"
                        />
                        <div className="message-content">
                            <h2>{message.sender}</h2>
                            <p>{message.content}</p>
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>
            <div className="bottomsection">
                <div className="messagebar">
                    <input
                        type="text"
                        placeholder="Type a message..."
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyPress={handleKeyPress}
                    />
                    <button onClick={handleSendMessage}>Send</button>
                    <button onClick={startSpeechRecognition}>🎤</button>
                </div>
            </div>
        </div>
    );
}

export default ChatApp;
