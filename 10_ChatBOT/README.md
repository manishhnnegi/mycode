src/
├── components/
│   ├── ChatApp.js
│   ├── Message.js
│   ├── MessageInput.js
│   └── Suggestions.js
├── App.js
├── App.css
└── index.js


# format of the folder structure:
# -------------------------------------------------
# -------------------------------------------------


your-react-app/
│
├── public/
│   ├── index.html
│   └── ... (other public files)
│
├── src/
│   ├── assets/             // Folder for all asset files (images, logos, etc.)
│   │   ├── user-avatar.png // User avatar image
│   │   └── chatgpt-avatar.png // ChatGPT avatar image
│   │
│   ├── components/         // Folder for your React components
│   │   ├── ChatApp.js      // ChatApp component file
│   │   └── ... (other components)
│   │
│   ├── App.js              // Main application file
│   ├── index.js            // Entry point of your app
│   └── ... (other source files)
│
└── package.json



# ------------------------------------------------------------

uvicorn backend.main:app --reload