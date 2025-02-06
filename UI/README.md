# DL-GPT Web Application

## Overview
This web application provides an interactive interface for interacting with the DL-GPT model. Users can test both the classification and assistant functionalities through a simple and intuitive UI.

---

## Features
- **Spam Classification**: Classify messages as spam or non-spam.
- **Assistant Mode**: Input queries and receive responses from the fine-tuned GPT model.
- **Real-time Inference**: Communicates with the backend API for model predictions.

---

## Tech Stack
- **Frontend**: Vue.js 3
- **Backend**: Flask API
- **Containerization**: Docker (Future Work)

---

## Setup Instructions
1. Clone the repository:
   ```sh
   git clone https://github.com/Joancf1997/DL-GPT-WebApp.git
   ```
2. Navigate to the project folder:
   ```sh
   cd UI
   ```
3. Install dependencies:
   ```sh
   npm install 
   ```
4. Run the frontend:
   ```sh
   npm run dev
   ```
5. Start the backend:
   ```sh
   cd backend
   python app.py
   ```


---

## Future Improvements
- Deploy the application with Docker
- Enhance UI for better user experience
- Expand assistant capabilities
