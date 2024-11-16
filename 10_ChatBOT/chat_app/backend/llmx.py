import os
import google.generativeai as genai
from dotenv import load_dotenv

class LLM:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Set API key for the Generative AI model
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("API key not found. Please set the GEMINI_API_KEY in your environment variables.")
        
        # Configure API with key
        genai.configure(api_key=api_key)
        
        # Model configuration settings
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        # Initialize model and start chat session
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=self.generation_config
        )
        
        # Start a new chat session
        self.chat_session = self.model.start_chat(history=[])

    def get_response(self, query):
        """
        Sends a user query to the chat session and returns the response.
        
        :param query: str - The user's input text.
        :return: str - The model's response text.
        """
        # Send the message to the model's chat session
        response = self.chat_session.send_message(query)
        
        # Return the response text
        return response.text
    
    
