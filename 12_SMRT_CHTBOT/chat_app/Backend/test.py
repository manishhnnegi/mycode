
import urllib.parse
import json
import re
from Prompts.react_prompt2 import TOOL_ASSISTANCE_PROMPT_TEMPLATE,FORMAT_INSTRUCTIONS_USER_FUNCTION
import os
import google.generativeai as genai



class Google_LLM:

    def __init__(self,api_key):
        
        self.api_key = api_key
        self.model = "gemini-1.5-flash"
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        # Configure API with key
        genai.configure(api_key=self.api_key)
         # Model configuration settings
        # Initialize model and start chat session
        self.model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=self.generation_config
        )
        # Start a new chat session
        self.chat_session = self.model.start_chat(history=[])

    
    def run_agent(self, tool_response, question):

        system_prompt = TOOL_ASSISTANCE_PROMPT_TEMPLATE.format(tool_response = tool_response)
        #self.user_prompt = FORMAT_INSTRUCTIONS_USER_FUNCTION.format(question= question)
        user_prompt = FORMAT_INSTRUCTIONS_USER_FUNCTION.format(question= question)

        #self.hist = [{'role': 'model','parts':[system_prompt]}]
        #self.messages = [{'role': 'system','content':[system_prompt]},{"role": "user", "content": [user_prompt2]}]
        messages = system_prompt + user_prompt

        chat_completion_response = self.chat_session.send_message(messages)
        if chat_completion_response:
            flage = True
        else:
            flage = False
            
        if flage:
            thought, final_answer = self.pattern_match(chat_completion_response)
        else:
            thought,final_answer = None,  f"some error related to {chat_completion_response}"
        return thought, final_answer

    def pattern_match(self, input_string):
        # Input string
        #input_string = chat_completion.choices[0].message.content
        # Define regex pattern to match "Thought" and "Final Answer" separately
        pattern = r'Thought: (.*?)\nFinal Answer: (.*)'
        try:
            # Use re.search() with the re.DOTALL flag to find the first occurrence of the pattern
            match = re.search(pattern, input_string, re.DOTALL)

            # Check if the pattern is found
            if match:
                # Extract the content of "Thought" and "Final Answer"
                thought = match.group(1)
                final_answer = match.group(2)
            
            else:
                thought = "error"
                final_answer = "error"
                
            return thought, final_answer
        
        except:
            thought = "error"
            final_answer = "error"
            return thought, final_answer
        


if __name__ == "__main__":
    import os
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = GEMINI_KEY
    tool_response = "'cloud_pct': 0, 'temp': 40, 'feels_like': 37, 'humidity': 11, 'min_temp': 40,"
    question = "what is weather in Delhi today?"
    question = "what is temperature in Delhi today?"
    cls =  Google_LLM(GEMINI_KEY)
    thought, final_answer = cls.run_agent(tool_response, question )
    print(f"Final Answer: {final_answer}")