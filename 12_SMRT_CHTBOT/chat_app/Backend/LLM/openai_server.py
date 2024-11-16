# Import Chat completion template and set-up variables
#!pip install openai==0.28.1 &> /dev/null
import openai
import urllib.parse
import json
import re
from Prompts.react_prompt import TOOL_ASSISTANCE_PROMPT_TEMPLATE,FORMAT_INSTRUCTIONS_USER_FUNCTION
import openai





class Openai_LLM:
    
    def __init__(self,api_key, tool_response, question):
        system_prompt = TOOL_ASSISTANCE_PROMPT_TEMPLATE.format(tool_response = tool_response)
        user_prompt = FORMAT_INSTRUCTIONS_USER_FUNCTION.format(question= question)
        self.api_key = api_key
        self.messages = [{'role': 'system','content':system_prompt},{"role": "user", "content": user_prompt}]
        self.model = "gpt-3.5-turbo-0125"

    # Query Gorilla server
    def get_openai_response(self):
        openai.api_key = self.api_key 
        
        try:
            chat_completion = openai.ChatCompletion.create(model=self.model, messages=self.messages)
            return True, chat_completion.choices[0].message.content, chat_completion
        except Exception as e:
            return False, e, None
        

    def run_agent(self):
        flage, chat_completion_response, _ = self.get_openai_response()
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
    #from openai_server import Openai_LLM
    import os
    OPENAI_KEY = os.getenv("MY_KEY")
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
    tool_response = "'cloud_pct': 0, 'temp': 40, 'feels_like': 37, 'humidity': 11, 'min_temp': 40,"
    question = "what is weather in Delhi today?"

    cls =  Openai_LLM(OPENAI_KEY,tool_response, question )
    thought, final_answer = cls.run_agent()
    print(f"Final Answer: {final_answer}")


