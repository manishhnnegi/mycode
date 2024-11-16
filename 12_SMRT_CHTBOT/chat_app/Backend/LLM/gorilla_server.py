# Import Chat completion template and set-up variables
#!pip install openai==0.28.1 &> /dev/null
import openai
import urllib.parse
import json



class Gorilla_LLM:
    # Report issues
    def raise_issue(self,e, model, prompt):
        issue_title = urllib.parse.quote("[bug] Hosted Gorilla: <Issue>")
        issue_body = urllib.parse.quote(f"Exception: {e}\nFailed model: {model}, for prompt: {prompt}")
        issue_url = f"https://github.com/ShishirPatil/gorilla/issues/new?assignees=&labels=hosted-gorilla&projects=&template=hosted-gorilla-.md&title={issue_title}&body={issue_body}"
        print(f"An exception has occurred: {e} \nPlease raise an issue here: {issue_url}")

    # Query Gorilla server
    def get_gorilla_response(self, prompt="Call me an Uber ride type \"Plus\" in Berkeley at zipcode 94704 in 10 minutes", model="gorilla-openfunctions-v0", functions=[]):
        openai.api_key = "EMPTY" # Hosted for free with ❤️ from UC Berkeley
        openai.api_base = "http://luigi.millennium.berkeley.edu:8000/v1"
        try:
            completion = openai.ChatCompletion.create(
            model="gorilla-openfunctions-v1",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            functions=functions,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(e, model, prompt)

# queryQ['query']

# get_gorilla_response(prompt=queryQ['query'], functions=function_documentation)

