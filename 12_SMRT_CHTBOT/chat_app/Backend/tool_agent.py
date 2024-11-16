from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from myretriever import MyRetriver
import requests
import argparse
import os
# Define models
class Query(BaseModel):
    query: str
    num_tools: int

class Model:
    def __init__(self):
        self.args = self.get_args()
        


    def get_args(self):
        tool_root_dir_path = os.path.join('toolbench','data', 'toolenv', 'tools')
        parser = argparse.ArgumentParser()
        parser.add_argument('--corpus_tsv_path', type=str, default="toolbench/data/retrieval/G1/corpus.tsv", required=False, help='')
        parser.add_argument('--retrieval_model_path', type=str, default="ToolBench/ToolBench_IR_bert_based_uncased", required=False, help='')
        #parser.add_argument('--retrieved_api_nums', type=int, default=1, required=False, help='')
        parser.add_argument('--tool_root_dir', type=str, default=tool_root_dir_path, required=False, help='')
        #parser.add_argument('--QuerY', type=str, required=True, help='Query to be translated')
        args = parser.parse_args()
        args.api_customization = True

        args = parser.parse_args()
        return args

    def run_pipeline(self,user_input, top_k):
        self.args.QuerY = user_input
        self.args.retrieved_api_nums = top_k

        myretriver = MyRetriver(self.args)
        api_response_status, result = myretriver.myrun()
        if api_response_status:
            print(f"final result from the API is : \n {result}")
        else:
            print(f"there is API response error is : \n {result}")
        return api_response_status, result




model = Model()

# Initialize FastAPI instances
app_tool = FastAPI()

# Endpoint in Tool Server to process queries
@app_tool.post("/process-query/")
async def process_query(query: Query):

    print('in tool_--------------------------------------',query)

    user_query = query.query
    num_tools = query.num_tools

    print(user_query)
    print(num_tools)

    api_response_status,result = model.run_pipeline(user_query, num_tools)

    if not api_response_status:
        result = "sorry there is no reference for this query"

    # Here you can add your logic to process the query
    # For demonstration purposes, simply echo back the query
    return {"response": f"Received query: {result}"}
    

# Run the servers
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app_tool, host="0.0.0.0", port=8001)