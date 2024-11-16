#giving you the best api document and api end point to answer your query 


from toolbench.inference.LLM.retriever import ToolRetriever
import os
from tqdm import tqdm
import json
from LLM.gorilla_server import Gorilla_LLM
from toolbench.utils import fun_name_extractor

class MyRetriver:
    def __init__(self,args,add_retrieval=True):
        self.args = args

    def get_myargs(self):
        return self.args

    def get_retriever(self):
        return ToolRetriever(corpus_tsv_path=self.args.corpus_tsv_path, model_path=self.args.retrieval_model_path)
    
    def retrieve_rapidapi_tools(self, query, top_k, jsons_path):
        retriever = self.get_retriever()
        retrieved_tools = retriever.retrieving(query, top_k=top_k)
        query_json = {"api_list":[]}
        for tool_dict in retrieved_tools:
            if len(query_json["api_list"]) == top_k:
                break
            category = tool_dict["category"]
            tool_name = tool_dict["tool_name"]
            api_name = tool_dict["api_name"]
            if os.path.exists(jsons_path):
                if os.path.exists(os.path.join(jsons_path, category)):
                    if os.path.exists(os.path.join(jsons_path, category, api_name+".json")):
                        query_json["api_list"].append({
                            "category_name": category,
                            "tool_name": tool_name,
                            "api_name": api_name
                        })
        return query_json,retrieved_tools

    def myrun(self): 

        try:      
            queryQ = {'query': self.args.QuerY,
            'query_id': 822190}
            task_list = [('DFS_woFilter_w2','chatgpt_function',queryQ['query_id'],queryQ, self.args, None)]
            new_task_list = []
            for task in task_list:
                out_dir_path = task[-2]
                query_id = task[2]
            task_list = new_task_list
            #retriever = self.get_retriever()
            for k, task in enumerate(task_list):
                        #print(f"process[{pipeline_runner.process_id}] doing task {k}/{len(task_list)}: real_task_id_{task[2]}")
                        pass
            data_dict = task[3]
            tool_des = None
            input_description = data_dict["query"]
            functions = []
            api_name_reflect = {}
            query_json,retrieved_tools   = self.retrieve_rapidapi_tools( input_description, self.args.retrieved_api_nums, self.args.tool_root_dir )
            
            # print("-------------------------------------",retrieved_tools)
            # print("-------------------------------------",query_json)
            query_json["api_list"][0]
            file_path_list = []
            for api in query_json["api_list"]:
                #print(api)
                file_path = os.path.join(api['category_name'], f"{api['api_name']}"+".json")
                #print(file_path)
                file_path_list.append(file_path)

            file_path_list = []
            for api in query_json["api_list"]:
                #print(api)
                file_path = os.path.join(api['category_name'], f"{api['api_name']}"+".json")
                #print(file_path)
                file_path_list.append(file_path)

            function_documentation = []
            for file_path  in file_path_list:
                pth = os.path.join(self.args.tool_root_dir,file_path)
                with open(pth, "r") as file:
                    data = file.read()
                    #print(data)
                    function_documentation.append(data)
            #print(function_documentation)
            #print(queryQ['query'])

            api_path_list = []
            for api in query_json["api_list"]:
                #print(api)
                file_path = os.path.join(self.args.tool_root_dir,api['category_name'], f"{api['api_name']}","api")
                #print(file_path)
                api_path_list.append(file_path.replace('\\','.'))

            gorilla_instance = Gorilla_LLM()
            api_with_get_params = gorilla_instance.get_gorilla_response(prompt=queryQ['query'], functions=function_documentation)
            print("----------------------------------------------",api_with_get_params)
            try:
                if api_with_get_params.startswith("gorilla_llm."):
                    if len(api_with_get_params.split("."))==3:
                        api_with_get_params =api_with_get_params.split(".")[2]
                    else:
                        api_with_get_params =api_with_get_params.split(".")[1]

            except:
                pass


            #return get_params
            api_name = fun_name_extractor(api_with_get_params)
            api_import_string =  f"from {api_path_list[0]} import {api_name}"
            exec(api_import_string)
            #api_response = eval(api_with_get_params)
            #return api_response['status'], api_response['data']['translatedText']

            try:
                api_response = eval(api_with_get_params)
                return True, api_response
            except Exception as e:
                return False, e
            
        except Exception as e:

            return False, e







    


