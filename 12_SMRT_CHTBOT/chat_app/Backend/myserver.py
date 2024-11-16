'''
Open-domain QA Pipeline
'''
import argparse
# from toolbench.inference.Downstream_tasks.rapidapi import pipeline_runner
from myretriever import MyRetriver
import os

if __name__ == "__main__":
    
    tool_root_dir_path = os.path.join('toolbench','data', 'toolenv', 'tools')
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_tsv_path', type=str, default="toolbench/data/retrieval/G1/corpus.tsv", required=False, help='')
    parser.add_argument('--retrieval_model_path', type=str, default="ToolBench/ToolBench_IR_bert_based_uncased", required=False, help='')
    parser.add_argument('--retrieved_api_nums', type=int, default=1, required=False, help='')
    parser.add_argument('--tool_root_dir', type=str, default=tool_root_dir_path, required=False, help='')
    parser.add_argument('--QuerY', type=str, required=True, help='Query to be translated')

    args = parser.parse_args()
    args.api_customization = True

    myretriver = MyRetriver(args)
    api_response_status, result = myretriver.myrun()

    if api_response_status:
        print(f"final result from the API is : \n {result}")
    else:
        print(f"there is API response error is : \n {result}")