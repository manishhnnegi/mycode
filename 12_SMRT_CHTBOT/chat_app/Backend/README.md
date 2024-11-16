<div align= "center">
    <h1> 🛠️Smart Assistant Agent with Real-Time External Tool Access🤖</h1>
</div>



<!-- <div align="center">
<img src="https://cdn.discordapp.com/attachments/941582479117127680/1111543600879259749/20230526075532.png" width="350px"> -->
</div>

🔨This project aims to create a **Smart Assistant Agent with Real-Time External Tool Access** which can assist the human as per their instruction. It can also access external APIs to answer queries related to real-time information. It can also show **tool-use** capability.

Till now . . . 

🔨below external APIs(Tools) has been integrated.

1. **Ecommerce API**
2. **Wikipidea Search API**
3. **Translation API**
4. **Weather API**

🔨Memory has been integrated using langchain's Generative Agent Memory.Based on pthe paper [["Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/abs/2304.03442)] 



### Install
Clone this repository and navigate to the ToolBench folder.
```bash
https://github.com/manishhnnegi/Tool-Retrieval-System.git
cd Tool-Retrieval-System
```
Install Package (python>=3.9)
```bash
pip install -r requirements.txt
```

## Inference With Our RapidAPI Server
- In gitbash terminal:
```bash
source activate <env_name>
```
- Then run the following command to run the experiments:
```bash
bash run.sh
```

## Inference With Our Streamlit Server
- To inference with ToolLLaMA, run the following commands:
```bash
python retrival_agent_streamlit.py
```
- Then run the following command to run the experiments:
```bash
python tool_agent.py
```
- Then run the streamlit frontend server:
```bash
streamlit run app.py
```
This server will be available on `http://localhost:8501/`. To start a request, call `http://localhost:8501/stream` with a GET or POST request containing a JSON object with the following fields:
```json
{

    "query": "what is weather in Delhi today?",
    "top_k": 1,  
}
```
## Inference With Our React App
- To inference with ToolLLaMA, run the following commands:
```bash
python retrival_agent_streamlit.py
```
- Then run the following command to run the experiments:
```bash
cd Frontend
```
- Then run the streamlit frontend server:
```bash
npx create-react-app myui
cd myui
npm install axios
npm start
```

## Directory Structure
- Folder structure:
```bash
Tool-Retrieval-System/
├── LLM/
│   ├── __init__.py
│   └── openai_server.py
└── Prompts/
    ├── __init__.py
    └── react_prompt.py
```

- Now the file structure under `data/toolenv/` should be:
```
├── /tools/
│  ├── /eCommerce/
│  │  ├── search.json
│  │  ├── /search/
│  │  │  └── api.py
│  │  └── ...
│  ├── ...
│  ├── /Translation/
│  │  ├── translate.json
│  │  ├── /translate/
│  │  │  └── api.py
└─
```

### Block Duagram:
✨Here is an overview of the dataset construction, training, and evaluation.

<br>
<div align="center">

<img src="Images\blockdig2.png" width="800px">

</div>
<br>

### Demo UI in React

I shown **A demo of using ToolLLaMA**

<br>
<div align="center">

<img src="Images\amazon.png" width="800px">

</div>
<br>

**Generated Response**

<br>
<div align="center">

<img src="Images\amazon2.png" width="800px">

</div>
<br>

<!-- We also provide **A demo of using ToolLLaMA**

<div align="center">

https://github.com/OpenBMB/ToolBench/assets/25274507/f1151d85-747b-4fac-92ff-6c790d8d9a31

</div> -->
