import streamlit as st
import requests

def process_data(query, num_tools,from_openai):
    # Process your data here, you can replace this with your own logic
    data = {
        "query": query,
        "num_tools": num_tools,
        "from_openai" : bool(from_openai)
    }
    return data

def main():
    st.title("Smart Assistant Agent with Real-Time External Tool Access!")
    
    query = st.text_input("Enter your query:")
    num_tools = st.number_input("Enter number of tools:", min_value=1, max_value=10, value=1, step=1)
    from_openai = st.text_input("DISABLE MEMORY: True/False")
    agent_server_url = "http://localhost:8000/send-query/"

    if st.button("Process"):
        # Process data and send to backend
        data = process_data(query, num_tools, from_openai)
        
        response = requests.post(agent_server_url, json=data)
        print( response)
        print(response.json())
        print( type(response))
        print(type(response.json()))

        import  json
        my_dict = json.loads(response.json())

        #response = {"ff":"123"}
        # Display results
        st.write("tool server response:")
        st.write(my_dict['tool_response'])

        st.write("Agent Response:")
        st.write(my_dict['final_answer'])


if __name__ == "__main__":
    main()