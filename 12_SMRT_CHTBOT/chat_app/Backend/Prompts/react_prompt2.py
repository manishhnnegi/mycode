TOOL_ASSISTANCE_PROMPT_TEMPLATE = """You have given the below reference document which has information to answer the question, use it to analyse the answer:
{tool_response}
By taking reference of it, generate the final answer to the original input question.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Final Answer: the final answer to the original input question

Begin!"""


# Question:{question}
# Final Answer:"""


FORMAT_INSTRUCTIONS_USER_FUNCTION = """
{question}
Final Answer:"""