import os
import asyncio
from openai import AsyncOpenAI
from langchain.agents import initialize_agent
from langchain_openai import OpenAI,ChatOpenAI
from langchain.agents import load_tools
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import os

OPENAI_KEY = os.getenv("MY_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

client = AsyncOpenAI()


async def main() -> None:
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    )
    #print(chat_completion)

asyncio.run(main())

response = asyncio.run(main())
print(response)