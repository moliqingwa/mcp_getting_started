import httpx
import mcp
from mcp.server import FastMCP
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

llm = OpenAI(api_key=os.environ['OPENAI_API_KEY'],
             base_url=os.environ['OPENAI_BASE_URL'])

# # 初始化 FastMCP 服务器
app = FastMCP('Demo')


# Resources
# Resources are how you expose data to LLMs. They're similar to GET endpoints in a REST API - they provide data but shouldn't perform significant computation or have side effects:
# Add a dynamic greeting resource
@app.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

@app.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "App configuration here"

# Tools
# Tools let LLMs take actions through your server. Unlike resources, tools are expected to perform computation and have side effects:
# Add an addition tool
@app.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Prompts
# Prompts are reusable templates that help LLMs interact with your server effectively:
@app.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"


if __name__ == "__main__":
    app.run(transport='stdio')
