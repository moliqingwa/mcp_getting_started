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
app = FastMCP('web-search')

@app.tool()
async def web_search(query: str) -> str:
    """
    搜索互联网内容

    Args:
        query: 要搜索内容

    Returns:
        搜索结果的总结
    """
    response = llm.chat.completions.create(
        model=os.environ['OPENAI_MODEL'],
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    app.run(transport='stdio')
