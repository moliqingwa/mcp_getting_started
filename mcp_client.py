import json
import asyncio
import os
import httpx
from typing import Optional
from contextlib import AsyncExitStack

from openai import OpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm_helper import LLMHelper
from function_calling import process_function_call

load_dotenv()


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], 
        #                      base_url=os.environ['OPENAI_BASE_URL'],
        #      )
        self.llm = LLMHelper()

    async def connect_to_server(self):
        server_params = StdioServerParameters(
            command='uv',
            args=['run', 'web_search.py'],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write))

        await self.session.initialize()
        
    async def process_query(self, query: str) -> str:
        # 获取所有 mcp 服务器 工具列表信息
        tools_response = await self.session.list_tools()
        
        # 生成 function tool_call 的描述信息
        available_tools = []
        for tool in tools_response.tools:
            # 打印工具信息以便调试
            print(json.dumps(tool.__dict__, indent=2, ensure_ascii=False))
            
            # 默认使用 custom 类型
            tool_format = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
            available_tools.append(tool_format)

        completion_params = {
            "provider": "one_api",
            "llm_name": os.getenv("OPENAI_MODEL"),
            "temperature": 0.7,
            "max_new_tokens": 2048,
            "top_p": 0.9
        }
        tool_calls = await process_function_call(self.llm, query, available_tools, completion_params)
        
        # 处理返回的内容
        if tool_calls:
            messages = []
            for tool_call in tool_calls:
                # 执行工具
                result = await self.session.call_tool(tool_call['name'], tool_call['arguments'])
                print(f"\n\n[Calling tool {tool_call['name']} with args {tool_call['arguments']}]\n\n")

                # 将 deepseek 返回的调用哪个工具数据和工具执行完成后的数据都存入messages中
                # messages.append(content.message.model_dump())
                messages.append({
                    "role": "tool",
                    "content": result.content[0].text,
                    "tool_call_id": 'xxx', #tool_call.id,
                })

            # TODO 将上面的结果再返回给 deepseek 用于生产最终的结果
            response = self.llm.chat.completions.create(
                model=os.getenv("OPENAI_MODEL"),
                messages=messages,
            )
            return response.choices[0].message.content

        return "\nNo known function calls needed for this request."

    async def chat_loop(self):
        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    

    asyncio.run(main())
