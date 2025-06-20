#!/usr/bin/env python3
"""
Azure OpenAI Client with Google Sheets MCP Integration using .env
"""

import os
import json
import asyncio
import logging
import subprocess
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

# Load .env variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SheetsAgent:
    """Agent that uses Azure OpenAI with Google Sheets MCP tools"""

    def __init__(self):
        """Initialize the Azure OpenAI client from .env"""
        
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not all([api_key, endpoint, deployment, api_version]):
            raise ValueError("Missing one or more required Azure OpenAI environment variables.")

        self.model = deployment
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{endpoint}/openai/deployments/{deployment}",
            default_headers={"api-key": api_key}
        )

        # Available MCP tools
        self.available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_sheet",
                    "description": "Create a new Google Sheet",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Title of the new sheet"
                            }
                        },
                        "required": ["title"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_sheets",
                    "description": "List all available Google Sheets",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_sheet",
                    "description": "Read data from a Google Sheet",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sheet_id": {
                                "type": "string",
                                "description": "The ID of the Google Sheet"
                            },
                            "range": {
                                "type": "string",
                                "description": "A1 notation range (e.g., 'A1:Z1000')",
                                "default": "A1:Z1000"
                            }
                        },
                        "required": ["sheet_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_sheet",
                    "description": "Write data to a Google Sheet",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sheet_id": {
                                "type": "string",
                                "description": "The ID of the Google Sheet"
                            },
                            "data": {
                                "type": "string",
                                "description": "JSON data to write to the sheet"
                            },
                            "range": {
                                "type": "string",
                                "description": "A1 notation range (e.g., 'Sheet1!A1')",
                                "default": "A1"
                            }
                        },
                        "required": ["sheet_id", "data"]
                    }
                }
            }
        ]

    async def call_mcp_tool(self, tool_name: str, **kwargs) -> str:
        """Call an MCP tool via CLI"""
        try:
            if tool_name == "create_sheet":
                cmd = ["python", "google_sheets.py", "--cli", "--create-sheet", kwargs["title"]]
            elif tool_name == "list_sheets":
                cmd = ["python", "google_sheets.py", "--cli", "--list-sheets"]
            elif tool_name == "read_sheet":
                range_param = kwargs.get("range", "A1:Z1000")
                cmd = ["python", "google_sheets.py", "--cli", "--read-sheet", kwargs["sheet_id"], "--range", range_param]
            elif tool_name == "write_sheet":
                return "Write operations need to be implemented in CLI"
            else:
                return f"Unknown tool: {tool_name}"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"

        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return f"Error calling tool: {str(e)}"

    async def process_message(self, user_message: str, conversation_history: List[Dict] = None) -> str:
        """Process a user message and handle tool calls"""
        if conversation_history is None:
            conversation_history = []

        messages = conversation_history + [{"role": "user", "content": user_message}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.available_tools,
                tool_choice="auto",
                temperature=0.7
            )

            response_message = response.choices[0].message

            if response_message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in response_message.tool_calls
                    ]
                })

                for tc in response_message.tool_calls:
                    function_name = tc.function.name
                    function_args = json.loads(tc.function.arguments)
                    print(f"🔧 Calling tool: {function_name} with args: {function_args}")
                    tool_result = await self.call_mcp_tool(function_name, **function_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result
                    })

                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7
                )

                return final_response.choices[0].message.content

            else:
                return response_message.content

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Sorry, I encountered an error: {str(e)}"


async def main():
    """Main CLI interface"""
    print("🚀 Google Sheets Assistant with Azure OpenAI")
    print("=" * 50)

    agent = SheetsAgent()
    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("🤔 Thinking...")
            response = await agent.process_message(user_input, conversation_history)
            print(f"\n🤖 Assistant: {response}\n")

            conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ])

            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
