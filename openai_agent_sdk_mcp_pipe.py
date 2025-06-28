"""
title: OpenAI Agent SDK Pipe
author: Gaussian
required_open_webui_version: 0.5.0
version: 0.0.1
license: MIT
"""

import inspect
import logging
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Any,
    Literal,
    Mapping,
    Optional,
    TypedDict,
    Union,
)
import html
import asyncio
import os
from pydantic import BaseModel, Field
import json

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from openai.types import Reasoning
from agents import AgentHooks, ItemHelpers, MessageOutputItem, ModelSettings, RunItem, RunItemStreamEvent, RunResultStreaming, TContext, Tool, ToolCallItem, ToolCallOutputItem, set_default_openai_client, function_tool
from agents import Agent, Runner, RunHooks, RunContextWrapper, Usage, FunctionTool, RunContextWrapper
from agents.extensions import handoff_prompt
from agents.run import DEFAULT_MAX_TURNS
from agents.mcp import MCPServerStdio, MCPServerStdioParams
from agents.mcp.util import MCPUtil
from agents import set_tracing_disabled

set_tracing_disabled(True)

class EventEmitterMessageData(TypedDict):
    content: str

class EventEmitterStatusData(TypedDict):
    description: str
    done: Optional[bool]

class EventEmitterNotificationData(TypedDict):
    type: Literal["info","success", "warning", "error"]
    content: str

class EventEmitterStatus(TypedDict):
    type: Literal["status"]
    data: EventEmitterStatusData

class EventEmitterNotification(TypedDict):
    type: Literal["notification"]
    data: EventEmitterNotificationData

class EventEmitterMessage(TypedDict):
    type: Literal["chat:message","chat:message:delta"]
    data: EventEmitterMessageData


class Metadata(TypedDict):
    chat_id: str
    user_id: str
    message_id: str


class EventEmitter:
    def __init__(
        self,
        __event_emitter__: Optional[
            Callable[[Mapping[str, Any]], Awaitable[None]]
        ] = None,
    ):
        self.event_emitter = __event_emitter__

    async def emit(
        self, message: Union[EventEmitterMessage, EventEmitterStatus, EventEmitterNotification]
    ) -> None:
        if self.event_emitter:
            maybe_future = self.event_emitter(message)
            if asyncio.isfuture(maybe_future) or inspect.isawaitable(maybe_future):
                await maybe_future

    async def status(self, description: str, done: Optional[bool] = None) -> None:
        await self.emit(
            EventEmitterStatus(
                type="status",
                data=EventEmitterStatusData(description=description, done=done),
            )
        )

    async def notification(self, content: str, type: Literal["info", "success", "warning", "error"] = "info") -> None:
        await self.emit(
            EventEmitterNotification(
                type="notification",
                data=EventEmitterNotificationData(content=content, type=type),
            )
        )

    async def message(self,content: str, append = False) -> None:
        await self.emit(
            EventEmitterMessage(
                type="chat:message:delta" if append else "chat:message",
                data=EventEmitterMessageData(
                    content=content
                ),
            )
        )

def create_stdio_server(server_name:str, server_config:dict, cwd=None, timeout=30):
    params_kwargs = dict(server_config)
    if cwd is not None:
        params_kwargs["cwd"] = cwd

    server = MCPServerStdio(
        params=MCPServerStdioParams(**params_kwargs),
        name=server_name,
        client_session_timeout_seconds=timeout,
    )
    return server

class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
        OPENAI_BASE_URL: str = Field(
            default="https://api.openai.com/v1", description="OpenAI API base URL"
        )
        TRIAGE_MODEL: str = Field(
            default="gpt-4.1-nano",
            description="Triage Agent Model ID",
        )
        HTTP_PROXY: Optional[str] = Field(
            default=None,
            description="HTTP Proxy URL for OpenAI API requests",
        )
        STRUCTURE_TOOL_CALL: bool = Field(
            default=True,
            description="Whether to use strict JSON schema for tool calls",
        )
        MAX_TURNS: int = Field(
            default=DEFAULT_MAX_TURNS,
            description="The maximum number of turns to run the agent for",
        )
        MCP_ENV_PATH: Optional[str] = Field(
            default=None,
            description="Additional environment variable $PATH to include",
        )
        MCP_CONFIG_FILE: Optional[str] = Field(
            default=None,
            description="Path to the MCP configuration file",
        )
        MCP_CWD: Optional[str] = Field(
            default=None,
            description="MCP working directory for MCP servers",
        )
        MCP_CONNECT_TIMEOUT: int = Field(
            default=45,
            description="Timeout for MCP server client first connection in seconds",
        )


    class EventHooks(RunHooks):
        def __init__(self, event_emitter:EventEmitter):
            self.event_counter = 0
            self.event_emitter = event_emitter

        def _usage_to_str(self, usage: Usage) -> str:
            message = f"Usage: {usage.requests} requests, {usage.input_tokens} input tokens, {usage.output_tokens} output tokens, {usage.total_tokens} total tokens"
            return message
        
        async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
            self.event_counter += 1
            await self.event_emitter.status(f"{agent.name} is Processing...", done=False)

        async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
            self.event_counter += 1
            await self.event_emitter.status(f"{agent.name} Process Complete!",done=True)
        
        async def on_tool_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool) -> None:
            self.event_counter += 1
            logging.info(f"{agent.name} is calling tool {tool.name}...")
            await self.event_emitter.status(f"{agent.name} is Calling Tool {tool.name}...")

        async def on_tool_end(self, context: RunContextWrapper[TContext], agent: Agent[TContext], tool: Tool, result: str) -> None:
            self.event_counter += 1
            logging.info(f"{agent.name} called tool {tool.name} with result: {result}")
            await self.event_emitter.status(f"{agent.name} Call Tool {tool.name} completed ✔")

    def __init__(self):
        self.valves = self.Valves()
        self.type = "manifold"
        self.name = "openai-agent-sdk/"
        self.mcp_servers = []
        self.mcp_configs = {}
        
        logging.info(f"OpenAI Agent SDK Pipe initialized with valves: {self.valves.model_dump()}")

    async def pipe(
        self,
        body: dict,
        __metadata__: Metadata,
        __user__: dict | None = None,
        __task__: str | None = None,
        __tools__: Optional[dict[str, dict[str,Any]]] = None,
        __event_emitter__: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    ):
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        ev = EventEmitter(__event_emitter__)
        ev_hooks = self.EventHooks(ev)

        if self.valves.HTTP_PROXY:
            os.environ["HTTP_PROXY"] = self.valves.HTTP_PROXY
            print(f"Using HTTP Proxy: {self.valves.HTTP_PROXY}")

        custom_client = AsyncOpenAI(base_url=self.valves.OPENAI_BASE_URL, api_key=self.valves.OPENAI_API_KEY)
        set_default_openai_client(custom_client,False)

        # Set up the MCP servers if not already created
        if len(self.mcp_servers)==0:
            if self.valves.MCP_CONFIG_FILE is not None and self.valves.MCP_CONFIG_FILE != "":
                self.append_env_path()

                logging.info("Loading MCP Servers from configuration file...")

                with open(self.valves.MCP_CONFIG_FILE) as f:
                    configs:dict = json.load(f)
                
                if configs.get("mcpServers", {}) == {}:
                    raise ValueError("Please provide the correct MCP server configurations file.")
                
                # Extract each server's dict content from mcpServers
                self.mcp_configs:dict = configs.get("mcpServers", {})

            print(f"MCP Configs: {self.mcp_configs}")

            self.mcp_servers = [
                create_stdio_server(server_name, server_config, cwd=self.valves.MCP_CWD, timeout=self.valves.MCP_CONNECT_TIMEOUT)
                for server_name, server_config in self.mcp_configs.items()
            ]

            try:
                await asyncio.gather(server.connect() for server in self.mcp_servers)
            except Exception as e:
                logging.error(f"Error creating MCP servers: {e}")
                raise e

        logging.info(f"Number of MCP Servers: {len(self.mcp_servers)}")

        body["messages"] = self.transform_message_content(body["messages"])
        
        general_agent_instructions = handoff_prompt.prompt_with_handoff_instructions("You answer general questions and perform basic coding tasks.")
        general_agent = Agent(
            "General Agent",
            model="gpt-4.1-mini",
            instructions=general_agent_instructions,
            handoff_description="Use this agent for general questions and basic coding tasks.",
            mcp_servers=self.mcp_servers, # type: ignore
        )

        reasoning_agent_instructions = handoff_prompt.prompt_with_handoff_instructions("You perform complex reasoning tasks and provide detailed explanations, also perform complex coding tasks.")
        reasoning_agent = Agent(
            "Reasoning Agent",
            model="o4-mini",
            instructions=reasoning_agent_instructions,
            handoff_description="Use this agent for complex reasoning tasks that require detailed explanations or advanced coding.",
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="medium"),
            ),
            mcp_servers=self.mcp_servers, # type: ignore
        )

        triage_agent_name = "Triage agent"
        triage_agent_instructions = "You determine which agent to use based on the user's question, and you should always handoff to the appropriate agent for processing."
        triage_agent_instructions = handoff_prompt.prompt_with_handoff_instructions(triage_agent_instructions)
        triage_agent = Agent(
            triage_agent_name,
            model=self.valves.TRIAGE_MODEL,
            instructions=triage_agent_instructions,
            handoffs=[general_agent,reasoning_agent],
            mcp_servers=self.mcp_servers, # type: ignore
        )

        result = Runner.run_streamed(
            triage_agent, body["messages"],
            max_turns=self.valves.MAX_TURNS,
            hooks=ev_hooks
        )

        await ev.status("Processing...",done=False)

        return self.run(result)

    def append_env_path(self):
        if self.valves.MCP_ENV_PATH:
            env_path = os.environ.get("PATH", "")
            if self.valves.MCP_ENV_PATH not in env_path:
                os.environ["PATH"] = f"{self.valves.MCP_ENV_PATH}:{env_path}"
                logging.info(f"Added {self.valves.MCP_ENV_PATH} to PATH")
            else:
                logging.info(f"{self.valves.MCP_ENV_PATH} already in PATH")

    async def run(self,result:RunResultStreaming):
        try:
            async for event in result.stream_events():
                # We'll ignore the raw responses event deltas
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    yield event.data.delta

                # When the agent updates, print that
                elif event.type == "agent_updated_stream_event":
                    # await ev.status(f"{event.new_agent.name} is Processing...", done=False)
                    continue
                
                # When items are generated, print them
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        logging.info(f"Tool Call: {event.item.raw_item.name}") # type: ignore
                        if event.item.raw_item.type == "function_call":
                            # This is a function call item
                            yield f"**Call Tool {event.item.raw_item.name}**\n"
                        else:
                            # This is a built-in tool call item
                            yield f"**Call Tool {event.item.raw_item.type}**\n"
                    elif event.item.type == "tool_call_output_item":
                        if event.item.raw_item["type"] == "function_call_output":
                            # This is a function call output item
                            pass
                    elif event.item.type == "message_output_item":
                        pass
                    elif event.item.type == "reasoning_item":
                        yield f"\n**Reasoning Completed!**\n"
                        pass
                    else:
                        pass  # Ignore other event types
                    continue
                    
        except Exception as e:
            logging.exception(f"Error during streaming: {e}")
            raise e

    def transform_message_content(self, messages:list):
        """
        Transform the content of messages to match the OpenAI Response API format.
        """
        for msg in messages:
            if isinstance(msg.get("content"), list):
                new_content = []
                for content in msg["content"]:
                    if isinstance(content, dict):
                        type_map = {"text": "input_text", "image_url": "input_image"}
                        if content.get("type") in type_map:
                            content["type"] = type_map[content["type"]]
                            
                            # Handle input_image type specifically
                            if content["type"] == "input_image" and isinstance(content.get("image_url"), dict):
                                url_obj:dict = content.get("image_url") # type: ignore
                                if "url" in url_obj:
                                    content["image_url"] = url_obj["url"]
                        else:
                            raise ValueError(f"Unknown content type: {content.get('type')}")
                        new_content.append(content)
                    else:
                        new_content.append(content)
                msg["content"] = new_content

        return messages

    def filter_messages_role(self, messages:list, role:str = "system"):
        if isinstance(messages, list):
            filtered_messages = [
                msg for msg in messages if msg.get("role") != role
            ]
        else:
            filtered_messages = messages
        return filtered_messages

    def get_user_instruction(self, body):
        try:
            system_instruction = ""
            if (
                "messages" in body
                and isinstance(body["messages"], list)
                and len(body["messages"]) > 0
                and "role" in body["messages"][0]
                and body["messages"][0]["role"] == "system"
                and "content" in body["messages"][0]
            ):
                system_instruction = str(body["messages"][0]["content"])
        except Exception:
            system_instruction = ""
        return system_instruction