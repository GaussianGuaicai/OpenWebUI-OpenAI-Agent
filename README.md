# OpenWebUI-OpenAI-Agent

## Overview

**OpenWebUI-OpenAI-Agent** is a multi-agent pipeline implementation based on the [OpenAI Agent SDK](https://openai.github.io/openai-agents-python/). It enables flexible multi-agent conversations, tool-calling, and advanced reasoning for [OpenWebUI](https://github.com/open-webui/open-webui). The project supports multiple models, agent collaboration, and both standard OpenAI SDK and MCP (Multi-Component Pipeline) backends.

## Features

- **Multi-agent collaboration**: Built-in General Agent, Reasoning Agent, Research Agent, and Triage Agent for automatic task routing.
- **Tool calling**: Register and execute custom tools (FunctionTool) asynchronously. Includes built-in support for image generation.
- **Streaming output**: Real-time event streaming for responsive user experience.
- **Extensible**: Easily integrate custom models, tools, and pipeline logic.
- **MCP support**: Optional integration with MCP servers for advanced tool execution and environment isolation.
- **Observability**: Optional tracing support for debugging and monitoring agent behavior.

## Main Files

- `openai_agent_sdk_pipe.py`: Main pipeline using the OpenAI Agent SDK, including agent definitions, tool registration, and event stream handling.
- `openai_agent_sdk_mcp_pipe.py`: An advanced pipeline with MCP (Multi-Component Pipeline) server support for robust tool execution and environment management.

## Agents

- **General Agent**: Handles general questions and basic coding tasks.
- **Reasoning Agent**: Handles complex reasoning and advanced coding tasks.
- **Research Agent**: Performs in-depth research and answers questions requiring current information.
- **Triage Agent**: Automatically routes user requests to the appropriate agent.

## Usage

1. **Configure API Keys and Models**  
   Set your OpenAI API key and model preferences in the pipeline valves or environment variables.

2. **Register Tools**  
   Define and register custom tools (functions) to extend agent capabilities.

3. **Run the Pipeline**  
   Integrate the pipeline with OpenWebUI or call it directly in your Python code.

4. **(Optional) MCP Configuration**  
   For advanced tool execution, configure MCP servers in `openai_agent_sdk_mcp_pipe.py` and provide a valid MCP config file.

## Configuration

- Edit the `Valves` class in each pipeline file to set API keys, model IDs, proxy settings, MCP options, and other parameters.
- For MCP, provide a JSON config file with your MCP server definitions.

## License

MIT License.

---

This project is built with the [OpenAI Agent SDK](https://openai.github.io/openai-agents-python/).  
For SDK usage and advanced features, please refer to the official documentation.
