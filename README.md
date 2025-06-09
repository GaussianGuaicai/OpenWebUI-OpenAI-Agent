# OpenWebUI-OpenAI-Agent

## Project Overview

OpenWebUI-OpenAI-Agent is a multi-agent pipeline implementation based on the [OpenAI Agent SDK](https://openai.github.io/openai-agents-python/). It is designed to provide flexible multi-agent conversation and tool-calling capabilities for OpenWebUI. The project supports multiple models and agent collaboration, suitable for general Q&A, complex reasoning, and code generation scenarios.

## Features
- **Multi-agent collaboration**: Built-in General Agent, Reasoning Agent, and Main Agent for automatic task routing.
- **Tool calling**: Supports custom tool (FunctionTool) registration and asynchronous execution.
- **Streaming output**: Real-time event streaming for better user experience.
- **Extensible**: Easily integrate custom models, tools, and pipeline logic.

## Main Files
- `openai_agent_sdk_pipe.py`: Main pipeline implementation, including agent definitions, tool registration, and event stream handling.

## Agents
- **General Agent**: Handles general questions and basic coding tasks.
- **Reasoning Agent**: Handles complex reasoning and advanced coding tasks.
- **Main Agent**: Automatically routes user requests to the appropriate agent.

## License
MIT License.

---

This project is built with the [OpenAI Agent SDK](https://openai.github.io/openai-agents-python/). For SDK usage and advanced features, please refer to the official documentation.
