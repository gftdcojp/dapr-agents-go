# Dapr Agent SDK for Go

A Go SDK for building Dapr Agents - durable, LLM-powered autonomous applications with workflow-backed execution and MCP (Model Context Protocol) support.

## Features

- **LLM Integration**: Connect to any LLM via Dapr Conversation API (OpenAI, Anthropic, etc.)
- **Durable Execution**: Workflow-backed execution with automatic state persistence
- **Tool System**: Extensible tool framework with built-in Dapr integrations
- **Multi-Agent Patterns**: Chain, Parallel, Router, Supervisor, and Collaborative patterns
- **MCP Server**: Built-in Model Context Protocol server for agent exposure
- **Actor Foundation**: Built on Dapr Virtual Actors for reliable state management

## Installation

```bash
go get github.com/gftdcojp/dapr-agents-go
```

## Quick Start

```go
package main

import (
    "context"
    "log"
    "time"

    agent "github.com/gftdcojp/dapr-agents-go"
)

func main() {
    // Create agent configuration
    config := &agent.AgentConfig{
        Name:         "MyAgent",
        LLMComponent: "llm",
        LLMModel:     "gpt-4-turbo",
        SystemPrompt: "You are a helpful assistant.",
    }

    // Create and configure the agent
    myAgent := agent.NewBaseAgent(config)

    // Register tools
    myAgent.RegisterTool(agent.NewToolBuilder("greet").
        Description("Greet a user by name").
        AddStringParam("name", "The name to greet", true).
        BuildFunc(func(ctx context.Context, params map[string]interface{}) (interface{}, error) {
            name := params["name"].(string)
            return map[string]string{"message": "Hello, " + name + "!"}, nil
        }))

    // Run the agent server
    if err := agent.RunAgent(config, myAgent.GetTools()...); err != nil {
        log.Fatal(err)
    }
}
```

## Agent Configuration

```go
type AgentConfig struct {
    Name            string        // Agent name (actor type)
    LLMComponent    string        // Dapr conversation component
    LLMModel        string        // LLM model name
    MemoryStore     string        // State store for memory
    WorkflowStore   string        // State store for workflow state
    MaxSteps        int           // Max steps per run
    StepTimeout     time.Duration // Timeout per step
    MemoryMaxTokens int           // Max tokens in memory
    SystemPrompt    string        // System prompt for LLM
}
```

## Tools

### Built-in Tool Types

1. **FuncTool** - Wrap any Go function as a tool
2. **HTTPTool** - Call HTTP endpoints
3. **DaprServiceTool** - Invoke Dapr services
4. **DaprActorTool** - Invoke Dapr actors
5. **DaprStateTool** - Read/write Dapr state
6. **DaprPubSubTool** - Publish to Dapr pub/sub

### Creating Tools

```go
// Using the builder pattern
tool := agent.NewToolBuilder("search").
    Description("Search the web").
    AddStringParam("query", "Search query", true).
    AddNumberParam("limit", "Max results", false).
    BuildFunc(searchHandler)

// Or implement the Tool interface directly
type MyTool struct{}
func (t *MyTool) Name() string { return "my_tool" }
func (t *MyTool) Description() string { return "My custom tool" }
func (t *MyTool) Schema() *agent.ToolSchema { return schema }
func (t *MyTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
    // Implementation
}
```

## Multi-Agent Patterns

### Chain Pattern

Execute agents in sequence, passing output to the next:

```go
chain, _ := agent.NewChainPattern(
    agent.AgentRef{Type: "Researcher", ID: "1"},
    agent.AgentRef{Type: "Writer", ID: "1"},
    agent.AgentRef{Type: "Editor", ID: "1"},
)

output, _ := chain.Execute(ctx, &agent.MultiAgentInput{
    Prompt: "Write an article about AI agents",
})
```

### Parallel Pattern

Execute agents concurrently and aggregate results:

```go
parallel, _ := agent.NewParallelPattern(
    []agent.AgentRef{
        {Type: "NewsAgent", ID: "1"},
        {Type: "WeatherAgent", ID: "1"},
        {Type: "StockAgent", ID: "1"},
    },
    aggregator, // Custom aggregation function
)
```

### Router Pattern

Route requests to different agents based on criteria:

```go
router, _ := agent.NewRouterPattern(
    agents,
    func(input *agent.MultiAgentInput) agent.AgentRef {
        // Routing logic
        if strings.Contains(input.Prompt, "weather") {
            return agents["weather"]
        }
        return agents["general"]
    },
)
```

### Supervisor Pattern

A supervisor agent delegates to worker agents:

```go
supervisor, _ := agent.NewSupervisorPattern(
    agent.AgentRef{Type: "Supervisor", ID: "1"},
    agent.AgentRef{Type: "Worker1", ID: "1"},
    agent.AgentRef{Type: "Worker2", ID: "1"},
)
```

## MCP Server

Expose your agents via Model Context Protocol:

```go
mcpServer := agent.NewMCPServer(&agent.MCPServerConfig{
    Port:        8081,
    Name:        "my-agent-mcp",
    Version:     "1.0.0",
    Description: "My Agent MCP Server",
    AuthType:    "apikey",
    APIKey:      os.Getenv("MCP_API_KEY"),
})

mcpServer.RegisterAgent("MyAgent", agentFactory)
mcpServer.RegisterTool(myTool)

go mcpServer.Start()
```

### MCP Endpoints

- `GET /mcp/v1/info` - Server information
- `GET /mcp/v1/tools` - List available tools
- `POST /mcp/v1/tools/{name}` - Execute a tool
- `GET /mcp/v1/prompts` - List available prompts
- `GET /mcp/v1/resources` - List resources
- `GET /mcp/v1/agents` - List agents
- `POST /mcp/v1/agents/{name}` - Run an agent
- `POST /mcp/v1/sse` - Server-sent events for streaming

## Deployment with GFTD

Create a `gftd.json` configuration:

```json
{
  "$schema": "https://gftd.ai/schemas/gftd.json",
  "name": "my-agent",
  "type": "agent",
  "dapr": {
    "enabled": true,
    "appId": "my-agent",
    "appProtocol": "grpc",
    "appPort": 50051,
    "agents": {
      "types": ["MyAgent"],
      "llmComponent": "llm",
      "llmModel": "gpt-4-turbo",
      "mcp": {
        "enabled": true,
        "port": 8081
      }
    }
  }
}
```

Deploy with:

```bash
gftd deploy --dapr
```

## Dapr Components

Required Dapr components for agents:

### LLM Component (Conversation API)

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: llm
spec:
  type: conversation.openai
  metadata:
  - name: key
    secretKeyRef:
      name: openai-secret
      key: api-key
  - name: model
    value: "gpt-4-turbo"
```

### State Store for Memory

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: conversation-statestore
spec:
  type: state.redis
  metadata:
  - name: redisHost
    value: redis:6379
  - name: actorStateStore
    value: "true"
```

### Workflow State Store

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: workflow-statestore
spec:
  type: state.redis
  metadata:
  - name: redisHost
    value: redis:6379
  - name: actorStateStore
    value: "true"
```

## API Reference

See [API Documentation](https://pkg.go.dev/github.com/gftdcojp/dapr-agents-go) for complete reference.

## Examples

- [Weather Agent](./examples/weather-agent/) - Simple agent with weather tools
- [Multi-Agent Chat](./examples/multi-agent-chat/) - Collaborative agents
- [MCP Integration](./examples/mcp-integration/) - Full MCP server setup

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

This project is a Go implementation inspired by [dapr/dapr-agents](https://github.com/dapr/dapr-agents).

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.
