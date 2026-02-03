// Package agent provides a Go SDK for building Dapr Agents - durable, LLM-powered
// autonomous applications with workflow-backed execution.
//
// Dapr Agents extend the Dapr Actor model with:
//   - LLM integration via Dapr Conversation API
//   - Durable execution via Dapr Workflow
//   - Tool execution with automatic state persistence
//   - Multi-agent collaboration patterns
//   - MCP (Model Context Protocol) server for agent exposure
//
// # Quick Start
//
// Create an agent with tools:
//
//	config := &AgentConfig{
//		Name:         "MyAgent",
//		LLMComponent: "llm",
//		SystemPrompt: "You are a helpful assistant.",
//	}
//
//	agent := NewBaseAgent(config)
//	agent.RegisterTool(myTool)
//
//	// Run the agent
//	output, err := agent.Run(ctx, &RunInput{
//		Prompt:      "Hello!",
//		Synchronous: true,
//	})
//
// # Server Setup
//
// Run agents as a gRPC/HTTP service:
//
//	server := NewServer(nil)
//	server.RegisterAgent(agentFactory)
//	server.Start()
//
// # MCP Server
//
// Expose agents via Model Context Protocol:
//
//	mcp := NewMCPServer(&MCPServerConfig{Port: 8081})
//	mcp.RegisterAgent("MyAgent", agentFactory)
//	mcp.Start()
//
// # Multi-Agent Patterns
//
// Chain multiple agents:
//
//	chain, _ := NewChainPattern(agent1Ref, agent2Ref, agent3Ref)
//	output, _ := chain.Execute(ctx, input)
//
// Run agents in parallel:
//
//	parallel, _ := NewParallelPattern(agentRefs, aggregator)
//	output, _ := parallel.Execute(ctx, input)
package agent
