// Package agent provides Python SDK compatible aliases for dapr-agents-go
//
// This file provides type aliases and wrapper functions to ensure
// API compatibility with the Python dapr-agents SDK.
package agent

import (
	"context"
)

// ============================================================================
// Agent Aliases (Python: AgentBase, Agent, DurableAgent)
// ============================================================================

// AgentBase is an alias for the Agent interface (Python SDK compatibility)
type AgentBase = Agent

// DurableAgent is an alias for BaseAgent with workflow support
// In Python SDK, DurableAgent integrates with Dapr Workflow
// In Go SDK, BaseAgent with Actor provides similar durability
type DurableAgent = BaseAgent

// ============================================================================
// Tool Aliases (Python: AgentTool, AgentToolExecutor)
// ============================================================================

// AgentTool is an alias for Tool interface (Python SDK compatibility)
type AgentTool = Tool

// AgentToolExecutor manages tool registration and execution
// Mirrors Python SDK's AgentToolExecutor class
type AgentToolExecutor struct {
	tools map[string]Tool
}

// NewAgentToolExecutor creates a new tool executor
func NewAgentToolExecutor() *AgentToolExecutor {
	return &AgentToolExecutor{
		tools: make(map[string]Tool),
	}
}

// RegisterTool registers a tool with the executor
func (e *AgentToolExecutor) RegisterTool(tool Tool) {
	e.tools[tool.Name()] = tool
}

// GetTool retrieves a tool by name
func (e *AgentToolExecutor) GetTool(name string) (Tool, bool) {
	tool, ok := e.tools[name]
	return tool, ok
}

// GetToolNames returns all registered tool names
func (e *AgentToolExecutor) GetToolNames() []string {
	names := make([]string, 0, len(e.tools))
	for name := range e.tools {
		names = append(names, name)
	}
	return names
}

// GetToolSignatures returns tool schemas for LLM
func (e *AgentToolExecutor) GetToolSignatures() []*ToolSchema {
	schemas := make([]*ToolSchema, 0, len(e.tools))
	for _, tool := range e.tools {
		schemas = append(schemas, tool.Schema())
	}
	return schemas
}

// RunTool executes a tool by name
func (e *AgentToolExecutor) RunTool(ctx context.Context, name string, args map[string]interface{}) (interface{}, error) {
	tool, ok := e.tools[name]
	if !ok {
		return nil, nil
	}
	return tool.Execute(ctx, args)
}

// ============================================================================
// Memory Aliases (Python: MemoryBase, ConversationListMemory, etc.)
// ============================================================================

// Note: MemoryBase is defined in memory.go as the interface for memory implementations
// ConversationMemoryStore is the Go SDK equivalent

// ============================================================================
// LLM Client Aliases (Python: LLMClientBase, ChatClientBase)
// ============================================================================

// LLMClientBase is an alias for LLMClient interface
type LLMClientBase = LLMClient

// ChatClientBase is an alias for LLMClient with chat capabilities
type ChatClientBase = LLMClient

// OpenAIChatClient is an alias for OpenAIClient (Python SDK compatibility)
type OpenAIChatClient = OpenAIClient

// AzureOpenAIChatClient is an alias for AzureOpenAIClient
type AzureOpenAIChatClient = AzureOpenAIClient

// ============================================================================
// MCP Aliases
// ============================================================================

// Note: MCPClientConfig is defined in mcp_client.go

// ============================================================================
// Orchestrator Aliases (Python: OrchestratorBase)
// ============================================================================

// OrchestratorBase is an alias for Orchestrator interface
type OrchestratorBase = Orchestrator

// ============================================================================
// Code Executor Aliases (Python: CodeExecutorBase)
// ============================================================================

// CodeExecutorBase is an alias for CodeExecutor interface
type CodeExecutorBase = CodeExecutor

// ============================================================================
// Document Processing Aliases (Python: ReaderBase, SplitterBase, EmbedderBase)
// ============================================================================

// ReaderBase is an alias for DocumentLoader interface
type ReaderBase = DocumentLoader

// SplitterBase is an alias for TextSplitter interface
type SplitterBase = TextSplitter

// EmbedderBase is an alias for Embedder interface
type EmbedderBase = Embedder

// TextLoader is an alias for TextDocumentLoader (Python SDK compatibility)
type TextLoader = TextDocumentLoader

// ============================================================================
// Observability Aliases (Python: DaprAgentsInstrumentor)
// ============================================================================

// DaprAgentsInstrumentor is an alias for Instrumentor
type DaprAgentsInstrumentor = Instrumentor

// ============================================================================
// Factory Functions (Python SDK style constructors)
// ============================================================================

// NewAgent creates a new agent (Python: Agent())
func NewAgent(config *AgentConfig) *BaseAgent {
	return NewBaseAgent(config)
}

// NewDurableAgent creates a durable agent (Python: DurableAgent())
func NewDurableAgent(config *AgentConfig) *BaseAgent {
	return NewBaseAgent(config)
}

// NewLLMClientFromProvider creates an LLM client by provider name
// Python SDK style factory function
func NewLLMClientFromProvider(provider string, config map[string]interface{}) (LLMClient, error) {
	return NewLLMClient(provider, config)
}

// NewCodeExecutor creates a code executor by type
// Python SDK style factory function
func NewCodeExecutor(executorType string, config interface{}) (CodeExecutor, error) {
	switch executorType {
	case "local":
		if cfg, ok := config.(*LocalCodeExecutorConfig); ok {
			return NewLocalCodeExecutor(cfg)
		}
		return NewLocalCodeExecutor(nil)
	case "docker":
		if cfg, ok := config.(*DockerCodeExecutorConfig); ok {
			return NewDockerCodeExecutor(cfg)
		}
		return NewDockerCodeExecutor(nil)
	default:
		return NewLocalCodeExecutor(nil)
	}
}

// ============================================================================
// Python SDK Method Compatibility
// ============================================================================

// BuildInitialMessages builds initial messages for an agent (Python: build_initial_messages)
func (a *BaseAgent) BuildInitialMessages(systemPrompt string) []Message {
	messages := make([]Message, 0)
	if systemPrompt != "" {
		messages = append(messages, Message{
			Role:    "system",
			Content: systemPrompt,
		})
	}
	return messages
}

// GetChatHistory returns chat history (Python: get_chat_history)
func (a *BaseAgent) GetChatHistory(ctx context.Context) ([]Message, error) {
	memory, err := a.GetMemory(ctx)
	if err != nil {
		return nil, err
	}
	return memory.Messages, nil
}

// ResetMemory resets agent memory (Python: reset_memory)
func (a *BaseAgent) ResetMemory(ctx context.Context) error {
	return a.ClearMemory(ctx)
}

// GetLLMTools returns tools formatted for LLM (Python: get_llm_tools)
func (a *BaseAgent) GetLLMTools() []*ToolSchema {
	tools := a.GetTools()
	schemas := make([]*ToolSchema, len(tools))
	for i, tool := range tools {
		schemas[i] = tool.Schema()
	}
	return schemas
}

// ConstructMessages builds messages without running (Python: construct_messages)
func (a *BaseAgent) ConstructMessages(ctx context.Context, prompt string) ([]Message, error) {
	memory, err := a.GetMemory(ctx)
	if err != nil {
		memory = &ConversationMemory{Messages: make([]Message, 0)}
	}

	messages := make([]Message, len(memory.Messages))
	copy(messages, memory.Messages)

	if a.config.SystemPrompt != "" && len(messages) == 0 {
		messages = append(messages, Message{
			Role:    "system",
			Content: a.config.SystemPrompt,
		})
	}

	messages = append(messages, Message{
		Role:    "user",
		Content: prompt,
	})

	return messages, nil
}
