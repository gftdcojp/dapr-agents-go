// Package agent provides a Go SDK for building Dapr Agents - durable, LLM-powered
// autonomous applications with workflow-backed execution.
//
// Dapr Agents extend the Dapr Actor model with:
//   - LLM integration via Dapr Conversation API
//   - Durable execution via Dapr Workflow
//   - Tool execution with automatic state persistence
//   - Multi-agent collaboration patterns
//
// This SDK is designed to work seamlessly with the existing Dapr Actor infrastructure.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/dapr/go-sdk/actor"
	"github.com/dapr/go-sdk/client"
)

// Agent represents a Dapr Agent that can process tasks using LLM and tools.
// It embeds actor capabilities for state management and adds agent-specific features.
type Agent interface {
	// Core actor methods
	actor.Server

	// Agent-specific methods
	Run(ctx context.Context, input *RunInput) (*RunOutput, error)
	GetStatus(ctx context.Context, workflowID string) (*AgentStatus, error)

	// Tool management
	RegisterTool(tool Tool)
	GetTools() []Tool

	// Memory management
	GetMemory(ctx context.Context) (*ConversationMemory, error)
	ClearMemory(ctx context.Context) error
}

// RunInput represents the input for an agent run
type RunInput struct {
	Prompt      string                 `json:"prompt"`
	Context     map[string]interface{} `json:"context,omitempty"`
	MaxSteps    int                    `json:"maxSteps,omitempty"`
	Timeout     time.Duration          `json:"timeout,omitempty"`
	WorkflowID  string                 `json:"workflowId,omitempty"`
	Synchronous bool                   `json:"synchronous,omitempty"`
}

// RunOutput represents the output from an agent run
type RunOutput struct {
	WorkflowID string                 `json:"workflowId"`
	Status     AgentStatusType        `json:"status"`
	Result     string                 `json:"result,omitempty"`
	Error      string                 `json:"error,omitempty"`
	Steps      []StepResult           `json:"steps,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// AgentStatusType represents the status of an agent execution
type AgentStatusType string

const (
	StatusPending    AgentStatusType = "pending"
	StatusRunning    AgentStatusType = "running"
	StatusCompleted  AgentStatusType = "completed"
	StatusFailed     AgentStatusType = "failed"
	StatusCancelled  AgentStatusType = "cancelled"
	StatusWaiting    AgentStatusType = "waiting" // Waiting for external input
)

// AgentStatus provides detailed status information about an agent run
type AgentStatus struct {
	WorkflowID    string                 `json:"workflowId"`
	Status        AgentStatusType        `json:"status"`
	CurrentStep   int                    `json:"currentStep"`
	TotalSteps    int                    `json:"totalSteps"`
	LastUpdate    time.Time              `json:"lastUpdate"`
	Result        string                 `json:"result,omitempty"`
	Error         string                 `json:"error,omitempty"`
	Steps         []StepResult           `json:"steps,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// StepResult represents the result of a single agent step
type StepResult struct {
	StepNumber  int                    `json:"stepNumber"`
	Type        StepType               `json:"type"`
	Input       string                 `json:"input,omitempty"`
	Output      string                 `json:"output,omitempty"`
	ToolName    string                 `json:"toolName,omitempty"`
	ToolInput   map[string]interface{} `json:"toolInput,omitempty"`
	ToolOutput  interface{}            `json:"toolOutput,omitempty"`
	Duration    time.Duration          `json:"duration"`
	StartedAt   time.Time              `json:"startedAt"`
	CompletedAt time.Time              `json:"completedAt"`
	Error       string                 `json:"error,omitempty"`
}

// StepType represents the type of agent step
type StepType string

const (
	StepTypeLLM     StepType = "llm"
	StepTypeTool    StepType = "tool"
	StepTypeMemory  StepType = "memory"
	StepTypeDecide  StepType = "decide"
)

// Tool represents an executable tool that an agent can use
type Tool interface {
	// Name returns the unique name of the tool
	Name() string

	// Description returns a human-readable description for LLM context
	Description() string

	// Schema returns the JSON schema for tool parameters
	Schema() *ToolSchema

	// Execute runs the tool with the given parameters
	Execute(ctx context.Context, params map[string]interface{}) (interface{}, error)
}

// ToolSchema defines the expected parameters for a tool
type ToolSchema struct {
	Type        string                    `json:"type"`
	Properties  map[string]PropertySchema `json:"properties,omitempty"`
	Required    []string                  `json:"required,omitempty"`
	Description string                    `json:"description,omitempty"`
}

// PropertySchema defines a single property in a tool schema
type PropertySchema struct {
	Type        string   `json:"type"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
	Default     any      `json:"default,omitempty"`
}

// ConversationMemory stores conversation history for context
type ConversationMemory struct {
	Messages   []Message `json:"messages"`
	Summary    string    `json:"summary,omitempty"`
	TokenCount int       `json:"tokenCount,omitempty"`
	UpdatedAt  time.Time `json:"updatedAt"`
}

// Message represents a single message in a conversation
type Message struct {
	Role      string                 `json:"role"` // system, user, assistant, tool
	Content   string                 `json:"content"`
	Name      string                 `json:"name,omitempty"` // For tool messages
	ToolCalls []ToolCall             `json:"toolCalls,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
}

// ToolCall represents an LLM's request to execute a tool
type ToolCall struct {
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// AgentConfig configures an agent's behavior
type AgentConfig struct {
	// Name is the agent's unique identifier
	Name string `json:"name"`

	// LLM configuration
	LLMComponent string            `json:"llmComponent"` // Dapr conversation component name
	LLMModel     string            `json:"llmModel,omitempty"`
	LLMOptions   map[string]string `json:"llmOptions,omitempty"`

	// Memory configuration
	MemoryStore     string `json:"memoryStore,omitempty"`     // State store for conversation memory
	MemoryMaxTokens int    `json:"memoryMaxTokens,omitempty"` // Max tokens to keep in memory

	// Execution configuration
	WorkflowStore     string        `json:"workflowStore,omitempty"` // State store for workflow state
	MaxSteps          int           `json:"maxSteps,omitempty"`      // Default max steps per run
	StepTimeout       time.Duration `json:"stepTimeout,omitempty"`   // Timeout per step
	EnableApiLogging  bool          `json:"enableApiLogging,omitempty"`

	// System prompt
	SystemPrompt string `json:"systemPrompt,omitempty"`
}

// DefaultAgentConfig returns sensible defaults for agent configuration
func DefaultAgentConfig(name string) *AgentConfig {
	return &AgentConfig{
		Name:            name,
		LLMComponent:    "llm",
		MemoryStore:     "conversation-statestore",
		WorkflowStore:   "workflow-statestore",
		MaxSteps:        10,
		StepTimeout:     30 * time.Second,
		MemoryMaxTokens: 4000,
	}
}

// BaseAgent provides a base implementation of the Agent interface
// that can be embedded in custom agent implementations.
type BaseAgent struct {
	actor.ServerImplBaseCtx
	config     *AgentConfig
	tools      map[string]Tool
	toolsMu    sync.RWMutex
	daprClient client.Client
}

// NewBaseAgent creates a new BaseAgent with the given configuration
func NewBaseAgent(config *AgentConfig) *BaseAgent {
	if config == nil {
		config = DefaultAgentConfig("default-agent")
	}
	return &BaseAgent{
		config: config,
		tools:  make(map[string]Tool),
	}
}

// Type returns the actor type name
func (a *BaseAgent) Type() string {
	return a.config.Name
}

// RegisterTool registers a tool for use by the agent
func (a *BaseAgent) RegisterTool(tool Tool) {
	a.toolsMu.Lock()
	defer a.toolsMu.Unlock()
	a.tools[tool.Name()] = tool
}

// GetTools returns all registered tools
func (a *BaseAgent) GetTools() []Tool {
	a.toolsMu.RLock()
	defer a.toolsMu.RUnlock()
	tools := make([]Tool, 0, len(a.tools))
	for _, t := range a.tools {
		tools = append(tools, t)
	}
	return tools
}

// GetTool retrieves a tool by name
func (a *BaseAgent) GetTool(name string) (Tool, bool) {
	a.toolsMu.RLock()
	defer a.toolsMu.RUnlock()
	tool, ok := a.tools[name]
	return tool, ok
}

// Run executes the agent with the given input
func (a *BaseAgent) Run(ctx context.Context, input *RunInput) (*RunOutput, error) {
	if input.MaxSteps == 0 {
		input.MaxSteps = a.config.MaxSteps
	}

	// Generate workflow ID if not provided
	workflowID := input.WorkflowID
	if workflowID == "" {
		workflowID = fmt.Sprintf("%s-%d", a.ID(), time.Now().UnixNano())
	}

	// Initialize state
	state := &agentState{
		WorkflowID: workflowID,
		Status:     StatusRunning,
		Steps:      make([]StepResult, 0),
		StartedAt:  time.Now(),
	}

	if err := a.saveState(ctx, state); err != nil {
		return nil, fmt.Errorf("failed to save initial state: %w", err)
	}

	// For synchronous execution, run the agent loop
	if input.Synchronous {
		return a.runSync(ctx, input, state)
	}

	// For async execution, start workflow and return immediately
	return &RunOutput{
		WorkflowID: workflowID,
		Status:     StatusPending,
	}, nil
}

// runSync executes the agent loop synchronously
func (a *BaseAgent) runSync(ctx context.Context, input *RunInput, state *agentState) (*RunOutput, error) {
	// Add timeout if specified
	if input.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, input.Timeout)
		defer cancel()
	}

	// Load or initialize memory
	memory, err := a.GetMemory(ctx)
	if err != nil {
		memory = &ConversationMemory{Messages: make([]Message, 0)}
	}

	// Add system prompt if configured
	if a.config.SystemPrompt != "" && len(memory.Messages) == 0 {
		memory.Messages = append(memory.Messages, Message{
			Role:      "system",
			Content:   a.config.SystemPrompt,
			Timestamp: time.Now(),
		})
	}

	// Add user message
	memory.Messages = append(memory.Messages, Message{
		Role:      "user",
		Content:   input.Prompt,
		Timestamp: time.Now(),
	})

	// Agent loop
	for step := 0; step < input.MaxSteps; step++ {
		select {
		case <-ctx.Done():
			state.Status = StatusCancelled
			state.Error = ctx.Err().Error()
			a.saveState(ctx, state)
			return &RunOutput{
				WorkflowID: state.WorkflowID,
				Status:     StatusCancelled,
				Error:      ctx.Err().Error(),
				Steps:      state.Steps,
			}, nil
		default:
		}

		stepResult, done, err := a.executeStep(ctx, memory, step)
		if err != nil {
			state.Status = StatusFailed
			state.Error = err.Error()
			a.saveState(ctx, state)
			return &RunOutput{
				WorkflowID: state.WorkflowID,
				Status:     StatusFailed,
				Error:      err.Error(),
				Steps:      state.Steps,
			}, nil
		}

		state.Steps = append(state.Steps, *stepResult)
		state.CurrentStep = step + 1
		a.saveState(ctx, state)

		if done {
			state.Status = StatusCompleted
			state.Result = stepResult.Output
			state.CompletedAt = time.Now()
			a.saveState(ctx, state)

			// Save memory
			a.saveMemory(ctx, memory)

			return &RunOutput{
				WorkflowID: state.WorkflowID,
				Status:     StatusCompleted,
				Result:     stepResult.Output,
				Steps:      state.Steps,
			}, nil
		}
	}

	// Max steps reached
	state.Status = StatusCompleted
	state.CompletedAt = time.Now()
	a.saveState(ctx, state)
	a.saveMemory(ctx, memory)

	lastStep := state.Steps[len(state.Steps)-1]
	return &RunOutput{
		WorkflowID: state.WorkflowID,
		Status:     StatusCompleted,
		Result:     lastStep.Output,
		Steps:      state.Steps,
	}, nil
}

// executeStep executes a single agent step
func (a *BaseAgent) executeStep(ctx context.Context, memory *ConversationMemory, stepNum int) (*StepResult, bool, error) {
	stepStart := time.Now()
	result := &StepResult{
		StepNumber: stepNum,
		StartedAt:  stepStart,
	}

	// Call LLM with current context
	llmResponse, err := a.callLLM(ctx, memory)
	if err != nil {
		result.Error = err.Error()
		result.CompletedAt = time.Now()
		result.Duration = result.CompletedAt.Sub(stepStart)
		return result, false, err
	}

	// Check if LLM wants to use tools
	if len(llmResponse.ToolCalls) > 0 {
		result.Type = StepTypeTool

		// Execute each tool call
		for _, toolCall := range llmResponse.ToolCalls {
			result.ToolName = toolCall.Name
			result.ToolInput = toolCall.Arguments

			tool, ok := a.GetTool(toolCall.Name)
			if !ok {
				result.Error = fmt.Sprintf("tool not found: %s", toolCall.Name)
				memory.Messages = append(memory.Messages, Message{
					Role:      "tool",
					Name:      toolCall.Name,
					Content:   result.Error,
					Timestamp: time.Now(),
				})
				continue
			}

			toolOutput, err := tool.Execute(ctx, toolCall.Arguments)
			if err != nil {
				result.Error = err.Error()
				memory.Messages = append(memory.Messages, Message{
					Role:      "tool",
					Name:      toolCall.Name,
					Content:   fmt.Sprintf("Error: %s", err.Error()),
					Timestamp: time.Now(),
				})
				continue
			}

			result.ToolOutput = toolOutput

			// Add tool result to memory
			toolOutputJSON, _ := json.Marshal(toolOutput)
			memory.Messages = append(memory.Messages, Message{
				Role:      "tool",
				Name:      toolCall.Name,
				Content:   string(toolOutputJSON),
				Timestamp: time.Now(),
			})
		}

		result.CompletedAt = time.Now()
		result.Duration = result.CompletedAt.Sub(stepStart)
		return result, false, nil
	}

	// LLM provided a final response
	result.Type = StepTypeLLM
	result.Output = llmResponse.Content
	result.CompletedAt = time.Now()
	result.Duration = result.CompletedAt.Sub(stepStart)

	// Add assistant response to memory
	memory.Messages = append(memory.Messages, Message{
		Role:      "assistant",
		Content:   llmResponse.Content,
		Timestamp: time.Now(),
	})

	// Check if this is a final answer (no tool calls)
	return result, true, nil
}

// LLMResponse represents a response from the LLM
type LLMResponse struct {
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"toolCalls,omitempty"`
}

// callLLM calls the LLM via Dapr Conversation API
func (a *BaseAgent) callLLM(ctx context.Context, memory *ConversationMemory) (*LLMResponse, error) {
	// Get Dapr client
	daprClient, err := a.getDaprClient()
	if err != nil {
		return nil, fmt.Errorf("failed to get Dapr client: %w", err)
	}

	// Build messages for LLM
	messages := make([]map[string]interface{}, 0, len(memory.Messages))
	for _, msg := range memory.Messages {
		m := map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		}
		if msg.Name != "" {
			m["name"] = msg.Name
		}
		messages = append(messages, m)
	}

	// Build tools definition for LLM
	tools := a.buildToolsDefinition()

	// Call Dapr Conversation API
	conversationInput := map[string]interface{}{
		"messages": messages,
	}
	if len(tools) > 0 {
		conversationInput["tools"] = tools
	}

	inputBytes, err := json.Marshal(conversationInput)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal conversation input: %w", err)
	}

	// Invoke conversation component
	resp, err := daprClient.InvokeBinding(ctx, &client.InvokeBindingRequest{
		Name:      a.config.LLMComponent,
		Operation: "conversation",
		Data:      inputBytes,
		Metadata: map[string]string{
			"model": a.config.LLMModel,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("LLM invocation failed: %w", err)
	}

	var llmResponse LLMResponse
	if err := json.Unmarshal(resp.Data, &llmResponse); err != nil {
		return nil, fmt.Errorf("failed to parse LLM response: %w", err)
	}

	return &llmResponse, nil
}

// buildToolsDefinition builds the tools definition for the LLM
func (a *BaseAgent) buildToolsDefinition() []map[string]interface{} {
	a.toolsMu.RLock()
	defer a.toolsMu.RUnlock()

	tools := make([]map[string]interface{}, 0, len(a.tools))
	for _, tool := range a.tools {
		schema := tool.Schema()
		toolDef := map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        tool.Name(),
				"description": tool.Description(),
				"parameters":  schema,
			},
		}
		tools = append(tools, toolDef)
	}
	return tools
}

// getDaprClient returns or creates a Dapr client
func (a *BaseAgent) getDaprClient() (client.Client, error) {
	if a.daprClient != nil {
		return a.daprClient, nil
	}

	c, err := client.NewClient()
	if err != nil {
		return nil, err
	}
	a.daprClient = c
	return c, nil
}

// GetStatus retrieves the status of an agent run
func (a *BaseAgent) GetStatus(ctx context.Context, workflowID string) (*AgentStatus, error) {
	state, err := a.loadState(ctx, workflowID)
	if err != nil {
		return nil, err
	}

	return &AgentStatus{
		WorkflowID:  state.WorkflowID,
		Status:      state.Status,
		CurrentStep: state.CurrentStep,
		TotalSteps:  len(state.Steps),
		LastUpdate:  state.UpdatedAt,
		Result:      state.Result,
		Error:       state.Error,
		Steps:       state.Steps,
	}, nil
}

// GetMemory retrieves the conversation memory
func (a *BaseAgent) GetMemory(ctx context.Context) (*ConversationMemory, error) {
	stateManager := a.GetStateManager()

	var data []byte
	if err := stateManager.Get(ctx, "memory", &data); err != nil {
		return nil, fmt.Errorf("failed to get memory: %w", err)
	}

	var memory ConversationMemory
	if err := json.Unmarshal(data, &memory); err != nil {
		return nil, fmt.Errorf("failed to unmarshal memory: %w", err)
	}

	return &memory, nil
}

// ClearMemory clears the conversation memory
func (a *BaseAgent) ClearMemory(ctx context.Context) error {
	stateManager := a.GetStateManager()
	return stateManager.Remove(ctx, "memory")
}

// saveMemory saves the conversation memory
func (a *BaseAgent) saveMemory(ctx context.Context, memory *ConversationMemory) error {
	memory.UpdatedAt = time.Now()
	data, err := json.Marshal(memory)
	if err != nil {
		return fmt.Errorf("failed to marshal memory: %w", err)
	}

	stateManager := a.GetStateManager()
	return stateManager.Set(ctx, "memory", data)
}

// agentState holds the internal state of an agent run
type agentState struct {
	WorkflowID  string          `json:"workflowId"`
	Status      AgentStatusType `json:"status"`
	CurrentStep int             `json:"currentStep"`
	Steps       []StepResult    `json:"steps"`
	Result      string          `json:"result,omitempty"`
	Error       string          `json:"error,omitempty"`
	StartedAt   time.Time       `json:"startedAt"`
	UpdatedAt   time.Time       `json:"updatedAt"`
	CompletedAt time.Time       `json:"completedAt,omitempty"`
}

// saveState saves the agent execution state
func (a *BaseAgent) saveState(ctx context.Context, state *agentState) error {
	state.UpdatedAt = time.Now()
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("failed to marshal state: %w", err)
	}

	stateManager := a.GetStateManager()
	return stateManager.Set(ctx, fmt.Sprintf("workflow-%s", state.WorkflowID), data)
}

// loadState loads the agent execution state
func (a *BaseAgent) loadState(ctx context.Context, workflowID string) (*agentState, error) {
	stateManager := a.GetStateManager()

	var data []byte
	if err := stateManager.Get(ctx, fmt.Sprintf("workflow-%s", workflowID), &data); err != nil {
		return nil, fmt.Errorf("failed to get state: %w", err)
	}

	var state agentState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("failed to unmarshal state: %w", err)
	}

	return &state, nil
}
