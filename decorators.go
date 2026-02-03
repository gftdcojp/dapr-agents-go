// Package agent provides Python SDK compatible decorator-style builders
//
// In Python, decorators like @tool, @workflow_entry are used.
// In Go, we provide builder functions and registration patterns.
package agent

import (
	"context"
	"reflect"
	"runtime"
	"strings"
	"time"
)

// ============================================================================
// Tool Decorator (Python: @tool)
// ============================================================================

// ToolFunc is a function that can be converted to a tool
type ToolFunc func(ctx context.Context, params map[string]interface{}) (interface{}, error)

// ToolDecorator creates a tool from a function (Python: @tool decorator equivalent)
// Usage: agent.ToolDecorator(myFunc, "description")
func ToolDecorator(fn ToolFunc, description string) Tool {
	// Get function name using reflection
	name := runtime.FuncForPC(reflect.ValueOf(fn).Pointer()).Name()
	// Extract just the function name
	parts := strings.Split(name, ".")
	name = parts[len(parts)-1]
	// Remove any suffix
	name = strings.TrimSuffix(name, "-fm")

	return NewToolBuilder(name).
		Description(description).
		BuildFunc(fn)
}

// ToolWithName creates a tool with explicit name
func ToolWithName(name, description string, fn ToolFunc) Tool {
	return NewToolBuilder(name).
		Description(description).
		BuildFunc(fn)
}

// ToolWithSchema creates a tool with full schema
func ToolWithSchema(name, description string, schema *ToolSchema, fn ToolFunc) Tool {
	return &FuncTool{
		name:        name,
		description: description,
		schema:      schema,
		fn:          fn,
	}
}

// ============================================================================
// Workflow Entry Decorator (Python: @workflow_entry)
// ============================================================================

// WorkflowEntryFunc is a workflow entry point function
type WorkflowEntryFunc func(ctx context.Context, input interface{}) (interface{}, error)

// WorkflowEntry marks a function as a workflow entry point
// Returns a WorkflowStepDef that can be used with WorkflowDefinition
func WorkflowEntry(name string, fn WorkflowEntryFunc) *WorkflowStepDef {
	return &WorkflowStepDef{
		Name:    name,
		Handler: fn,
	}
}

// WorkflowEntryBuilder builds workflow entry configurations
type WorkflowEntryBuilder struct {
	name        string
	fn          WorkflowEntryFunc
	retryPolicy *RetryPolicy
	timeout     time.Duration
}

// NewWorkflowEntry creates a new workflow entry builder
func NewWorkflowEntry(name string) *WorkflowEntryBuilder {
	return &WorkflowEntryBuilder{name: name}
}

// Handler sets the workflow handler function
func (b *WorkflowEntryBuilder) Handler(fn WorkflowEntryFunc) *WorkflowEntryBuilder {
	b.fn = fn
	return b
}

// WithRetry sets the retry policy
func (b *WorkflowEntryBuilder) WithRetry(policy *RetryPolicy) *WorkflowEntryBuilder {
	b.retryPolicy = policy
	return b
}

// WithTimeout sets the timeout
func (b *WorkflowEntryBuilder) WithTimeout(timeout time.Duration) *WorkflowEntryBuilder {
	b.timeout = timeout
	return b
}

// Build creates the workflow step definition
func (b *WorkflowEntryBuilder) Build() *WorkflowStepDef {
	step := &WorkflowStepDef{
		Name:    b.name,
		Handler: b.fn,
		Timeout: b.timeout,
	}
	if b.retryPolicy != nil {
		step.Retry = b.retryPolicy
	}
	return step
}

// ============================================================================
// Message Router Decorator (Python: @message_router)
// ============================================================================

// MessageHandler is a function that handles messages
type MessageHandler func(ctx context.Context, message interface{}) error

// MessageRouterEntry represents a message route entry
type MessageRouterEntry struct {
	Pattern string
	Handler MessageHandler
}

// MessageRouteBuilder builds message routes
type MessageRouteBuilder struct {
	routes []MessageRouterEntry
}

// NewMessageRouteBuilder creates a new message route builder
func NewMessageRouteBuilder() *MessageRouteBuilder {
	return &MessageRouteBuilder{
		routes: make([]MessageRouterEntry, 0),
	}
}

// Route adds a route pattern and handler
// Python: @message_router("pattern")
func (b *MessageRouteBuilder) Route(pattern string, handler MessageHandler) *MessageRouteBuilder {
	b.routes = append(b.routes, MessageRouterEntry{
		Pattern: pattern,
		Handler: handler,
	})
	return b
}

// Build creates the MessageRouter
func (b *MessageRouteBuilder) Build() *MessageRouter {
	router := NewMessageRouter()
	for _, route := range b.routes {
		handler := route.Handler
		router.Route(route.Pattern, func(ctx context.Context, msg Message) error {
			return handler(ctx, msg)
		})
	}
	return router
}

// ============================================================================
// HTTP Router Decorator (Python: @http_router)
// ============================================================================

// HTTPHandlerFunc is a function that handles HTTP requests
type HTTPHandlerFunc func(ctx context.Context, request interface{}) (interface{}, error)

// HTTPRouteEntry represents an HTTP route entry
type HTTPRouteEntry struct {
	Method  string
	Path    string
	Handler HTTPHandlerFunc
}

// HTTPRouteBuilder builds HTTP routes
type HTTPRouteBuilder struct {
	routes []HTTPRouteEntry
}

// NewHTTPRouteBuilder creates a new HTTP route builder
func NewHTTPRouteBuilder() *HTTPRouteBuilder {
	return &HTTPRouteBuilder{
		routes: make([]HTTPRouteEntry, 0),
	}
}

// Get adds a GET route
func (b *HTTPRouteBuilder) Get(path string, handler HTTPHandlerFunc) *HTTPRouteBuilder {
	return b.Route("GET", path, handler)
}

// Post adds a POST route
func (b *HTTPRouteBuilder) Post(path string, handler HTTPHandlerFunc) *HTTPRouteBuilder {
	return b.Route("POST", path, handler)
}

// Put adds a PUT route
func (b *HTTPRouteBuilder) Put(path string, handler HTTPHandlerFunc) *HTTPRouteBuilder {
	return b.Route("PUT", path, handler)
}

// Delete adds a DELETE route
func (b *HTTPRouteBuilder) Delete(path string, handler HTTPHandlerFunc) *HTTPRouteBuilder {
	return b.Route("DELETE", path, handler)
}

// Route adds a route with explicit method
func (b *HTTPRouteBuilder) Route(method, path string, handler HTTPHandlerFunc) *HTTPRouteBuilder {
	b.routes = append(b.routes, HTTPRouteEntry{
		Method:  method,
		Path:    path,
		Handler: handler,
	})
	return b
}

// Build creates the HTTPRouter
func (b *HTTPRouteBuilder) Build() *HTTPRouter {
	router := NewHTTPRouter()
	for _, route := range b.routes {
		router.Handle(route.Method, route.Path, route.Handler)
	}
	return router
}

// ============================================================================
// Agent Registration Helpers
// ============================================================================

// AgentBuilder builds agents with fluent API
type AgentBuilder struct {
	config       *AgentConfig
	tools        []Tool
	systemPrompt string
}

// NewAgentBuilder creates a new agent builder
func NewAgentBuilder(name string) *AgentBuilder {
	return &AgentBuilder{
		config: DefaultAgentConfig(name),
		tools:  make([]Tool, 0),
	}
}

// WithLLM sets the LLM component and model
func (b *AgentBuilder) WithLLM(component, model string) *AgentBuilder {
	b.config.LLMComponent = component
	b.config.LLMModel = model
	return b
}

// WithMemoryStore sets the memory store
func (b *AgentBuilder) WithMemoryStore(store string) *AgentBuilder {
	b.config.MemoryStore = store
	return b
}

// WithSystemPrompt sets the system prompt
func (b *AgentBuilder) WithSystemPrompt(prompt string) *AgentBuilder {
	b.config.SystemPrompt = prompt
	return b
}

// WithMaxSteps sets the maximum steps
func (b *AgentBuilder) WithMaxSteps(steps int) *AgentBuilder {
	b.config.MaxSteps = steps
	return b
}

// WithTool adds a tool
func (b *AgentBuilder) WithTool(tool Tool) *AgentBuilder {
	b.tools = append(b.tools, tool)
	return b
}

// WithTools adds multiple tools
func (b *AgentBuilder) WithTools(tools ...Tool) *AgentBuilder {
	b.tools = append(b.tools, tools...)
	return b
}

// Build creates the agent
func (b *AgentBuilder) Build() *BaseAgent {
	agent := NewBaseAgent(b.config)
	for _, tool := range b.tools {
		agent.RegisterTool(tool)
	}
	return agent
}

// BuildFactory returns a factory function for the agent
func (b *AgentBuilder) BuildFactory() func() Agent {
	config := b.config
	tools := b.tools
	return func() Agent {
		agent := NewBaseAgent(config)
		for _, tool := range tools {
			agent.RegisterTool(tool)
		}
		return agent
	}
}

// ============================================================================
// Convenience Registration Functions
// ============================================================================

// RegisterTools registers multiple tools with an agent
func RegisterTools(agent *BaseAgent, tools ...Tool) {
	for _, tool := range tools {
		agent.RegisterTool(tool)
	}
}

// RegisterToolFuncs registers multiple functions as tools
func RegisterToolFuncs(agent *BaseAgent, funcs map[string]ToolFunc) {
	for name, fn := range funcs {
		tool := NewToolBuilder(name).BuildFunc(fn)
		agent.RegisterTool(tool)
	}
}
