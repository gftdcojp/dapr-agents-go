package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

// WorkflowState represents the state of a workflow execution
type WorkflowState struct {
	InstanceID   string                 `json:"instanceId"`
	Status       WorkflowStatus         `json:"status"`
	Input        interface{}            `json:"input,omitempty"`
	Output       interface{}            `json:"output,omitempty"`
	Error        string                 `json:"error,omitempty"`
	CurrentStep  string                 `json:"currentStep,omitempty"`
	Steps        []WorkflowStepEntry    `json:"steps,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt    time.Time              `json:"createdAt"`
	UpdatedAt    time.Time              `json:"updatedAt"`
	CompletedAt  *time.Time             `json:"completedAt,omitempty"`
}

// WorkflowStatus represents the status of a workflow
type WorkflowStatus string

const (
	WorkflowStatusPending   WorkflowStatus = "pending"
	WorkflowStatusRunning   WorkflowStatus = "running"
	WorkflowStatusCompleted WorkflowStatus = "completed"
	WorkflowStatusFailed    WorkflowStatus = "failed"
	WorkflowStatusCancelled WorkflowStatus = "cancelled"
	WorkflowStatusSuspended WorkflowStatus = "suspended"
)

// WorkflowStepEntry represents a step in the workflow history
type WorkflowStepEntry struct {
	Name        string                 `json:"name"`
	Input       interface{}            `json:"input,omitempty"`
	Output      interface{}            `json:"output,omitempty"`
	Error       string                 `json:"error,omitempty"`
	StartedAt   time.Time              `json:"startedAt"`
	CompletedAt *time.Time             `json:"completedAt,omitempty"`
	Duration    time.Duration          `json:"duration,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// WorkflowDefinition defines a workflow
type WorkflowDefinition struct {
	Name        string
	Description string
	Steps       []WorkflowStepDef
	OnError     func(ctx context.Context, state *WorkflowState, err error) error
}

// WorkflowStepDef defines a workflow step
type WorkflowStepDef struct {
	Name     string
	Handler  func(ctx context.Context, input interface{}) (interface{}, error)
	Retry    *RetryPolicy
	Timeout  time.Duration
	OnError  func(ctx context.Context, err error) error
}

// RetryPolicy defines retry behavior
type RetryPolicy struct {
	MaxAttempts     int
	InitialInterval time.Duration
	MaxInterval     time.Duration
	Multiplier      float64
}

// DefaultRetryPolicy returns a default retry policy
func DefaultRetryPolicy() *RetryPolicy {
	return &RetryPolicy{
		MaxAttempts:     3,
		InitialInterval: time.Second,
		MaxInterval:     30 * time.Second,
		Multiplier:      2.0,
	}
}

// WorkflowEngine executes workflows
type WorkflowEngine struct {
	definitions map[string]*WorkflowDefinition
	states      map[string]*WorkflowState
	mu          sync.RWMutex
	stateStore  StateStore
}

// StateStore interface for persisting workflow state
type StateStore interface {
	Save(ctx context.Context, key string, value interface{}) error
	Load(ctx context.Context, key string, value interface{}) error
	Delete(ctx context.Context, key string) error
}

// InMemoryStateStore implements StateStore using in-memory storage
type InMemoryStateStore struct {
	data map[string][]byte
	mu   sync.RWMutex
}

// NewInMemoryStateStore creates a new in-memory state store
func NewInMemoryStateStore() *InMemoryStateStore {
	return &InMemoryStateStore{
		data: make(map[string][]byte),
	}
}

func (s *InMemoryStateStore) Save(ctx context.Context, key string, value interface{}) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[key] = data
	return nil
}

func (s *InMemoryStateStore) Load(ctx context.Context, key string, value interface{}) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	data, ok := s.data[key]
	if !ok {
		return fmt.Errorf("key not found: %s", key)
	}
	return json.Unmarshal(data, value)
}

func (s *InMemoryStateStore) Delete(ctx context.Context, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.data, key)
	return nil
}

// NewWorkflowEngine creates a new workflow engine
func NewWorkflowEngine(stateStore StateStore) *WorkflowEngine {
	if stateStore == nil {
		stateStore = NewInMemoryStateStore()
	}
	return &WorkflowEngine{
		definitions: make(map[string]*WorkflowDefinition),
		states:      make(map[string]*WorkflowState),
		stateStore:  stateStore,
	}
}

// RegisterWorkflow registers a workflow definition
func (e *WorkflowEngine) RegisterWorkflow(def *WorkflowDefinition) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.definitions[def.Name] = def
}

// StartWorkflow starts a new workflow instance
func (e *WorkflowEngine) StartWorkflow(ctx context.Context, workflowName string, instanceID string, input interface{}) error {
	e.mu.RLock()
	def, ok := e.definitions[workflowName]
	e.mu.RUnlock()

	if !ok {
		return fmt.Errorf("workflow not found: %s", workflowName)
	}

	state := &WorkflowState{
		InstanceID: instanceID,
		Status:     WorkflowStatusPending,
		Input:      input,
		Steps:      make([]WorkflowStepEntry, 0),
		Metadata:   make(map[string]interface{}),
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}

	e.mu.Lock()
	e.states[instanceID] = state
	e.mu.Unlock()

	// Save initial state
	if err := e.stateStore.Save(ctx, "workflow:"+instanceID, state); err != nil {
		return fmt.Errorf("failed to save workflow state: %w", err)
	}

	// Run workflow in goroutine
	go e.executeWorkflow(ctx, def, state)

	return nil
}

func (e *WorkflowEngine) executeWorkflow(ctx context.Context, def *WorkflowDefinition, state *WorkflowState) {
	state.Status = WorkflowStatusRunning
	state.UpdatedAt = time.Now()
	e.saveState(ctx, state)

	var lastOutput interface{} = state.Input

	for _, stepDef := range def.Steps {
		state.CurrentStep = stepDef.Name

		stepEntry := WorkflowStepEntry{
			Name:      stepDef.Name,
			Input:     lastOutput,
			StartedAt: time.Now(),
		}

		// Execute step with retries
		output, err := e.executeStepWithRetry(ctx, stepDef, lastOutput)

		now := time.Now()
		stepEntry.CompletedAt = &now
		stepEntry.Duration = now.Sub(stepEntry.StartedAt)

		if err != nil {
			stepEntry.Error = err.Error()
			state.Steps = append(state.Steps, stepEntry)
			state.Status = WorkflowStatusFailed
			state.Error = fmt.Sprintf("step %s failed: %v", stepDef.Name, err)
			state.UpdatedAt = time.Now()

			// Call error handler if defined
			if def.OnError != nil {
				def.OnError(ctx, state, err)
			}

			e.saveState(ctx, state)
			return
		}

		stepEntry.Output = output
		state.Steps = append(state.Steps, stepEntry)
		state.UpdatedAt = time.Now()
		e.saveState(ctx, state)

		lastOutput = output
	}

	// Workflow completed successfully
	now := time.Now()
	state.Status = WorkflowStatusCompleted
	state.Output = lastOutput
	state.CompletedAt = &now
	state.UpdatedAt = now
	state.CurrentStep = ""
	e.saveState(ctx, state)
}

func (e *WorkflowEngine) executeStepWithRetry(ctx context.Context, stepDef WorkflowStepDef, input interface{}) (interface{}, error) {
	policy := stepDef.Retry
	if policy == nil {
		policy = &RetryPolicy{MaxAttempts: 1}
	}

	var lastErr error
	interval := policy.InitialInterval

	for attempt := 0; attempt < policy.MaxAttempts; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(interval):
			}

			// Increase interval for next attempt
			interval = time.Duration(float64(interval) * policy.Multiplier)
			if interval > policy.MaxInterval {
				interval = policy.MaxInterval
			}
		}

		// Create step context with timeout if specified
		stepCtx := ctx
		var cancel context.CancelFunc
		if stepDef.Timeout > 0 {
			stepCtx, cancel = context.WithTimeout(ctx, stepDef.Timeout)
		}

		output, err := stepDef.Handler(stepCtx, input)

		if cancel != nil {
			cancel()
		}

		if err == nil {
			return output, nil
		}

		lastErr = err

		// Call step error handler if defined
		if stepDef.OnError != nil {
			stepDef.OnError(ctx, err)
		}
	}

	return nil, lastErr
}

func (e *WorkflowEngine) saveState(ctx context.Context, state *WorkflowState) {
	e.mu.Lock()
	e.states[state.InstanceID] = state
	e.mu.Unlock()

	e.stateStore.Save(ctx, "workflow:"+state.InstanceID, state)
}

// GetWorkflowState returns the state of a workflow instance
func (e *WorkflowEngine) GetWorkflowState(ctx context.Context, instanceID string) (*WorkflowState, error) {
	e.mu.RLock()
	state, ok := e.states[instanceID]
	e.mu.RUnlock()

	if ok {
		return state, nil
	}

	// Try loading from state store
	var loaded WorkflowState
	if err := e.stateStore.Load(ctx, "workflow:"+instanceID, &loaded); err != nil {
		return nil, fmt.Errorf("workflow instance not found: %s", instanceID)
	}

	return &loaded, nil
}

// CancelWorkflow cancels a running workflow
func (e *WorkflowEngine) CancelWorkflow(ctx context.Context, instanceID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	state, ok := e.states[instanceID]
	if !ok {
		return fmt.Errorf("workflow instance not found: %s", instanceID)
	}

	if state.Status != WorkflowStatusRunning && state.Status != WorkflowStatusPending {
		return fmt.Errorf("workflow is not running: %s", state.Status)
	}

	state.Status = WorkflowStatusCancelled
	state.UpdatedAt = time.Now()
	return e.stateStore.Save(ctx, "workflow:"+instanceID, state)
}

// WorkflowBuilder provides a fluent API for building workflows
type WorkflowBuilder struct {
	name        string
	description string
	steps       []WorkflowStepDef
	onError     func(ctx context.Context, state *WorkflowState, err error) error
}

// NewWorkflowBuilder creates a new workflow builder
func NewWorkflowBuilder(name string) *WorkflowBuilder {
	return &WorkflowBuilder{
		name:  name,
		steps: make([]WorkflowStepDef, 0),
	}
}

// Description sets the workflow description
func (b *WorkflowBuilder) Description(desc string) *WorkflowBuilder {
	b.description = desc
	return b
}

// AddStep adds a step to the workflow
func (b *WorkflowBuilder) AddStep(name string, handler func(ctx context.Context, input interface{}) (interface{}, error)) *WorkflowBuilder {
	b.steps = append(b.steps, WorkflowStepDef{
		Name:    name,
		Handler: handler,
	})
	return b
}

// AddStepWithRetry adds a step with retry policy
func (b *WorkflowBuilder) AddStepWithRetry(name string, handler func(ctx context.Context, input interface{}) (interface{}, error), policy *RetryPolicy) *WorkflowBuilder {
	b.steps = append(b.steps, WorkflowStepDef{
		Name:    name,
		Handler: handler,
		Retry:   policy,
	})
	return b
}

// AddStepWithTimeout adds a step with timeout
func (b *WorkflowBuilder) AddStepWithTimeout(name string, handler func(ctx context.Context, input interface{}) (interface{}, error), timeout time.Duration) *WorkflowBuilder {
	b.steps = append(b.steps, WorkflowStepDef{
		Name:    name,
		Handler: handler,
		Timeout: timeout,
	})
	return b
}

// OnError sets the error handler for the workflow
func (b *WorkflowBuilder) OnError(handler func(ctx context.Context, state *WorkflowState, err error) error) *WorkflowBuilder {
	b.onError = handler
	return b
}

// Build creates the workflow definition
func (b *WorkflowBuilder) Build() *WorkflowDefinition {
	return &WorkflowDefinition{
		Name:        b.name,
		Description: b.description,
		Steps:       b.steps,
		OnError:     b.onError,
	}
}

// AgentWorkflow wraps an agent as a workflow
type AgentWorkflow struct {
	agent  *BaseAgent
	engine *WorkflowEngine
}

// NewAgentWorkflow creates a new agent workflow
func NewAgentWorkflow(agent *BaseAgent, engine *WorkflowEngine) *AgentWorkflow {
	return &AgentWorkflow{
		agent:  agent,
		engine: engine,
	}
}

// CreateAgentWorkflow creates a workflow definition for an agent
func (aw *AgentWorkflow) CreateAgentWorkflow() *WorkflowDefinition {
	return NewWorkflowBuilder(aw.agent.Type()).
		Description(fmt.Sprintf("Workflow for agent: %s", aw.agent.Type())).
		AddStepWithRetry("agent_run", func(ctx context.Context, input interface{}) (interface{}, error) {
			// Convert input to RunInput
			var runInput RunInput
			switch v := input.(type) {
			case *RunInput:
				runInput = *v
			case RunInput:
				runInput = v
			case string:
				runInput = RunInput{Prompt: v, Synchronous: true}
			case map[string]interface{}:
				if prompt, ok := v["prompt"].(string); ok {
					runInput.Prompt = prompt
				}
				runInput.Synchronous = true
			default:
				return nil, fmt.Errorf("unsupported input type: %T", input)
			}

			return aw.agent.Run(ctx, &runInput)
		}, DefaultRetryPolicy()).
		Build()
}

// RegisterWithEngine registers the agent workflow with the engine
func (aw *AgentWorkflow) RegisterWithEngine() {
	aw.engine.RegisterWorkflow(aw.CreateAgentWorkflow())
}

// Run starts a new workflow instance for the agent
func (aw *AgentWorkflow) Run(ctx context.Context, instanceID string, input *RunInput) error {
	return aw.engine.StartWorkflow(ctx, aw.agent.Type(), instanceID, input)
}

// GetStatus returns the status of a workflow instance
func (aw *AgentWorkflow) GetStatus(ctx context.Context, instanceID string) (*WorkflowState, error) {
	return aw.engine.GetWorkflowState(ctx, instanceID)
}

// MessageRouter routes messages to handlers based on content
type MessageRouter struct {
	routes   []MessageRoute
	fallback func(ctx context.Context, msg Message) error
}

// MessageRoute defines a route for messages
type MessageRoute struct {
	Pattern string
	Handler func(ctx context.Context, msg Message) error
}

// NewMessageRouter creates a new message router
func NewMessageRouter() *MessageRouter {
	return &MessageRouter{
		routes: make([]MessageRoute, 0),
	}
}

// Route adds a route to the router
func (r *MessageRouter) Route(pattern string, handler func(ctx context.Context, msg Message) error) *MessageRouter {
	r.routes = append(r.routes, MessageRoute{
		Pattern: pattern,
		Handler: handler,
	})
	return r
}

// Fallback sets the fallback handler
func (r *MessageRouter) Fallback(handler func(ctx context.Context, msg Message) error) *MessageRouter {
	r.fallback = handler
	return r
}

// Handle routes a message to the appropriate handler
func (r *MessageRouter) Handle(ctx context.Context, msg Message) error {
	for _, route := range r.routes {
		// Simple pattern matching (could be enhanced with regex)
		if matchPattern(route.Pattern, msg.Content) {
			return route.Handler(ctx, msg)
		}
	}

	if r.fallback != nil {
		return r.fallback(ctx, msg)
	}

	return fmt.Errorf("no handler found for message")
}

func matchPattern(pattern, content string) bool {
	// Simple contains matching - could be enhanced with regex or more sophisticated matching
	if pattern == "*" {
		return true
	}
	return len(pattern) > 0 && len(content) >= len(pattern) && strings.Contains(content, pattern)
}

// HTTPRouter routes HTTP requests to handlers
type HTTPRouter struct {
	routes map[string]map[string]func(ctx context.Context, req interface{}) (interface{}, error)
}

// NewHTTPRouter creates a new HTTP router
func NewHTTPRouter() *HTTPRouter {
	return &HTTPRouter{
		routes: make(map[string]map[string]func(ctx context.Context, req interface{}) (interface{}, error)),
	}
}

// Handle registers a handler for a path and method
func (r *HTTPRouter) Handle(method, path string, handler func(ctx context.Context, req interface{}) (interface{}, error)) *HTTPRouter {
	if r.routes[path] == nil {
		r.routes[path] = make(map[string]func(ctx context.Context, req interface{}) (interface{}, error))
	}
	r.routes[path][method] = handler
	return r
}

// GET registers a GET handler
func (r *HTTPRouter) GET(path string, handler func(ctx context.Context, req interface{}) (interface{}, error)) *HTTPRouter {
	return r.Handle("GET", path, handler)
}

// POST registers a POST handler
func (r *HTTPRouter) POST(path string, handler func(ctx context.Context, req interface{}) (interface{}, error)) *HTTPRouter {
	return r.Handle("POST", path, handler)
}

// Route finds the handler for a request
func (r *HTTPRouter) Route(method, path string) (func(ctx context.Context, req interface{}) (interface{}, error), bool) {
	if methods, ok := r.routes[path]; ok {
		if handler, ok := methods[method]; ok {
			return handler, true
		}
	}
	return nil, false
}
