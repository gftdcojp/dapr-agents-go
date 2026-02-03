package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/dapr/go-sdk/client"
)

// Orchestrator defines the interface for multi-agent orchestration
type Orchestrator interface {
	// SelectNextAgent selects the next agent to execute
	SelectNextAgent(ctx context.Context, state *OrchestrationState) (AgentRef, error)

	// Execute runs the orchestration
	Execute(ctx context.Context, input *MultiAgentInput) (*MultiAgentOutput, error)
}

// OrchestrationState holds the current state of orchestration
type OrchestrationState struct {
	Input         *MultiAgentInput  `json:"input"`
	CurrentStep   int               `json:"currentStep"`
	MaxSteps      int               `json:"maxSteps"`
	AgentHistory  []AgentResult     `json:"agentHistory"`
	Context       map[string]interface{} `json:"context"`
	LastOutput    interface{}       `json:"lastOutput"`
	IsComplete    bool              `json:"isComplete"`
}

// RandomOrchestrator selects agents randomly
type RandomOrchestrator struct {
	agents     []AgentRef
	maxSteps   int
	daprClient client.Client
	rng        *rand.Rand
}

// NewRandomOrchestrator creates a random orchestrator
func NewRandomOrchestrator(agents []AgentRef, maxSteps int) (*RandomOrchestrator, error) {
	c, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	if maxSteps <= 0 {
		maxSteps = 10
	}

	return &RandomOrchestrator{
		agents:     agents,
		maxSteps:   maxSteps,
		daprClient: c,
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

func (o *RandomOrchestrator) SelectNextAgent(ctx context.Context, state *OrchestrationState) (AgentRef, error) {
	if len(o.agents) == 0 {
		return AgentRef{}, fmt.Errorf("no agents available")
	}
	idx := o.rng.Intn(len(o.agents))
	return o.agents[idx], nil
}

func (o *RandomOrchestrator) Execute(ctx context.Context, input *MultiAgentInput) (*MultiAgentOutput, error) {
	start := time.Now()
	state := &OrchestrationState{
		Input:        input,
		MaxSteps:     o.maxSteps,
		AgentHistory: make([]AgentResult, 0),
		Context:      make(map[string]interface{}),
		LastOutput:   input.Prompt,
	}

	for state.CurrentStep < state.MaxSteps && !state.IsComplete {
		agentRef, err := o.SelectNextAgent(ctx, state)
		if err != nil {
			return nil, err
		}

		result, err := o.invokeAgent(ctx, agentRef, state)
		if err != nil {
			result.Error = err.Error()
		}

		state.AgentHistory = append(state.AgentHistory, result)
		state.LastOutput = result.Output
		state.CurrentStep++

		// Check for completion (simple heuristic)
		if result.Error == "" && state.CurrentStep >= 3 {
			state.IsComplete = true
		}
	}

	return &MultiAgentOutput{
		Result:        state.LastOutput,
		AgentResults:  state.AgentHistory,
		TotalDuration: time.Since(start),
	}, nil
}

func (o *RandomOrchestrator) invokeAgent(ctx context.Context, ref AgentRef, state *OrchestrationState) (AgentResult, error) {
	start := time.Now()
	result := AgentResult{
		AgentType: ref.Type,
		AgentID:   ref.ID,
		Input:     fmt.Sprintf("%v", state.LastOutput),
	}

	runInput := &RunInput{
		Prompt:      fmt.Sprintf("%v", state.LastOutput),
		Context:     state.Context,
		Synchronous: true,
	}

	data, err := json.Marshal(runInput)
	if err != nil {
		result.Duration = time.Since(start)
		return result, err
	}

	resp, err := o.daprClient.InvokeActor(ctx, &client.InvokeActorRequest{
		ActorType: ref.Type,
		ActorID:   ref.ID,
		Method:    "Run",
		Data:      data,
	})
	if err != nil {
		result.Duration = time.Since(start)
		return result, err
	}

	var output RunOutput
	if err := json.Unmarshal(resp.Data, &output); err != nil {
		result.Duration = time.Since(start)
		return result, err
	}

	result.Output = output.Result
	result.Duration = time.Since(start)
	return result, nil
}

// RoundRobinOrchestrator selects agents in round-robin order
type RoundRobinOrchestrator struct {
	agents     []AgentRef
	maxSteps   int
	daprClient client.Client
	currentIdx int
	mu         sync.Mutex
}

// NewRoundRobinOrchestrator creates a round-robin orchestrator
func NewRoundRobinOrchestrator(agents []AgentRef, maxSteps int) (*RoundRobinOrchestrator, error) {
	c, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	if maxSteps <= 0 {
		maxSteps = 10
	}

	return &RoundRobinOrchestrator{
		agents:     agents,
		maxSteps:   maxSteps,
		daprClient: c,
	}, nil
}

func (o *RoundRobinOrchestrator) SelectNextAgent(ctx context.Context, state *OrchestrationState) (AgentRef, error) {
	if len(o.agents) == 0 {
		return AgentRef{}, fmt.Errorf("no agents available")
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	agent := o.agents[o.currentIdx]
	o.currentIdx = (o.currentIdx + 1) % len(o.agents)
	return agent, nil
}

func (o *RoundRobinOrchestrator) Execute(ctx context.Context, input *MultiAgentInput) (*MultiAgentOutput, error) {
	start := time.Now()
	state := &OrchestrationState{
		Input:        input,
		MaxSteps:     o.maxSteps,
		AgentHistory: make([]AgentResult, 0),
		Context:      make(map[string]interface{}),
		LastOutput:   input.Prompt,
	}

	for state.CurrentStep < state.MaxSteps && !state.IsComplete {
		agentRef, err := o.SelectNextAgent(ctx, state)
		if err != nil {
			return nil, err
		}

		result, err := o.invokeAgent(ctx, agentRef, state)
		if err != nil {
			result.Error = err.Error()
		}

		state.AgentHistory = append(state.AgentHistory, result)
		state.LastOutput = result.Output
		state.CurrentStep++

		// Complete after all agents have been called once
		if state.CurrentStep >= len(o.agents) {
			state.IsComplete = true
		}
	}

	return &MultiAgentOutput{
		Result:        state.LastOutput,
		AgentResults:  state.AgentHistory,
		TotalDuration: time.Since(start),
	}, nil
}

func (o *RoundRobinOrchestrator) invokeAgent(ctx context.Context, ref AgentRef, state *OrchestrationState) (AgentResult, error) {
	start := time.Now()
	result := AgentResult{
		AgentType: ref.Type,
		AgentID:   ref.ID,
		Input:     fmt.Sprintf("%v", state.LastOutput),
	}

	runInput := &RunInput{
		Prompt:      fmt.Sprintf("%v", state.LastOutput),
		Context:     state.Context,
		Synchronous: true,
	}

	data, err := json.Marshal(runInput)
	if err != nil {
		result.Duration = time.Since(start)
		return result, err
	}

	resp, err := o.daprClient.InvokeActor(ctx, &client.InvokeActorRequest{
		ActorType: ref.Type,
		ActorID:   ref.ID,
		Method:    "Run",
		Data:      data,
	})
	if err != nil {
		result.Duration = time.Since(start)
		return result, err
	}

	var output RunOutput
	if err := json.Unmarshal(resp.Data, &output); err != nil {
		result.Duration = time.Since(start)
		return result, err
	}

	result.Output = output.Result
	result.Duration = time.Since(start)
	return result, nil
}

// LLMOrchestrator uses an LLM to decide which agent to call next
type LLMOrchestrator struct {
	agents        []AgentRef
	plannerAgent  AgentRef
	maxSteps      int
	daprClient    client.Client
	llmComponent  string
}

// LLMOrchestratorConfig configures the LLM orchestrator
type LLMOrchestratorConfig struct {
	Agents       []AgentRef
	PlannerAgent AgentRef // Agent that decides next steps
	MaxSteps     int
	LLMComponent string
}

// NewLLMOrchestrator creates an LLM-based orchestrator
func NewLLMOrchestrator(config *LLMOrchestratorConfig) (*LLMOrchestrator, error) {
	c, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	if config.MaxSteps <= 0 {
		config.MaxSteps = 10
	}

	return &LLMOrchestrator{
		agents:       config.Agents,
		plannerAgent: config.PlannerAgent,
		maxSteps:     config.MaxSteps,
		daprClient:   c,
		llmComponent: config.LLMComponent,
	}, nil
}

// PlannerResponse represents the planner's decision
type PlannerResponse struct {
	NextAgent   string                 `json:"nextAgent"`
	Task        string                 `json:"task"`
	Reasoning   string                 `json:"reasoning"`
	IsComplete  bool                   `json:"isComplete"`
	FinalAnswer string                 `json:"finalAnswer,omitempty"`
}

func (o *LLMOrchestrator) SelectNextAgent(ctx context.Context, state *OrchestrationState) (AgentRef, error) {
	// Build context for the planner
	agentDescriptions := make([]map[string]string, len(o.agents))
	for i, a := range o.agents {
		agentDescriptions[i] = map[string]string{
			"type": a.Type,
			"id":   a.ID,
		}
	}

	history := make([]map[string]interface{}, len(state.AgentHistory))
	for i, h := range state.AgentHistory {
		history[i] = map[string]interface{}{
			"agent":  h.AgentType,
			"input":  h.Input,
			"output": h.Output,
			"error":  h.Error,
		}
	}

	plannerInput := map[string]interface{}{
		"task":           state.Input.Prompt,
		"availableAgents": agentDescriptions,
		"history":        history,
		"currentStep":    state.CurrentStep,
		"maxSteps":       state.MaxSteps,
	}

	// If we have a planner agent, use it
	if o.plannerAgent.Type != "" {
		data, err := json.Marshal(&RunInput{
			Prompt:      fmt.Sprintf("Decide the next step for task: %s", state.Input.Prompt),
			Context:     plannerInput,
			Synchronous: true,
		})
		if err != nil {
			return AgentRef{}, err
		}

		resp, err := o.daprClient.InvokeActor(ctx, &client.InvokeActorRequest{
			ActorType: o.plannerAgent.Type,
			ActorID:   o.plannerAgent.ID,
			Method:    "Run",
			Data:      data,
		})
		if err != nil {
			return AgentRef{}, err
		}

		var output RunOutput
		if err := json.Unmarshal(resp.Data, &output); err != nil {
			return AgentRef{}, err
		}

		// Parse the planner's response
		var plannerResp PlannerResponse
		if err := json.Unmarshal([]byte(output.Result), &plannerResp); err != nil {
			// If parsing fails, try to find an agent name in the response
			for _, a := range o.agents {
				if contains(output.Result, a.Type) {
					return a, nil
				}
			}
			// Default to first agent
			if len(o.agents) > 0 {
				return o.agents[0], nil
			}
		}

		if plannerResp.IsComplete {
			state.IsComplete = true
			state.LastOutput = plannerResp.FinalAnswer
			return AgentRef{}, nil
		}

		// Find the selected agent
		for _, a := range o.agents {
			if a.Type == plannerResp.NextAgent {
				return a, nil
			}
		}
	}

	// Fallback: use LLM directly via Dapr
	if o.llmComponent != "" {
		return o.selectViaLLM(ctx, plannerInput)
	}

	// Last resort: round-robin
	if len(o.agents) > 0 {
		return o.agents[state.CurrentStep%len(o.agents)], nil
	}

	return AgentRef{}, fmt.Errorf("no agents available")
}

func (o *LLMOrchestrator) selectViaLLM(ctx context.Context, plannerInput map[string]interface{}) (AgentRef, error) {
	prompt := fmt.Sprintf(`Given the following task and available agents, select the best agent to handle the next step.

Task: %v
Available agents: %v
History: %v

Respond with ONLY the agent type name (e.g., "WeatherAgent").`,
		plannerInput["task"],
		plannerInput["availableAgents"],
		plannerInput["history"])

	input := map[string]interface{}{
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}

	data, err := json.Marshal(input)
	if err != nil {
		return AgentRef{}, err
	}

	resp, err := o.daprClient.InvokeBinding(ctx, &client.InvokeBindingRequest{
		Name:      o.llmComponent,
		Operation: "conversation",
		Data:      data,
	})
	if err != nil {
		return AgentRef{}, err
	}

	var llmResp struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(resp.Data, &llmResp); err != nil {
		return AgentRef{}, err
	}

	// Find matching agent
	for _, a := range o.agents {
		if contains(llmResp.Content, a.Type) {
			return a, nil
		}
	}

	// Default to first
	if len(o.agents) > 0 {
		return o.agents[0], nil
	}

	return AgentRef{}, fmt.Errorf("no matching agent found")
}

func (o *LLMOrchestrator) Execute(ctx context.Context, input *MultiAgentInput) (*MultiAgentOutput, error) {
	start := time.Now()
	state := &OrchestrationState{
		Input:        input,
		MaxSteps:     o.maxSteps,
		AgentHistory: make([]AgentResult, 0),
		Context:      input.Context,
		LastOutput:   input.Prompt,
	}

	if state.Context == nil {
		state.Context = make(map[string]interface{})
	}

	for state.CurrentStep < state.MaxSteps && !state.IsComplete {
		agentRef, err := o.SelectNextAgent(ctx, state)
		if err != nil {
			if state.IsComplete {
				break
			}
			return nil, err
		}

		if agentRef.Type == "" {
			break
		}

		result, err := o.invokeAgent(ctx, agentRef, state)
		if err != nil {
			result.Error = err.Error()
		}

		state.AgentHistory = append(state.AgentHistory, result)
		state.LastOutput = result.Output
		state.CurrentStep++
	}

	return &MultiAgentOutput{
		Result:        state.LastOutput,
		AgentResults:  state.AgentHistory,
		TotalDuration: time.Since(start),
	}, nil
}

func (o *LLMOrchestrator) invokeAgent(ctx context.Context, ref AgentRef, state *OrchestrationState) (AgentResult, error) {
	start := time.Now()
	result := AgentResult{
		AgentType: ref.Type,
		AgentID:   ref.ID,
		Input:     fmt.Sprintf("%v", state.LastOutput),
	}

	runInput := &RunInput{
		Prompt:      fmt.Sprintf("%v", state.LastOutput),
		Context:     state.Context,
		Synchronous: true,
	}

	data, err := json.Marshal(runInput)
	if err != nil {
		result.Duration = time.Since(start)
		return result, err
	}

	resp, err := o.daprClient.InvokeActor(ctx, &client.InvokeActorRequest{
		ActorType: ref.Type,
		ActorID:   ref.ID,
		Method:    "Run",
		Data:      data,
	})
	if err != nil {
		result.Duration = time.Since(start)
		return result, err
	}

	var output RunOutput
	if err := json.Unmarshal(resp.Data, &output); err != nil {
		result.Duration = time.Since(start)
		return result, err
	}

	result.Output = output.Result
	result.Duration = time.Since(start)
	return result, nil
}

// OrchestratorFactory creates orchestrators by type
func NewOrchestrator(orchType string, agents []AgentRef, config map[string]interface{}) (Orchestrator, error) {
	maxSteps := 10
	if ms, ok := config["maxSteps"].(int); ok {
		maxSteps = ms
	}

	switch orchType {
	case "random":
		return NewRandomOrchestrator(agents, maxSteps)

	case "roundrobin":
		return NewRoundRobinOrchestrator(agents, maxSteps)

	case "llm":
		llmConfig := &LLMOrchestratorConfig{
			Agents:   agents,
			MaxSteps: maxSteps,
		}
		if comp, ok := config["llmComponent"].(string); ok {
			llmConfig.LLMComponent = comp
		}
		if planner, ok := config["plannerAgent"].(AgentRef); ok {
			llmConfig.PlannerAgent = planner
		}
		return NewLLMOrchestrator(llmConfig)

	case "chain":
		return NewChainPattern(agents...)

	case "parallel":
		var aggregator func([]AgentResult) interface{}
		if agg, ok := config["aggregator"].(func([]AgentResult) interface{}); ok {
			aggregator = agg
		}
		return NewParallelPattern(agents, aggregator)

	default:
		return nil, fmt.Errorf("unknown orchestrator type: %s", orchType)
	}
}
