package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/dapr/go-sdk/client"
)

// MultiAgentPattern defines patterns for multi-agent collaboration
type MultiAgentPattern interface {
	// Execute runs the multi-agent pattern
	Execute(ctx context.Context, input *MultiAgentInput) (*MultiAgentOutput, error)
}

// MultiAgentInput represents input for multi-agent execution
type MultiAgentInput struct {
	Prompt   string                 `json:"prompt"`
	Context  map[string]interface{} `json:"context,omitempty"`
	Timeout  time.Duration          `json:"timeout,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// MultiAgentOutput represents output from multi-agent execution
type MultiAgentOutput struct {
	Result        interface{}            `json:"result"`
	AgentResults  []AgentResult          `json:"agentResults,omitempty"`
	TotalDuration time.Duration          `json:"totalDuration"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// AgentResult represents the result from a single agent in a multi-agent pattern
type AgentResult struct {
	AgentType   string        `json:"agentType"`
	AgentID     string        `json:"agentId"`
	Input       string        `json:"input,omitempty"`
	Output      interface{}   `json:"output"`
	Duration    time.Duration `json:"duration"`
	Error       string        `json:"error,omitempty"`
}

// AgentRef references an agent in the cluster
type AgentRef struct {
	Type  string // Agent type (actor type)
	ID    string // Agent ID (actor ID)
	AppID string // Dapr app ID (if remote)
}

// ChainPattern executes agents in sequence, passing output to the next
type ChainPattern struct {
	agents     []AgentRef
	daprClient client.Client
}

// NewChainPattern creates a chain of agents
func NewChainPattern(agents ...AgentRef) (*ChainPattern, error) {
	c, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	return &ChainPattern{
		agents:     agents,
		daprClient: c,
	}, nil
}

// Execute runs agents in sequence
func (p *ChainPattern) Execute(ctx context.Context, input *MultiAgentInput) (*MultiAgentOutput, error) {
	start := time.Now()
	results := make([]AgentResult, 0, len(p.agents))

	currentInput := input.Prompt
	currentContext := input.Context

	for _, agentRef := range p.agents {
		agentStart := time.Now()

		runInput := &RunInput{
			Prompt:      currentInput,
			Context:     currentContext,
			Synchronous: true,
		}

		output, err := p.invokeAgent(ctx, agentRef, runInput)

		result := AgentResult{
			AgentType: agentRef.Type,
			AgentID:   agentRef.ID,
			Input:     currentInput,
			Duration:  time.Since(agentStart),
		}

		if err != nil {
			result.Error = err.Error()
			results = append(results, result)
			return &MultiAgentOutput{
				AgentResults:  results,
				TotalDuration: time.Since(start),
			}, err
		}

		result.Output = output.Result
		results = append(results, result)

		// Use output as input for next agent
		currentInput = output.Result
		if currentContext == nil {
			currentContext = make(map[string]interface{})
		}
		currentContext["previousResult"] = output.Result
	}

	// Return the last result
	var finalResult interface{}
	if len(results) > 0 {
		finalResult = results[len(results)-1].Output
	}

	return &MultiAgentOutput{
		Result:        finalResult,
		AgentResults:  results,
		TotalDuration: time.Since(start),
	}, nil
}

func (p *ChainPattern) invokeAgent(ctx context.Context, ref AgentRef, input *RunInput) (*RunOutput, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	resp, err := p.daprClient.InvokeActor(ctx, &client.InvokeActorRequest{
		ActorType: ref.Type,
		ActorID:   ref.ID,
		Method:    "Run",
		Data:      data,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to invoke agent: %w", err)
	}

	var output RunOutput
	if err := json.Unmarshal(resp.Data, &output); err != nil {
		return nil, fmt.Errorf("failed to unmarshal output: %w", err)
	}

	return &output, nil
}

// ParallelPattern executes agents concurrently and aggregates results
type ParallelPattern struct {
	agents      []AgentRef
	aggregator  func(results []AgentResult) interface{}
	daprClient  client.Client
}

// NewParallelPattern creates a parallel execution pattern
func NewParallelPattern(agents []AgentRef, aggregator func(results []AgentResult) interface{}) (*ParallelPattern, error) {
	c, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	if aggregator == nil {
		// Default aggregator: return all results as array
		aggregator = func(results []AgentResult) interface{} {
			outputs := make([]interface{}, len(results))
			for i, r := range results {
				outputs[i] = r.Output
			}
			return outputs
		}
	}

	return &ParallelPattern{
		agents:     agents,
		aggregator: aggregator,
		daprClient: c,
	}, nil
}

// Execute runs agents in parallel
func (p *ParallelPattern) Execute(ctx context.Context, input *MultiAgentInput) (*MultiAgentOutput, error) {
	start := time.Now()

	var wg sync.WaitGroup
	resultsCh := make(chan AgentResult, len(p.agents))

	for _, agentRef := range p.agents {
		wg.Add(1)
		go func(ref AgentRef) {
			defer wg.Done()

			agentStart := time.Now()
			runInput := &RunInput{
				Prompt:      input.Prompt,
				Context:     input.Context,
				Synchronous: true,
			}

			output, err := p.invokeAgent(ctx, ref, runInput)

			result := AgentResult{
				AgentType: ref.Type,
				AgentID:   ref.ID,
				Input:     input.Prompt,
				Duration:  time.Since(agentStart),
			}

			if err != nil {
				result.Error = err.Error()
			} else {
				result.Output = output.Result
			}

			resultsCh <- result
		}(agentRef)
	}

	wg.Wait()
	close(resultsCh)

	results := make([]AgentResult, 0, len(p.agents))
	for result := range resultsCh {
		results = append(results, result)
	}

	return &MultiAgentOutput{
		Result:        p.aggregator(results),
		AgentResults:  results,
		TotalDuration: time.Since(start),
	}, nil
}

func (p *ParallelPattern) invokeAgent(ctx context.Context, ref AgentRef, input *RunInput) (*RunOutput, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	resp, err := p.daprClient.InvokeActor(ctx, &client.InvokeActorRequest{
		ActorType: ref.Type,
		ActorID:   ref.ID,
		Method:    "Run",
		Data:      data,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to invoke agent: %w", err)
	}

	var output RunOutput
	if err := json.Unmarshal(resp.Data, &output); err != nil {
		return nil, fmt.Errorf("failed to unmarshal output: %w", err)
	}

	return &output, nil
}

// RouterPattern routes requests to different agents based on criteria
type RouterPattern struct {
	router     func(input *MultiAgentInput) AgentRef
	agents     map[string]AgentRef
	daprClient client.Client
}

// NewRouterPattern creates a router pattern
func NewRouterPattern(agents map[string]AgentRef, router func(input *MultiAgentInput) AgentRef) (*RouterPattern, error) {
	c, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	return &RouterPattern{
		router:     router,
		agents:     agents,
		daprClient: c,
	}, nil
}

// Execute routes and executes the appropriate agent
func (p *RouterPattern) Execute(ctx context.Context, input *MultiAgentInput) (*MultiAgentOutput, error) {
	start := time.Now()

	// Determine which agent to use
	agentRef := p.router(input)

	runInput := &RunInput{
		Prompt:      input.Prompt,
		Context:     input.Context,
		Synchronous: true,
	}

	output, err := p.invokeAgent(ctx, agentRef, runInput)

	result := AgentResult{
		AgentType: agentRef.Type,
		AgentID:   agentRef.ID,
		Input:     input.Prompt,
		Duration:  time.Since(start),
	}

	if err != nil {
		result.Error = err.Error()
		return &MultiAgentOutput{
			AgentResults:  []AgentResult{result},
			TotalDuration: time.Since(start),
		}, err
	}

	result.Output = output.Result

	return &MultiAgentOutput{
		Result:        output.Result,
		AgentResults:  []AgentResult{result},
		TotalDuration: time.Since(start),
	}, nil
}

func (p *RouterPattern) invokeAgent(ctx context.Context, ref AgentRef, input *RunInput) (*RunOutput, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	resp, err := p.daprClient.InvokeActor(ctx, &client.InvokeActorRequest{
		ActorType: ref.Type,
		ActorID:   ref.ID,
		Method:    "Run",
		Data:      data,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to invoke agent: %w", err)
	}

	var output RunOutput
	if err := json.Unmarshal(resp.Data, &output); err != nil {
		return nil, fmt.Errorf("failed to unmarshal output: %w", err)
	}

	return &output, nil
}

// SupervisorPattern has a supervisor agent that delegates to worker agents
type SupervisorPattern struct {
	supervisor  AgentRef
	workers     []AgentRef
	daprClient  client.Client
}

// SupervisorTask represents a task delegated to a worker
type SupervisorTask struct {
	WorkerType string                 `json:"workerType"`
	WorkerID   string                 `json:"workerId"`
	Task       string                 `json:"task"`
	Context    map[string]interface{} `json:"context,omitempty"`
}

// NewSupervisorPattern creates a supervisor pattern
func NewSupervisorPattern(supervisor AgentRef, workers ...AgentRef) (*SupervisorPattern, error) {
	c, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	return &SupervisorPattern{
		supervisor: supervisor,
		workers:    workers,
		daprClient: c,
	}, nil
}

// Execute runs the supervisor pattern
func (p *SupervisorPattern) Execute(ctx context.Context, input *MultiAgentInput) (*MultiAgentOutput, error) {
	start := time.Now()
	results := make([]AgentResult, 0)

	// First, ask supervisor to plan
	planInput := &RunInput{
		Prompt:      input.Prompt,
		Context:     input.Context,
		Synchronous: true,
	}

	// Add worker info to context
	if planInput.Context == nil {
		planInput.Context = make(map[string]interface{})
	}
	workers := make([]map[string]string, len(p.workers))
	for i, w := range p.workers {
		workers[i] = map[string]string{
			"type": w.Type,
			"id":   w.ID,
		}
	}
	planInput.Context["availableWorkers"] = workers

	supervisorStart := time.Now()
	planOutput, err := p.invokeAgent(ctx, p.supervisor, planInput)

	supervisorResult := AgentResult{
		AgentType: p.supervisor.Type,
		AgentID:   p.supervisor.ID,
		Input:     input.Prompt,
		Duration:  time.Since(supervisorStart),
	}

	if err != nil {
		supervisorResult.Error = err.Error()
		results = append(results, supervisorResult)
		return &MultiAgentOutput{
			AgentResults:  results,
			TotalDuration: time.Since(start),
		}, err
	}

	supervisorResult.Output = planOutput.Result
	results = append(results, supervisorResult)

	// The supervisor's response should contain delegated tasks
	// In a real implementation, you'd parse the supervisor's output
	// and execute the delegated tasks

	return &MultiAgentOutput{
		Result:        planOutput.Result,
		AgentResults:  results,
		TotalDuration: time.Since(start),
	}, nil
}

func (p *SupervisorPattern) invokeAgent(ctx context.Context, ref AgentRef, input *RunInput) (*RunOutput, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	resp, err := p.daprClient.InvokeActor(ctx, &client.InvokeActorRequest{
		ActorType: ref.Type,
		ActorID:   ref.ID,
		Method:    "Run",
		Data:      data,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to invoke agent: %w", err)
	}

	var output RunOutput
	if err := json.Unmarshal(resp.Data, &output); err != nil {
		return nil, fmt.Errorf("failed to unmarshal output: %w", err)
	}

	return &output, nil
}

// CollaborativePattern allows agents to communicate and collaborate freely
type CollaborativePattern struct {
	agents       []AgentRef
	coordinator  AgentRef
	maxRounds    int
	daprClient   client.Client
}

// NewCollaborativePattern creates a collaborative pattern
func NewCollaborativePattern(coordinator AgentRef, agents []AgentRef, maxRounds int) (*CollaborativePattern, error) {
	c, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	if maxRounds <= 0 {
		maxRounds = 5
	}

	return &CollaborativePattern{
		agents:      agents,
		coordinator: coordinator,
		maxRounds:   maxRounds,
		daprClient:  c,
	}, nil
}

// Execute runs the collaborative pattern
func (p *CollaborativePattern) Execute(ctx context.Context, input *MultiAgentInput) (*MultiAgentOutput, error) {
	start := time.Now()
	results := make([]AgentResult, 0)

	// Add all agents to context
	agentInfo := make([]map[string]string, len(p.agents))
	for i, a := range p.agents {
		agentInfo[i] = map[string]string{
			"type": a.Type,
			"id":   a.ID,
		}
	}

	currentContext := input.Context
	if currentContext == nil {
		currentContext = make(map[string]interface{})
	}
	currentContext["collaborators"] = agentInfo
	currentContext["maxRounds"] = p.maxRounds

	// Start with coordinator
	coordInput := &RunInput{
		Prompt:      input.Prompt,
		Context:     currentContext,
		Synchronous: true,
	}

	coordStart := time.Now()
	coordOutput, err := p.invokeAgent(ctx, p.coordinator, coordInput)

	coordResult := AgentResult{
		AgentType: p.coordinator.Type,
		AgentID:   p.coordinator.ID,
		Input:     input.Prompt,
		Duration:  time.Since(coordStart),
	}

	if err != nil {
		coordResult.Error = err.Error()
		results = append(results, coordResult)
		return &MultiAgentOutput{
			AgentResults:  results,
			TotalDuration: time.Since(start),
		}, err
	}

	coordResult.Output = coordOutput.Result
	results = append(results, coordResult)

	// In a full implementation, the coordinator would direct collaboration
	// between agents over multiple rounds

	return &MultiAgentOutput{
		Result:        coordOutput.Result,
		AgentResults:  results,
		TotalDuration: time.Since(start),
	}, nil
}

func (p *CollaborativePattern) invokeAgent(ctx context.Context, ref AgentRef, input *RunInput) (*RunOutput, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	resp, err := p.daprClient.InvokeActor(ctx, &client.InvokeActorRequest{
		ActorType: ref.Type,
		ActorID:   ref.ID,
		Method:    "Run",
		Data:      data,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to invoke agent: %w", err)
	}

	var output RunOutput
	if err := json.Unmarshal(resp.Data, &output); err != nil {
		return nil, fmt.Errorf("failed to unmarshal output: %w", err)
	}

	return &output, nil
}
