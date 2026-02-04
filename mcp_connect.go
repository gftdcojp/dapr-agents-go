package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"connectrpc.com/connect"
	mcpv1 "github.com/gftdcojp/dapr-agents-go/gen/mcp/v1"
	"github.com/gftdcojp/dapr-agents-go/gen/mcp/v1/mcpv1connect"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
)

// MCPConnectServer implements Model Context Protocol over Connect RPC
type MCPConnectServer struct {
	config *MCPConnectConfig
	agents map[string]func() Agent
	tools  map[string]Tool
	server *http.Server
	logger *log.Logger
	mu     sync.RWMutex
}

// MCPConnectConfig configures the MCP Connect server
type MCPConnectConfig struct {
	Port         int
	Name         string
	Version      string
	Description  string
	AllowOrigins []string
	Logger       *log.Logger
}

// DefaultMCPConnectConfig returns sensible defaults
func DefaultMCPConnectConfig() *MCPConnectConfig {
	return &MCPConnectConfig{
		Port:         8082,
		Name:         "dapr-agent-mcp-connect",
		Version:      Version,
		Description:  "Dapr Agent MCP Connect Server",
		AllowOrigins: []string{"*"},
		Logger:       log.Default(),
	}
}

// NewMCPConnectServer creates a new MCP Connect server
func NewMCPConnectServer(config *MCPConnectConfig) *MCPConnectServer {
	if config == nil {
		config = DefaultMCPConnectConfig()
	}
	if config.Logger == nil {
		config.Logger = log.Default()
	}

	return &MCPConnectServer{
		config: config,
		agents: make(map[string]func() Agent),
		tools:  make(map[string]Tool),
		logger: config.Logger,
	}
}

// RegisterAgent registers an agent to be exposed via MCP Connect
func (s *MCPConnectServer) RegisterAgent(name string, factory func() Agent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.agents[name] = factory
	s.logger.Printf("MCP Connect: Registered agent: %s", name)
}

// RegisterTool registers a tool to be exposed via MCP Connect
func (s *MCPConnectServer) RegisterTool(tool Tool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tools[tool.Name()] = tool
	s.logger.Printf("MCP Connect: Registered tool: %s", tool.Name())
}

// Start starts the MCP Connect server
func (s *MCPConnectServer) Start() error {
	mux := http.NewServeMux()

	// Register Connect service
	path, handler := mcpv1connect.NewMCPServiceHandler(s, s.withCORSInterceptor())
	mux.Handle(path, handler)

	// Health endpoints
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/healthz", s.handleHealth)

	// Add CORS preflight handler
	corsHandler := s.corsMiddleware(mux)

	s.server = &http.Server{
		Addr:         fmt.Sprintf(":%d", s.config.Port),
		Handler:      h2c.NewHandler(corsHandler, &http2.Server{}),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
	}

	s.logger.Printf("MCP Connect server starting on port %d", s.config.Port)
	return s.server.ListenAndServe()
}

// Stop stops the MCP Connect server
func (s *MCPConnectServer) Stop(ctx context.Context) error {
	if s.server != nil {
		return s.server.Shutdown(ctx)
	}
	return nil
}

func (s *MCPConnectServer) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin == "" {
			origin = "*"
		}

		allowed := false
		for _, o := range s.config.AllowOrigins {
			if o == "*" || o == origin {
				allowed = true
				break
			}
		}

		if allowed {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Connect-Protocol-Version, Connect-Timeout-Ms, Grpc-Timeout, X-Grpc-Web, X-User-Agent")
			w.Header().Set("Access-Control-Expose-Headers", "Grpc-Status, Grpc-Message, Grpc-Status-Details-Bin")
			w.Header().Set("Access-Control-Max-Age", "86400")
		}

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (s *MCPConnectServer) withCORSInterceptor() connect.HandlerOption {
	return connect.WithInterceptors(connect.UnaryInterceptorFunc(func(next connect.UnaryFunc) connect.UnaryFunc {
		return func(ctx context.Context, req connect.AnyRequest) (connect.AnyResponse, error) {
			return next(ctx, req)
		}
	}))
}

func (s *MCPConnectServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
	})
}

// GetServerInfo implements MCPServiceHandler
func (s *MCPConnectServer) GetServerInfo(
	ctx context.Context,
	req *connect.Request[mcpv1.GetServerInfoRequest],
) (*connect.Response[mcpv1.GetServerInfoResponse], error) {
	resp := &mcpv1.GetServerInfoResponse{
		Name:        s.config.Name,
		Version:     s.config.Version,
		Description: s.config.Description,
		Capabilities: []string{
			"tools",
			"prompts",
			"resources",
			"agents",
			"streaming",
		},
		Metadata: map[string]string{
			"provider":  "dapr-agent-sdk",
			"transport": "connect-rpc",
		},
	}
	return connect.NewResponse(resp), nil
}

// ListTools implements MCPServiceHandler
func (s *MCPConnectServer) ListTools(
	ctx context.Context,
	req *connect.Request[mcpv1.ListToolsRequest],
) (*connect.Response[mcpv1.ListToolsResponse], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	tools := make([]*mcpv1.Tool, 0, len(s.tools))

	// Add direct tools
	for _, tool := range s.tools {
		schema := tool.Schema()
		schemaJSON, _ := json.Marshal(map[string]interface{}{
			"type":       schema.Type,
			"properties": schema.Properties,
			"required":   schema.Required,
		})

		tools = append(tools, &mcpv1.Tool{
			Name:            tool.Name(),
			Description:     tool.Description(),
			InputSchemaJson: string(schemaJSON),
		})
	}

	// Add agent tools
	for name, factory := range s.agents {
		agent := factory()
		for _, tool := range agent.GetTools() {
			schema := tool.Schema()
			schemaJSON, _ := json.Marshal(map[string]interface{}{
				"type":       schema.Type,
				"properties": schema.Properties,
				"required":   schema.Required,
			})

			tools = append(tools, &mcpv1.Tool{
				Name:            fmt.Sprintf("%s/%s", name, tool.Name()),
				Description:     tool.Description(),
				InputSchemaJson: string(schemaJSON),
			})
		}
	}

	return connect.NewResponse(&mcpv1.ListToolsResponse{Tools: tools}), nil
}

// CallTool implements MCPServiceHandler
func (s *MCPConnectServer) CallTool(
	ctx context.Context,
	req *connect.Request[mcpv1.CallToolRequest],
) (*connect.Response[mcpv1.CallToolResponse], error) {
	toolName := req.Msg.Name

	var args map[string]interface{}
	if err := json.Unmarshal([]byte(req.Msg.ArgumentsJson), &args); err != nil {
		return nil, connect.NewError(connect.CodeInvalidArgument, fmt.Errorf("invalid arguments JSON: %w", err))
	}

	s.mu.RLock()
	var tool Tool
	var found bool

	// Check for agent tool (format: agentName/toolName)
	if idx := findSlashIndex(toolName); idx > 0 {
		agentName := toolName[:idx]
		localToolName := toolName[idx+1:]
		if factory, ok := s.agents[agentName]; ok {
			agent := factory()
			for _, t := range agent.GetTools() {
				if t.Name() == localToolName {
					tool = t
					found = true
					break
				}
			}
		}
	} else {
		tool, found = s.tools[toolName]
	}
	s.mu.RUnlock()

	if !found {
		return nil, connect.NewError(connect.CodeNotFound, fmt.Errorf("tool not found: %s", toolName))
	}

	result, err := tool.Execute(ctx, args)

	resp := &mcpv1.CallToolResponse{
		Content: make([]*mcpv1.Content, 0),
	}

	if err != nil {
		resp.IsError = true
		resp.Content = append(resp.Content, &mcpv1.Content{
			Type: "text",
			Text: fmt.Sprintf("Error: %s", err.Error()),
		})
	} else {
		var text string
		switch v := result.(type) {
		case string:
			text = v
		default:
			data, _ := json.Marshal(result)
			text = string(data)
		}
		resp.Content = append(resp.Content, &mcpv1.Content{
			Type: "text",
			Text: text,
		})
	}

	return connect.NewResponse(resp), nil
}

// ListPrompts implements MCPServiceHandler
func (s *MCPConnectServer) ListPrompts(
	ctx context.Context,
	req *connect.Request[mcpv1.ListPromptsRequest],
) (*connect.Response[mcpv1.ListPromptsResponse], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	prompts := make([]*mcpv1.Prompt, 0)
	for name := range s.agents {
		prompts = append(prompts, &mcpv1.Prompt{
			Name:        name,
			Description: fmt.Sprintf("Run %s agent with a prompt", name),
			Arguments: []*mcpv1.PromptArgument{
				{Name: "prompt", Description: "The prompt to send to the agent", Required: true},
				{Name: "context", Description: "Additional context (JSON)", Required: false},
			},
		})
	}

	return connect.NewResponse(&mcpv1.ListPromptsResponse{Prompts: prompts}), nil
}

// GetPrompt implements MCPServiceHandler
func (s *MCPConnectServer) GetPrompt(
	ctx context.Context,
	req *connect.Request[mcpv1.GetPromptRequest],
) (*connect.Response[mcpv1.GetPromptResponse], error) {
	promptName := req.Msg.Name

	s.mu.RLock()
	_, exists := s.agents[promptName]
	s.mu.RUnlock()

	if !exists {
		return nil, connect.NewError(connect.CodeNotFound, fmt.Errorf("prompt not found: %s", promptName))
	}

	resp := &mcpv1.GetPromptResponse{
		Description: fmt.Sprintf("Run %s agent", promptName),
		Messages: []*mcpv1.PromptMessage{
			{Role: "user", Content: "{{prompt}}"},
		},
	}

	return connect.NewResponse(resp), nil
}

// ListResources implements MCPServiceHandler
func (s *MCPConnectServer) ListResources(
	ctx context.Context,
	req *connect.Request[mcpv1.ListResourcesRequest],
) (*connect.Response[mcpv1.ListResourcesResponse], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	resources := make([]*mcpv1.Resource, 0)
	for name := range s.agents {
		resources = append(resources, &mcpv1.Resource{
			Uri:         fmt.Sprintf("agent://%s", name),
			Name:        name,
			Description: fmt.Sprintf("Agent: %s", name),
			MimeType:    "application/json",
		})
	}

	return connect.NewResponse(&mcpv1.ListResourcesResponse{Resources: resources}), nil
}

// GetResource implements MCPServiceHandler
func (s *MCPConnectServer) GetResource(
	ctx context.Context,
	req *connect.Request[mcpv1.GetResourceRequest],
) (*connect.Response[mcpv1.GetResourceResponse], error) {
	uri := req.Msg.Uri

	if len(uri) > 8 && uri[:8] == "agent://" {
		agentName := uri[8:]
		s.mu.RLock()
		factory, exists := s.agents[agentName]
		s.mu.RUnlock()

		if !exists {
			return nil, connect.NewError(connect.CodeNotFound, fmt.Errorf("agent not found: %s", agentName))
		}

		agent := factory()
		tools := agent.GetTools()
		toolNames := make([]string, len(tools))
		for i, t := range tools {
			toolNames[i] = t.Name()
		}

		contentJSON, _ := json.Marshal(map[string]interface{}{
			"name":  agentName,
			"type":  "agent",
			"tools": toolNames,
		})

		resp := &mcpv1.GetResourceResponse{
			Contents: []*mcpv1.ResourceContent{
				{
					Uri:      uri,
					MimeType: "application/json",
					Text:     string(contentJSON),
				},
			},
		}

		return connect.NewResponse(resp), nil
	}

	return nil, connect.NewError(connect.CodeNotFound, fmt.Errorf("resource not found: %s", uri))
}

// ListAgents implements MCPServiceHandler
func (s *MCPConnectServer) ListAgents(
	ctx context.Context,
	req *connect.Request[mcpv1.ListAgentsRequest],
) (*connect.Response[mcpv1.ListAgentsResponse], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	agents := make([]*mcpv1.Agent, 0, len(s.agents))
	for name, factory := range s.agents {
		agent := factory()
		var tools []string
		for _, t := range agent.GetTools() {
			tools = append(tools, t.Name())
		}

		agents = append(agents, &mcpv1.Agent{
			Name:  name,
			Tools: tools,
		})
	}

	return connect.NewResponse(&mcpv1.ListAgentsResponse{Agents: agents}), nil
}

// RunAgent implements MCPServiceHandler (streaming)
func (s *MCPConnectServer) RunAgent(
	ctx context.Context,
	req *connect.Request[mcpv1.RunAgentRequest],
	stream *connect.ServerStream[mcpv1.RunAgentResponse],
) error {
	agentName := req.Msg.Name

	s.mu.RLock()
	factory, exists := s.agents[agentName]
	s.mu.RUnlock()

	if !exists {
		return connect.NewError(connect.CodeNotFound, fmt.Errorf("agent not found: %s", agentName))
	}

	// Send start event
	if err := stream.Send(&mcpv1.RunAgentResponse{
		EventType: "start",
		DataJson:  fmt.Sprintf(`{"agent":"%s"}`, agentName),
	}); err != nil {
		return err
	}

	agent := factory()

	var contextMap map[string]interface{}
	if req.Msg.ContextJson != "" {
		json.Unmarshal([]byte(req.Msg.ContextJson), &contextMap)
	}

	input := &RunInput{
		Prompt:      req.Msg.Prompt,
		Context:     contextMap,
		Synchronous: true,
	}

	output, err := agent.Run(ctx, input)
	if err != nil {
		errorJSON, _ := json.Marshal(map[string]string{"error": err.Error()})
		stream.Send(&mcpv1.RunAgentResponse{
			EventType: "error",
			DataJson:  string(errorJSON),
		})
		return nil
	}

	// Stream steps
	for _, step := range output.Steps {
		stepJSON, _ := json.Marshal(step)
		if err := stream.Send(&mcpv1.RunAgentResponse{
			EventType: "step",
			DataJson:  string(stepJSON),
		}); err != nil {
			return err
		}
	}

	// Send result
	resultJSON, _ := json.Marshal(output)
	if err := stream.Send(&mcpv1.RunAgentResponse{
		EventType: "result",
		DataJson:  string(resultJSON),
	}); err != nil {
		return err
	}

	// Send end event
	return stream.Send(&mcpv1.RunAgentResponse{
		EventType: "end",
		DataJson:  "{}",
	})
}

// RunAgentSync implements MCPServiceHandler (synchronous)
func (s *MCPConnectServer) RunAgentSync(
	ctx context.Context,
	req *connect.Request[mcpv1.RunAgentRequest],
) (*connect.Response[mcpv1.RunAgentSyncResponse], error) {
	agentName := req.Msg.Name

	s.mu.RLock()
	factory, exists := s.agents[agentName]
	s.mu.RUnlock()

	if !exists {
		return nil, connect.NewError(connect.CodeNotFound, fmt.Errorf("agent not found: %s", agentName))
	}

	agent := factory()

	var contextMap map[string]interface{}
	if req.Msg.ContextJson != "" {
		json.Unmarshal([]byte(req.Msg.ContextJson), &contextMap)
	}

	input := &RunInput{
		Prompt:      req.Msg.Prompt,
		Context:     contextMap,
		Synchronous: true,
	}

	output, err := agent.Run(ctx, input)
	if err != nil {
		return connect.NewResponse(&mcpv1.RunAgentSyncResponse{
			Status: "error",
			Error:  err.Error(),
		}), nil
	}

	steps := make([]*mcpv1.AgentStep, len(output.Steps))
	for i, step := range output.Steps {
		inputJSON, _ := json.Marshal(step.Input)
		outputJSON, _ := json.Marshal(step.Output)
		steps[i] = &mcpv1.AgentStep{
			Type:       string(step.Type),
			Name:       step.ToolName,
			InputJson:  string(inputJSON),
			OutputJson: string(outputJSON),
			Timestamp:  step.StartedAt.UnixMilli(),
		}
	}

	resp := &mcpv1.RunAgentSyncResponse{
		WorkflowId: output.WorkflowID,
		Status:     string(output.Status),
		Result:     output.Result,
		Steps:      steps,
	}

	return connect.NewResponse(resp), nil
}

func findSlashIndex(s string) int {
	for i, c := range s {
		if c == '/' {
			return i
		}
	}
	return -1
}
