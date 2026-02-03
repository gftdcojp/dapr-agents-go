package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// MCPServer implements Model Context Protocol server for exposing agents
// See: https://docs.dapr.io/developing-ai/mcp/
type MCPServer struct {
	config   *MCPServerConfig
	agents   map[string]func() Agent
	tools    map[string]Tool
	server   *http.Server
	logger   *log.Logger
	mu       sync.RWMutex
}

// MCPServerConfig configures the MCP server
type MCPServerConfig struct {
	Port         int
	Name         string
	Version      string
	Description  string
	AuthType     string            // none, apikey, oauth2
	APIKey       string
	AllowOrigins []string
	Logger       *log.Logger
}

// DefaultMCPServerConfig returns sensible defaults
func DefaultMCPServerConfig() *MCPServerConfig {
	return &MCPServerConfig{
		Port:         8081,
		Name:         "dapr-agent-mcp",
		Version:      "1.0.0",
		Description:  "Dapr Agent MCP Server",
		AuthType:     "none",
		AllowOrigins: []string{"*"},
		Logger:       log.Default(),
	}
}

// NewMCPServer creates a new MCP server
func NewMCPServer(config *MCPServerConfig) *MCPServer {
	if config == nil {
		config = DefaultMCPServerConfig()
	}
	if config.Logger == nil {
		config.Logger = log.Default()
	}

	return &MCPServer{
		config: config,
		agents: make(map[string]func() Agent),
		tools:  make(map[string]Tool),
		logger: config.Logger,
	}
}

// RegisterAgent registers an agent to be exposed via MCP
func (s *MCPServer) RegisterAgent(name string, factory func() Agent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.agents[name] = factory
	s.logger.Printf("MCP: Registered agent: %s", name)
}

// RegisterTool registers a tool to be exposed via MCP
func (s *MCPServer) RegisterTool(tool Tool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tools[tool.Name()] = tool
	s.logger.Printf("MCP: Registered tool: %s", tool.Name())
}

// Start starts the MCP server
func (s *MCPServer) Start() error {
	mux := http.NewServeMux()

	// MCP endpoints
	mux.HandleFunc("/mcp/v1/info", s.withCORS(s.withAuth(s.handleInfo)))
	mux.HandleFunc("/mcp/v1/tools", s.withCORS(s.withAuth(s.handleListTools)))
	mux.HandleFunc("/mcp/v1/tools/", s.withCORS(s.withAuth(s.handleToolCall)))
	mux.HandleFunc("/mcp/v1/prompts", s.withCORS(s.withAuth(s.handleListPrompts)))
	mux.HandleFunc("/mcp/v1/prompts/", s.withCORS(s.withAuth(s.handleGetPrompt)))
	mux.HandleFunc("/mcp/v1/resources", s.withCORS(s.withAuth(s.handleListResources)))
	mux.HandleFunc("/mcp/v1/resources/", s.withCORS(s.withAuth(s.handleGetResource)))

	// Agent-specific endpoints
	mux.HandleFunc("/mcp/v1/agents", s.withCORS(s.withAuth(s.handleListAgents)))
	mux.HandleFunc("/mcp/v1/agents/", s.withCORS(s.withAuth(s.handleAgentRun)))

	// Health check
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/healthz", s.handleHealth)

	// SSE endpoint for streaming
	mux.HandleFunc("/mcp/v1/sse", s.withCORS(s.withAuth(s.handleSSE)))

	s.server = &http.Server{
		Addr:         fmt.Sprintf(":%d", s.config.Port),
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second, // Longer for streaming
	}

	s.logger.Printf("MCP server starting on port %d", s.config.Port)
	return s.server.ListenAndServe()
}

// Stop stops the MCP server
func (s *MCPServer) Stop(ctx context.Context) error {
	if s.server != nil {
		return s.server.Shutdown(ctx)
	}
	return nil
}

// withCORS adds CORS headers
func (s *MCPServer) withCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
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
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
			w.Header().Set("Access-Control-Max-Age", "86400")
		}

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next(w, r)
	}
}

// withAuth adds authentication middleware
func (s *MCPServer) withAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch s.config.AuthType {
		case "apikey":
			apiKey := r.Header.Get("X-API-Key")
			if apiKey == "" {
				apiKey = r.URL.Query().Get("api_key")
			}
			if apiKey != s.config.APIKey {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}
		case "oauth2":
			// TODO: Implement OAuth2 validation
			auth := r.Header.Get("Authorization")
			if !strings.HasPrefix(auth, "Bearer ") {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}
		}
		next(w, r)
	}
}

// MCPServerInfo represents server information
type MCPServerInfo struct {
	Name         string            `json:"name"`
	Version      string            `json:"version"`
	Description  string            `json:"description,omitempty"`
	Capabilities []string          `json:"capabilities"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

func (s *MCPServer) handleInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	info := MCPServerInfo{
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
			"provider": "dapr-agent-sdk",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(info)
}

// MCPTool represents a tool in MCP format
type MCPTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

func (s *MCPServer) handleListTools(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	tools := make([]MCPTool, 0, len(s.tools))
	for _, tool := range s.tools {
		schema := tool.Schema()
		inputSchema := map[string]interface{}{
			"type":       schema.Type,
			"properties": schema.Properties,
		}
		if len(schema.Required) > 0 {
			inputSchema["required"] = schema.Required
		}

		tools = append(tools, MCPTool{
			Name:        tool.Name(),
			Description: tool.Description(),
			InputSchema: inputSchema,
		})
	}

	// Also expose agents as tools
	for name, factory := range s.agents {
		agent := factory()
		if true { // Agent interface provides GetTools()
			baseAgent := agent
			for _, tool := range baseAgent.GetTools() {
				schema := tool.Schema()
				inputSchema := map[string]interface{}{
					"type":       schema.Type,
					"properties": schema.Properties,
				}
				if len(schema.Required) > 0 {
					inputSchema["required"] = schema.Required
				}

				tools = append(tools, MCPTool{
					Name:        fmt.Sprintf("%s/%s", name, tool.Name()),
					Description: tool.Description(),
					InputSchema: inputSchema,
				})
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"tools": tools,
	})
}

// MCPToolCallRequest represents a tool call request
type MCPToolCallRequest struct {
	Arguments map[string]interface{} `json:"arguments"`
}

// MCPToolCallResponse represents a tool call response
type MCPToolCallResponse struct {
	Content []MCPContent `json:"content"`
	IsError bool         `json:"isError,omitempty"`
}

// MCPContent represents content in MCP format
type MCPContent struct {
	Type string `json:"type"` // text, image, resource
	Text string `json:"text,omitempty"`
	Data string `json:"data,omitempty"` // base64 for images
	URI  string `json:"uri,omitempty"`  // for resources
}

func (s *MCPServer) handleToolCall(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract tool name from path
	toolName := strings.TrimPrefix(r.URL.Path, "/mcp/v1/tools/")
	if toolName == "" {
		http.Error(w, "Tool name required", http.StatusBadRequest)
		return
	}

	var req MCPToolCallRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Check if it's an agent tool (format: agentName/toolName)
	parts := strings.SplitN(toolName, "/", 2)

	s.mu.RLock()
	var tool Tool
	var found bool

	if len(parts) == 2 {
		// Agent tool
		if factory, ok := s.agents[parts[0]]; ok {
			agent := factory()
			// Find tool by name from agent's tools
			for _, t := range agent.GetTools() {
				if t.Name() == parts[1] {
					tool = t
					found = true
					break
				}
			}
		}
	} else {
		// Direct tool
		tool, found = s.tools[toolName]
	}
	s.mu.RUnlock()

	if !found {
		http.Error(w, fmt.Sprintf("Tool not found: %s", toolName), http.StatusNotFound)
		return
	}

	// Execute tool
	ctx := r.Context()
	result, err := tool.Execute(ctx, req.Arguments)

	resp := MCPToolCallResponse{
		Content: []MCPContent{},
	}

	if err != nil {
		resp.IsError = true
		resp.Content = append(resp.Content, MCPContent{
			Type: "text",
			Text: fmt.Sprintf("Error: %s", err.Error()),
		})
	} else {
		// Convert result to text
		var text string
		switch v := result.(type) {
		case string:
			text = v
		default:
			data, _ := json.Marshal(result)
			text = string(data)
		}
		resp.Content = append(resp.Content, MCPContent{
			Type: "text",
			Text: text,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// MCPPrompt represents a prompt template
type MCPPrompt struct {
	Name        string       `json:"name"`
	Description string       `json:"description,omitempty"`
	Arguments   []MCPArgument `json:"arguments,omitempty"`
}

// MCPArgument represents a prompt argument
type MCPArgument struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required,omitempty"`
}

func (s *MCPServer) handleListPrompts(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Create prompts from agents
	prompts := make([]MCPPrompt, 0)
	for name := range s.agents {
		prompts = append(prompts, MCPPrompt{
			Name:        name,
			Description: fmt.Sprintf("Run %s agent with a prompt", name),
			Arguments: []MCPArgument{
				{Name: "prompt", Description: "The prompt to send to the agent", Required: true},
				{Name: "context", Description: "Additional context (JSON)", Required: false},
			},
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"prompts": prompts,
	})
}

func (s *MCPServer) handleGetPrompt(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	promptName := strings.TrimPrefix(r.URL.Path, "/mcp/v1/prompts/")
	if promptName == "" {
		http.Error(w, "Prompt name required", http.StatusBadRequest)
		return
	}

	s.mu.RLock()
	_, exists := s.agents[promptName]
	s.mu.RUnlock()

	if !exists {
		http.Error(w, fmt.Sprintf("Prompt not found: %s", promptName), http.StatusNotFound)
		return
	}

	// Return the prompt template
	response := map[string]interface{}{
		"description": fmt.Sprintf("Run %s agent", promptName),
		"messages": []map[string]string{
			{"role": "user", "content": "{{prompt}}"},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// MCPResource represents a resource
type MCPResource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
}

func (s *MCPServer) handleListResources(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// For now, expose agents as resources
	s.mu.RLock()
	defer s.mu.RUnlock()

	resources := make([]MCPResource, 0)
	for name := range s.agents {
		resources = append(resources, MCPResource{
			URI:         fmt.Sprintf("agent://%s", name),
			Name:        name,
			Description: fmt.Sprintf("Agent: %s", name),
			MimeType:    "application/json",
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"resources": resources,
	})
}

func (s *MCPServer) handleGetResource(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	uri := strings.TrimPrefix(r.URL.Path, "/mcp/v1/resources/")
	if uri == "" {
		http.Error(w, "Resource URI required", http.StatusBadRequest)
		return
	}

	// Parse agent URI
	if strings.HasPrefix(uri, "agent://") {
		agentName := strings.TrimPrefix(uri, "agent://")
		s.mu.RLock()
		factory, exists := s.agents[agentName]
		s.mu.RUnlock()

		if !exists {
			http.Error(w, fmt.Sprintf("Agent not found: %s", agentName), http.StatusNotFound)
			return
		}

		agent := factory()
		if true { // Agent interface provides GetTools()
			baseAgent := agent
			tools := baseAgent.GetTools()
			toolNames := make([]string, len(tools))
			for i, t := range tools {
				toolNames[i] = t.Name()
			}

			response := map[string]interface{}{
				"contents": []map[string]interface{}{
					{
						"uri":      uri,
						"mimeType": "application/json",
						"text": map[string]interface{}{
							"name":  agentName,
							"type":  "agent",
							"tools": toolNames,
						},
					},
				},
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(response)
			return
		}
	}

	http.Error(w, fmt.Sprintf("Resource not found: %s", uri), http.StatusNotFound)
}

// MCPAgent represents an agent in MCP format
type MCPAgent struct {
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	Tools       []string `json:"tools,omitempty"`
}

func (s *MCPServer) handleListAgents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	agents := make([]MCPAgent, 0, len(s.agents))
	for name, factory := range s.agents {
		agent := factory()
		var tools []string
		if true { // Agent interface provides GetTools()
			baseAgent := agent
			for _, t := range baseAgent.GetTools() {
				tools = append(tools, t.Name())
			}
		}

		agents = append(agents, MCPAgent{
			Name:  name,
			Tools: tools,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"agents": agents,
	})
}

// MCPAgentRunRequest represents a request to run an agent
type MCPAgentRunRequest struct {
	Prompt      string                 `json:"prompt"`
	Context     map[string]interface{} `json:"context,omitempty"`
	Synchronous bool                   `json:"synchronous,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
}

func (s *MCPServer) handleAgentRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	agentName := strings.TrimPrefix(r.URL.Path, "/mcp/v1/agents/")
	if agentName == "" {
		http.Error(w, "Agent name required", http.StatusBadRequest)
		return
	}

	// Remove trailing /run if present
	agentName = strings.TrimSuffix(agentName, "/run")

	s.mu.RLock()
	factory, exists := s.agents[agentName]
	s.mu.RUnlock()

	if !exists {
		http.Error(w, fmt.Sprintf("Agent not found: %s", agentName), http.StatusNotFound)
		return
	}

	var req MCPAgentRunRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	agent := factory()
	if true { // Agent interface provides GetTools()
			baseAgent := agent
		input := &RunInput{
			Prompt:      req.Prompt,
			Context:     req.Context,
			Synchronous: req.Synchronous || !req.Stream,
		}

		output, err := baseAgent.Run(r.Context(), input)
		if err != nil {
			http.Error(w, fmt.Sprintf("Agent error: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(output)
		return
	}

	http.Error(w, "Agent does not support MCP interface", http.StatusInternalServerError)
}

func (s *MCPServer) handleSSE(w http.ResponseWriter, r *http.Request) {
	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	// Read the request body for tool call
	body, err := io.ReadAll(r.Body)
	if err != nil {
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
		flusher.Flush()
		return
	}

	var req struct {
		Type      string                 `json:"type"` // tool_call, agent_run
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
		Prompt    string                 `json:"prompt"`
	}

	if err := json.Unmarshal(body, &req); err != nil {
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
		flusher.Flush()
		return
	}

	// Send start event
	fmt.Fprintf(w, "event: start\ndata: {\"type\":\"%s\",\"name\":\"%s\"}\n\n", req.Type, req.Name)
	flusher.Flush()

	ctx := r.Context()

	switch req.Type {
	case "tool_call":
		s.mu.RLock()
		tool, found := s.tools[req.Name]
		s.mu.RUnlock()

		if !found {
			fmt.Fprintf(w, "event: error\ndata: Tool not found: %s\n\n", req.Name)
			flusher.Flush()
			return
		}

		result, err := tool.Execute(ctx, req.Arguments)
		if err != nil {
			fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
		} else {
			data, _ := json.Marshal(result)
			fmt.Fprintf(w, "event: result\ndata: %s\n\n", string(data))
		}
		flusher.Flush()

	case "agent_run":
		s.mu.RLock()
		factory, found := s.agents[req.Name]
		s.mu.RUnlock()

		if !found {
			fmt.Fprintf(w, "event: error\ndata: Agent not found: %s\n\n", req.Name)
			flusher.Flush()
			return
		}

		agent := factory()
		if true { // Agent interface provides GetTools()
			baseAgent := agent
			input := &RunInput{
				Prompt:      req.Prompt,
				Context:     req.Arguments,
				Synchronous: true,
			}

			output, err := baseAgent.Run(ctx, input)
			if err != nil {
				fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())
			} else {
				// Stream steps
				for _, step := range output.Steps {
					stepData, _ := json.Marshal(step)
					fmt.Fprintf(w, "event: step\ndata: %s\n\n", string(stepData))
					flusher.Flush()
				}
				// Send final result
				data, _ := json.Marshal(output)
				fmt.Fprintf(w, "event: result\ndata: %s\n\n", string(data))
			}
			flusher.Flush()
		}
	}

	// Send end event
	fmt.Fprintf(w, "event: end\ndata: {}\n\n")
	flusher.Flush()
}

func (s *MCPServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
	})
}
