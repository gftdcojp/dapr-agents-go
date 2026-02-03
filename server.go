package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/dapr/go-sdk/actor"
	daprd "github.com/dapr/go-sdk/service/grpc"
	daprdhttp "github.com/dapr/go-sdk/service/http"
)

// Server is the main entry point for running Dapr Agents
type Server struct {
	config      *ServerConfig
	agents      map[string]func() actor.ServerContext
	httpHandler *HTTPHandler
	daprService interface{ Start() error }
}

// ServerConfig configures the agent server
type ServerConfig struct {
	// Port is the application port (default: 50051 for gRPC, 8080 for HTTP)
	Port string

	// Protocol is the server protocol (grpc or http)
	Protocol string

	// EnableHTTPEndpoints enables REST API endpoints for agents
	EnableHTTPEndpoints bool

	// HTTPPort is the port for HTTP endpoints (if different from main port)
	HTTPPort string

	// GracefulShutdownTimeout is the timeout for graceful shutdown
	GracefulShutdownTimeout time.Duration

	// Logger is the logger to use
	Logger *log.Logger
}

// DefaultServerConfig returns sensible defaults
func DefaultServerConfig() *ServerConfig {
	return &ServerConfig{
		Port:                    "50051",
		Protocol:                "grpc",
		EnableHTTPEndpoints:     true,
		HTTPPort:                "8080",
		GracefulShutdownTimeout: 30 * time.Second,
		Logger:                  log.Default(),
	}
}

// NewServer creates a new agent server
func NewServer(config *ServerConfig) *Server {
	if config == nil {
		config = DefaultServerConfig()
	}

	// Override port from environment if set
	if port := os.Getenv("APP_PORT"); port != "" {
		config.Port = port
	}
	if port := os.Getenv("HTTP_PORT"); port != "" {
		config.HTTPPort = port
	}

	return &Server{
		config: config,
		agents: make(map[string]func() actor.ServerContext),
	}
}

// RegisterAgent registers an agent factory with the server
func (s *Server) RegisterAgent(factory func() actor.ServerContext) {
	agent := factory()
	agentType := agent.Type()
	s.agents[agentType] = factory
	s.config.Logger.Printf("Registered agent: %s", agentType)
}

// RegisterAgentWithConfig creates and registers an agent with the given config
func (s *Server) RegisterAgentWithConfig(config *AgentConfig) {
	s.RegisterAgent(func() actor.ServerContext {
		return NewBaseAgent(config)
	})
}

// Start starts the agent server
func (s *Server) Start() error {
	var err error

	switch s.config.Protocol {
	case "grpc":
		err = s.startGRPC()
	case "http":
		err = s.startHTTP()
	default:
		return fmt.Errorf("unsupported protocol: %s", s.config.Protocol)
	}

	return err
}

func (s *Server) startGRPC() error {
	server, err := daprd.NewService(":" + s.config.Port)
	if err != nil {
		return fmt.Errorf("failed to create gRPC service: %w", err)
	}

	// Register all agents
	for _, factory := range s.agents {
		server.RegisterActorImplFactoryContext(factory)
	}

	s.daprService = server
	s.config.Logger.Printf("Agent server starting on port %s (gRPC)", s.config.Port)

	// Start HTTP endpoints in a separate goroutine if enabled
	if s.config.EnableHTTPEndpoints {
		go s.startHTTPEndpoints()
	}

	// Setup graceful shutdown
	go s.setupGracefulShutdown()

	return server.Start()
}

func (s *Server) startHTTP() error {
	server := daprdhttp.NewService(":" + s.config.Port)

	// Register all agents
	for _, factory := range s.agents {
		server.RegisterActorImplFactoryContext(factory)
	}

	s.daprService = server
	s.config.Logger.Printf("Agent server starting on port %s (HTTP)", s.config.Port)

	// Setup graceful shutdown
	go s.setupGracefulShutdown()

	return server.Start()
}

func (s *Server) startHTTPEndpoints() {
	s.httpHandler = NewHTTPHandler(s.agents, s.config.Logger)

	mux := http.NewServeMux()

	// Agent endpoints
	mux.HandleFunc("/run", s.httpHandler.HandleRun)
	mux.HandleFunc("/run/", s.httpHandler.HandleGetStatus)
	mux.HandleFunc("/agents", s.httpHandler.HandleListAgents)
	mux.HandleFunc("/health", s.httpHandler.HandleHealth)
	mux.HandleFunc("/healthz", s.httpHandler.HandleHealth)
	mux.HandleFunc("/readyz", s.httpHandler.HandleHealth)

	httpServer := &http.Server{
		Addr:         ":" + s.config.HTTPPort,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
	}

	s.config.Logger.Printf("HTTP endpoints starting on port %s", s.config.HTTPPort)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		s.config.Logger.Printf("HTTP server error: %v", err)
	}
}

func (s *Server) setupGracefulShutdown() {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	s.config.Logger.Println("Shutting down gracefully...")

	// Create a deadline for shutdown
	ctx, cancel := context.WithTimeout(context.Background(), s.config.GracefulShutdownTimeout)
	defer cancel()

	// Wait for context deadline or signal again
	select {
	case <-ctx.Done():
		s.config.Logger.Println("Graceful shutdown completed")
	case <-sigCh:
		s.config.Logger.Println("Forced shutdown")
	}

	os.Exit(0)
}

// HTTPHandler handles HTTP requests for agents
type HTTPHandler struct {
	agents map[string]func() actor.ServerContext
	logger *log.Logger
}

// NewHTTPHandler creates a new HTTP handler
func NewHTTPHandler(agents map[string]func() actor.ServerContext, logger *log.Logger) *HTTPHandler {
	return &HTTPHandler{
		agents: agents,
		logger: logger,
	}
}

// HandleRun handles POST /run requests to execute an agent
func (h *HTTPHandler) HandleRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		AgentType   string                 `json:"agentType"`
		AgentID     string                 `json:"agentId,omitempty"`
		Prompt      string                 `json:"prompt"`
		Context     map[string]interface{} `json:"context,omitempty"`
		MaxSteps    int                    `json:"maxSteps,omitempty"`
		Timeout     string                 `json:"timeout,omitempty"`
		Synchronous bool                   `json:"synchronous,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate agent type exists
	_, ok := h.agents[req.AgentType]
	if !ok {
		http.Error(w, fmt.Sprintf("Unknown agent type: %s", req.AgentType), http.StatusNotFound)
		return
	}

	// Parse timeout if provided
	var timeout time.Duration
	if req.Timeout != "" {
		var err error
		timeout, err = time.ParseDuration(req.Timeout)
		if err != nil {
			http.Error(w, fmt.Sprintf("Invalid timeout: %v", err), http.StatusBadRequest)
			return
		}
	}

	// Generate workflow ID
	workflowID := fmt.Sprintf("%s-%s-%d", req.AgentType, req.AgentID, time.Now().UnixNano())

	// For now, return the workflow ID immediately
	// In a full implementation, this would start the workflow via Dapr Workflow
	resp := map[string]interface{}{
		"workflowId": workflowID,
		"status":     "pending",
		"agentType":  req.AgentType,
	}

	if req.Synchronous {
		resp["status"] = "running"
		// TODO: Execute synchronously and return result
	}

	_ = timeout // Will be used in full implementation

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// HandleGetStatus handles GET /run/{workflowId} requests
func (h *HTTPHandler) HandleGetStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract workflow ID from path
	workflowID := r.URL.Path[len("/run/"):]
	if workflowID == "" {
		http.Error(w, "Workflow ID required", http.StatusBadRequest)
		return
	}

	// TODO: Get status from Dapr state/workflow
	resp := map[string]interface{}{
		"workflowId": workflowID,
		"status":     "unknown",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// HandleListAgents handles GET /agents requests
func (h *HTTPHandler) HandleListAgents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	agents := make([]map[string]interface{}, 0, len(h.agents))
	for name := range h.agents {
		agents = append(agents, map[string]interface{}{
			"type": name,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"agents": agents,
	})
}

// HandleHealth handles health check requests
func (h *HTTPHandler) HandleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
	})
}

// RunAgent is a convenience function to run a single agent
func RunAgent(config *AgentConfig, tools ...Tool) error {
	server := NewServer(nil)

	agent := NewBaseAgent(config)
	for _, tool := range tools {
		agent.RegisterTool(tool)
	}

	server.RegisterAgent(func() actor.ServerContext {
		a := NewBaseAgent(config)
		for _, tool := range tools {
			a.RegisterTool(tool)
		}
		return a
	})

	return server.Start()
}

// RunAgents is a convenience function to run multiple agents
func RunAgents(configs ...*AgentConfig) error {
	server := NewServer(nil)

	for _, config := range configs {
		server.RegisterAgentWithConfig(config)
	}

	return server.Start()
}
