package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// MCPClient is a client for Model Context Protocol servers
type MCPClient struct {
	transport  MCPTransport
	tools      map[string]*MCPToolDef
	prompts    map[string]*MCPPromptDef
	resources  map[string]*MCPResourceDef
	mu         sync.RWMutex
	persistent bool
}

// MCPTransport defines the interface for MCP transports
type MCPTransport interface {
	// Connect establishes a connection
	Connect(ctx context.Context) error

	// Close closes the connection
	Close() error

	// Send sends a request and returns the response
	Send(ctx context.Context, method string, params interface{}) (json.RawMessage, error)

	// IsConnected returns true if connected
	IsConnected() bool
}

// MCPToolDef represents an MCP tool definition
type MCPToolDef struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

// MCPPromptDef represents an MCP prompt definition
type MCPPromptDef struct {
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Arguments   []MCPArgument `json:"arguments"`
}

// MCPResourceDef represents an MCP resource definition
type MCPResourceDef struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description"`
	MimeType    string `json:"mimeType"`
}

// MCPClientConfig configures the MCP client
type MCPClientConfig struct {
	TransportType string            // stdio, sse, http, websocket
	Command       string            // for stdio
	Args          []string          // for stdio
	URL           string            // for sse, http, websocket
	Headers       map[string]string // for http-based transports
	Persistent    bool              // keep connection open
	AllowedTools  []string          // filter tools (nil = all)
}

// NewMCPClient creates a new MCP client
func NewMCPClient(config *MCPClientConfig) (*MCPClient, error) {
	var transport MCPTransport
	var err error

	switch config.TransportType {
	case "stdio":
		transport = NewSTDIOTransport(config.Command, config.Args)
	case "sse":
		transport = NewSSETransport(config.URL, config.Headers)
	case "http":
		transport = NewHTTPTransport(config.URL, config.Headers)
	case "websocket":
		transport = NewWebSocketTransport(config.URL, config.Headers)
	default:
		return nil, fmt.Errorf("unsupported transport type: %s", config.TransportType)
	}

	client := &MCPClient{
		transport:  transport,
		tools:      make(map[string]*MCPToolDef),
		prompts:    make(map[string]*MCPPromptDef),
		resources:  make(map[string]*MCPResourceDef),
		persistent: config.Persistent,
	}

	// Connect if persistent
	if config.Persistent {
		if err = transport.Connect(context.Background()); err != nil {
			return nil, fmt.Errorf("failed to connect: %w", err)
		}

		// Load tools, prompts, resources
		if err = client.loadCapabilities(context.Background()); err != nil {
			transport.Close()
			return nil, fmt.Errorf("failed to load capabilities: %w", err)
		}
	}

	return client, nil
}

func (c *MCPClient) loadCapabilities(ctx context.Context) error {
	// Load tools
	if err := c.loadTools(ctx); err != nil {
		return err
	}

	// Load prompts
	if err := c.loadPrompts(ctx); err != nil {
		return err
	}

	// Load resources
	if err := c.loadResources(ctx); err != nil {
		return err
	}

	return nil
}

func (c *MCPClient) loadTools(ctx context.Context) error {
	resp, err := c.transport.Send(ctx, "tools/list", nil)
	if err != nil {
		return err
	}

	var result struct {
		Tools []MCPToolDef `json:"tools"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	for i := range result.Tools {
		c.tools[result.Tools[i].Name] = &result.Tools[i]
	}

	return nil
}

func (c *MCPClient) loadPrompts(ctx context.Context) error {
	resp, err := c.transport.Send(ctx, "prompts/list", nil)
	if err != nil {
		return err
	}

	var result struct {
		Prompts []MCPPromptDef `json:"prompts"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	for i := range result.Prompts {
		c.prompts[result.Prompts[i].Name] = &result.Prompts[i]
	}

	return nil
}

func (c *MCPClient) loadResources(ctx context.Context) error {
	resp, err := c.transport.Send(ctx, "resources/list", nil)
	if err != nil {
		return err
	}

	var result struct {
		Resources []MCPResourceDef `json:"resources"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	for i := range result.Resources {
		c.resources[result.Resources[i].URI] = &result.Resources[i]
	}

	return nil
}

// GetTools returns all available tools
func (c *MCPClient) GetTools() []*MCPToolDef {
	c.mu.RLock()
	defer c.mu.RUnlock()

	tools := make([]*MCPToolDef, 0, len(c.tools))
	for _, t := range c.tools {
		tools = append(tools, t)
	}
	return tools
}

// CallTool executes a tool
func (c *MCPClient) CallTool(ctx context.Context, name string, args map[string]interface{}) (interface{}, error) {
	if !c.persistent {
		if err := c.transport.Connect(ctx); err != nil {
			return nil, err
		}
		defer c.transport.Close()
	}

	params := map[string]interface{}{
		"name":      name,
		"arguments": args,
	}

	resp, err := c.transport.Send(ctx, "tools/call", params)
	if err != nil {
		return nil, err
	}

	var result struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, err
	}

	if result.IsError {
		if len(result.Content) > 0 {
			return nil, fmt.Errorf("tool error: %s", result.Content[0].Text)
		}
		return nil, fmt.Errorf("tool error")
	}

	if len(result.Content) > 0 {
		return result.Content[0].Text, nil
	}

	return nil, nil
}

// WrapMCPTool wraps an MCP tool as an AgentTool
func (c *MCPClient) WrapMCPTool(toolDef *MCPToolDef) Tool {
	return &mcpToolWrapper{
		client:  c,
		toolDef: toolDef,
	}
}

// WrapAllTools wraps all MCP tools as AgentTools
func (c *MCPClient) WrapAllTools() []Tool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	tools := make([]Tool, 0, len(c.tools))
	for _, t := range c.tools {
		tools = append(tools, c.WrapMCPTool(t))
	}
	return tools
}

// Close closes the client
func (c *MCPClient) Close() error {
	return c.transport.Close()
}

// mcpToolWrapper wraps an MCP tool as an AgentTool
type mcpToolWrapper struct {
	client  *MCPClient
	toolDef *MCPToolDef
}

func (w *mcpToolWrapper) Name() string {
	return w.toolDef.Name
}

func (w *mcpToolWrapper) Description() string {
	return w.toolDef.Description
}

func (w *mcpToolWrapper) Schema() *ToolSchema {
	schema := &ToolSchema{
		Type:       "object",
		Properties: make(map[string]PropertySchema),
	}

	if props, ok := w.toolDef.InputSchema["properties"].(map[string]interface{}); ok {
		for name, propDef := range props {
			if propMap, ok := propDef.(map[string]interface{}); ok {
				propType := "string"
				if t, ok := propMap["type"].(string); ok {
					propType = t
				}
				desc := ""
				if d, ok := propMap["description"].(string); ok {
					desc = d
				}
				schema.Properties[name] = PropertySchema{
					Type:        propType,
					Description: desc,
				}
			}
		}
	}

	if required, ok := w.toolDef.InputSchema["required"].([]interface{}); ok {
		for _, r := range required {
			if s, ok := r.(string); ok {
				schema.Required = append(schema.Required, s)
			}
		}
	}

	return schema
}

func (w *mcpToolWrapper) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	return w.client.CallTool(ctx, w.toolDef.Name, params)
}

// STDIOTransport implements MCP transport over stdio
type STDIOTransport struct {
	command   string
	args      []string
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	stdout    *bufio.Reader
	connected bool
	mu        sync.Mutex
	reqID     int
}

// NewSTDIOTransport creates a new STDIO transport
func NewSTDIOTransport(command string, args []string) *STDIOTransport {
	return &STDIOTransport{
		command: command,
		args:    args,
	}
}

func (t *STDIOTransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.connected {
		return nil
	}

	t.cmd = exec.CommandContext(ctx, t.command, t.args...)
	t.cmd.Stderr = os.Stderr

	var err error
	t.stdin, err = t.cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to get stdin pipe: %w", err)
	}

	stdout, err := t.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to get stdout pipe: %w", err)
	}
	t.stdout = bufio.NewReader(stdout)

	if err := t.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start process: %w", err)
	}

	t.connected = true

	// Send initialize request
	_, err = t.Send(ctx, "initialize", map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]interface{}{},
		"clientInfo": map[string]string{
			"name":    "dapr-agents-go",
			"version": "1.0.0",
		},
	})
	if err != nil {
		t.Close()
		return fmt.Errorf("failed to initialize: %w", err)
	}

	return nil
}

func (t *STDIOTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.connected {
		return nil
	}

	t.connected = false

	if t.stdin != nil {
		t.stdin.Close()
	}

	if t.cmd != nil && t.cmd.Process != nil {
		t.cmd.Process.Kill()
		t.cmd.Wait()
	}

	return nil
}

func (t *STDIOTransport) IsConnected() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.connected
}

func (t *STDIOTransport) Send(ctx context.Context, method string, params interface{}) (json.RawMessage, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.connected {
		return nil, fmt.Errorf("not connected")
	}

	t.reqID++
	req := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      t.reqID,
		"method":  method,
	}
	if params != nil {
		req["params"] = params
	}

	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	// Write request
	if _, err := fmt.Fprintf(t.stdin, "%s\n", data); err != nil {
		return nil, fmt.Errorf("failed to write request: %w", err)
	}

	// Read response
	line, err := t.stdout.ReadBytes('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	var resp struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      int             `json:"id"`
		Result  json.RawMessage `json:"result"`
		Error   *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(line, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("MCP error %d: %s", resp.Error.Code, resp.Error.Message)
	}

	return resp.Result, nil
}

// SSETransport implements MCP transport over Server-Sent Events
type SSETransport struct {
	url       string
	headers   map[string]string
	client    *http.Client
	connected bool
	mu        sync.Mutex
}

// NewSSETransport creates a new SSE transport
func NewSSETransport(url string, headers map[string]string) *SSETransport {
	return &SSETransport{
		url:     url,
		headers: headers,
		client:  &http.Client{Timeout: 60 * time.Second},
	}
}

func (t *SSETransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.connected = true
	return nil
}

func (t *SSETransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.connected = false
	return nil
}

func (t *SSETransport) IsConnected() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.connected
}

func (t *SSETransport) Send(ctx context.Context, method string, params interface{}) (json.RawMessage, error) {
	req := map[string]interface{}{
		"method": method,
		"params": params,
	}

	data, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", t.url, nil)
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	for k, v := range t.headers {
		httpReq.Header.Set(k, v)
	}

	// For SSE, we need to send the request body and then read the event stream
	httpReq.Body = io.NopCloser(NewBytesReader(data))

	resp, err := t.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Read SSE events until we get a result
	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}

		if len(line) > 6 && line[:6] == "data: " {
			eventData := line[6:]
			var result json.RawMessage
			if err := json.Unmarshal([]byte(eventData), &result); err == nil {
				return result, nil
			}
		}
	}

	return nil, fmt.Errorf("no result received")
}

// BytesReader is a simple bytes reader
type BytesReader struct {
	data []byte
	pos  int
}

// NewBytesReader creates a new bytes reader
func NewBytesReader(data []byte) *BytesReader {
	return &BytesReader{data: data}
}

func (r *BytesReader) Read(p []byte) (n int, err error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	n = copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

// HTTPTransport implements MCP transport over HTTP
type HTTPTransport struct {
	url       string
	headers   map[string]string
	client    *http.Client
	connected bool
	mu        sync.Mutex
}

// NewHTTPTransport creates a new HTTP transport
func NewHTTPTransport(url string, headers map[string]string) *HTTPTransport {
	return &HTTPTransport{
		url:     url,
		headers: headers,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

func (t *HTTPTransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.connected = true
	return nil
}

func (t *HTTPTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.connected = false
	return nil
}

func (t *HTTPTransport) IsConnected() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.connected
}

func (t *HTTPTransport) Send(ctx context.Context, method string, params interface{}) (json.RawMessage, error) {
	endpoint := t.url + "/" + method

	var body io.Reader
	if params != nil {
		data, err := json.Marshal(params)
		if err != nil {
			return nil, err
		}
		body = NewBytesReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, body)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("HTTP error: %d", resp.StatusCode)
	}

	var result json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}

// WebSocketTransport implements MCP transport over WebSocket
type WebSocketTransport struct {
	url       string
	headers   map[string]string
	conn      *websocket.Conn
	connected bool
	mu        sync.Mutex
	reqID     int
}

// NewWebSocketTransport creates a new WebSocket transport
func NewWebSocketTransport(url string, headers map[string]string) *WebSocketTransport {
	return &WebSocketTransport{
		url:     url,
		headers: headers,
	}
}

func (t *WebSocketTransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.connected {
		return nil
	}

	header := http.Header{}
	for k, v := range t.headers {
		header.Set(k, v)
	}

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.DialContext(ctx, t.url, header)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}

	t.conn = conn
	t.connected = true

	// Send initialize
	_, err = t.sendLocked(ctx, "initialize", map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]interface{}{},
		"clientInfo": map[string]string{
			"name":    "dapr-agents-go",
			"version": "1.0.0",
		},
	})
	if err != nil {
		t.conn.Close()
		t.connected = false
		return fmt.Errorf("failed to initialize: %w", err)
	}

	return nil
}

func (t *WebSocketTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.connected {
		return nil
	}

	t.connected = false
	if t.conn != nil {
		return t.conn.Close()
	}
	return nil
}

func (t *WebSocketTransport) IsConnected() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.connected
}

func (t *WebSocketTransport) Send(ctx context.Context, method string, params interface{}) (json.RawMessage, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.sendLocked(ctx, method, params)
}

func (t *WebSocketTransport) sendLocked(ctx context.Context, method string, params interface{}) (json.RawMessage, error) {
	if !t.connected || t.conn == nil {
		return nil, fmt.Errorf("not connected")
	}

	t.reqID++
	req := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      t.reqID,
		"method":  method,
	}
	if params != nil {
		req["params"] = params
	}

	if err := t.conn.WriteJSON(req); err != nil {
		return nil, fmt.Errorf("failed to send: %w", err)
	}

	var resp struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      int             `json:"id"`
		Result  json.RawMessage `json:"result"`
		Error   *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := t.conn.ReadJSON(&resp); err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("MCP error %d: %s", resp.Error.Code, resp.Error.Message)
	}

	return resp.Result, nil
}
