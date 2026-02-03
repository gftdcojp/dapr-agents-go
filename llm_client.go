package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// LLMClient is a generic interface for LLM providers
type LLMClient interface {
	// Complete generates a completion for the given prompt
	Complete(ctx context.Context, request *CompletionRequest) (*CompletionResponse, error)

	// Chat generates a chat completion
	Chat(ctx context.Context, request *ChatRequest) (*ChatResponse, error)

	// Stream generates a streaming chat completion
	Stream(ctx context.Context, request *ChatRequest) (<-chan *ChatStreamEvent, error)

	// Embed generates embeddings for the given text
	Embed(ctx context.Context, text string) ([]float64, error)

	// Provider returns the provider name
	Provider() string

	// Models returns available models
	Models() []string
}

// CompletionRequest represents a completion request
type CompletionRequest struct {
	Prompt      string                 `json:"prompt"`
	Model       string                 `json:"model,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
	TopP        float64                `json:"top_p,omitempty"`
	Stop        []string               `json:"stop,omitempty"`
	Extra       map[string]interface{} `json:"extra,omitempty"`
}

// CompletionResponse represents a completion response
type CompletionResponse struct {
	Text         string                 `json:"text"`
	Model        string                 `json:"model"`
	FinishReason string                 `json:"finish_reason"`
	Usage        *TokenUsage            `json:"usage,omitempty"`
	Extra        map[string]interface{} `json:"extra,omitempty"`
}

// ChatRequest represents a chat completion request
type ChatRequest struct {
	Messages      []ChatMessage          `json:"messages"`
	Model         string                 `json:"model,omitempty"`
	MaxTokens     int                    `json:"max_tokens,omitempty"`
	Temperature   float64                `json:"temperature,omitempty"`
	TopP          float64                `json:"top_p,omitempty"`
	Stop          []string               `json:"stop,omitempty"`
	Tools         []*LLMToolSchema       `json:"tools,omitempty"`
	ToolChoice    interface{}            `json:"tool_choice,omitempty"`
	ResponseFormat interface{}           `json:"response_format,omitempty"`
	Extra         map[string]interface{} `json:"extra,omitempty"`
}

// ChatMessage represents a chat message
type ChatMessage struct {
	Role       string        `json:"role"`
	Content    interface{}   `json:"content"` // string or []ContentPart
	Name       string        `json:"name,omitempty"`
	ToolCalls  []LLMToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

// ContentPart represents a content part for multimodal messages
type ContentPart struct {
	Type     string    `json:"type"` // "text", "image_url", "image"
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL represents an image URL
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "low", "high", "auto"
}

// LLMToolCall represents a tool call from the LLM model (OpenAI format)
type LLMToolCall struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Function LLMFunctionCall `json:"function"`
}

// LLMFunctionCall represents a function call in LLM responses
type LLMFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// LLMToolSchema defines tool schema for LLM APIs (OpenAI format)
type LLMToolSchema struct {
	Type     string             `json:"type"` // "function"
	Function *LLMFunctionSchema `json:"function,omitempty"`
}

// LLMFunctionSchema defines function schema for LLM APIs
type LLMFunctionSchema struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ChatResponse represents a chat completion response
type ChatResponse struct {
	Message      ChatMessage            `json:"message"`
	Model        string                 `json:"model"`
	FinishReason string                 `json:"finish_reason"`
	Usage        *TokenUsage            `json:"usage,omitempty"`
	Extra        map[string]interface{} `json:"extra,omitempty"`
}

// TokenUsage represents token usage
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatStreamEvent represents a streaming event
type ChatStreamEvent struct {
	Delta        *ChatMessage `json:"delta,omitempty"`
	FinishReason string       `json:"finish_reason,omitempty"`
	Error        error        `json:"error,omitempty"`
	Done         bool         `json:"done"`
}

// OpenAIClient is an OpenAI API client
type OpenAIClient struct {
	apiKey       string
	baseURL      string
	organization string
	model        string
	httpClient   *http.Client
	mu           sync.Mutex
}

// OpenAIConfig configures the OpenAI client
type OpenAIConfig struct {
	APIKey       string
	BaseURL      string
	Organization string
	Model        string
	Timeout      time.Duration
}

// NewOpenAIClient creates a new OpenAI client
func NewOpenAIClient(config *OpenAIConfig) (*OpenAIClient, error) {
	if config == nil {
		config = &OpenAIConfig{}
	}

	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("OpenAI API key is required")
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	model := config.Model
	if model == "" {
		model = "gpt-4o"
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	return &OpenAIClient{
		apiKey:       apiKey,
		baseURL:      baseURL,
		organization: config.Organization,
		model:        model,
		httpClient:   &http.Client{Timeout: timeout},
	}, nil
}

// Complete generates a completion
func (c *OpenAIClient) Complete(ctx context.Context, request *CompletionRequest) (*CompletionResponse, error) {
	// Convert to chat format for newer models
	chatReq := &ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: request.Prompt},
		},
		Model:       request.Model,
		MaxTokens:   request.MaxTokens,
		Temperature: request.Temperature,
		TopP:        request.TopP,
		Stop:        request.Stop,
	}
	if chatReq.Model == "" {
		chatReq.Model = c.model
	}

	resp, err := c.Chat(ctx, chatReq)
	if err != nil {
		return nil, err
	}

	content := ""
	if s, ok := resp.Message.Content.(string); ok {
		content = s
	}

	return &CompletionResponse{
		Text:         content,
		Model:        resp.Model,
		FinishReason: resp.FinishReason,
		Usage:        resp.Usage,
	}, nil
}

// Chat generates a chat completion
func (c *OpenAIClient) Chat(ctx context.Context, request *ChatRequest) (*ChatResponse, error) {
	if request.Model == "" {
		request.Model = c.model
	}

	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	if c.organization != "" {
		req.Header.Set("OpenAI-Organization", c.organization)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OpenAI API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		ID      string `json:"id"`
		Choices []struct {
			Message      ChatMessage `json:"message"`
			FinishReason string      `json:"finish_reason"`
		} `json:"choices"`
		Model string      `json:"model"`
		Usage *TokenUsage `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	return &ChatResponse{
		Message:      result.Choices[0].Message,
		Model:        result.Model,
		FinishReason: result.Choices[0].FinishReason,
		Usage:        result.Usage,
	}, nil
}

// Stream generates a streaming chat completion
func (c *OpenAIClient) Stream(ctx context.Context, request *ChatRequest) (<-chan *ChatStreamEvent, error) {
	if request.Model == "" {
		request.Model = c.model
	}

	reqBody := map[string]interface{}{
		"model":    request.Model,
		"messages": request.Messages,
		"stream":   true,
	}
	if request.MaxTokens > 0 {
		reqBody["max_tokens"] = request.MaxTokens
	}
	if request.Temperature > 0 {
		reqBody["temperature"] = request.Temperature
	}
	if request.Tools != nil {
		reqBody["tools"] = request.Tools
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("OpenAI API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	events := make(chan *ChatStreamEvent)

	go func() {
		defer resp.Body.Close()
		defer close(events)

		reader := NewSSEReader(resp.Body)
		for {
			select {
			case <-ctx.Done():
				events <- &ChatStreamEvent{Error: ctx.Err(), Done: true}
				return
			default:
			}

			event, err := reader.Read()
			if err != nil {
				if err == io.EOF {
					events <- &ChatStreamEvent{Done: true}
					return
				}
				events <- &ChatStreamEvent{Error: err, Done: true}
				return
			}

			if event.Data == "[DONE]" {
				events <- &ChatStreamEvent{Done: true}
				return
			}

			var chunk struct {
				Choices []struct {
					Delta        ChatMessage `json:"delta"`
					FinishReason string      `json:"finish_reason"`
				} `json:"choices"`
			}

			if err := json.Unmarshal([]byte(event.Data), &chunk); err != nil {
				continue
			}

			if len(chunk.Choices) > 0 {
				events <- &ChatStreamEvent{
					Delta:        &chunk.Choices[0].Delta,
					FinishReason: chunk.Choices[0].FinishReason,
				}
			}
		}
	}()

	return events, nil
}

// Embed generates embeddings
func (c *OpenAIClient) Embed(ctx context.Context, text string) ([]float64, error) {
	body, err := json.Marshal(map[string]interface{}{
		"input": text,
		"model": "text-embedding-3-small",
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OpenAI API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Data) == 0 {
		return nil, fmt.Errorf("no embeddings in response")
	}

	return result.Data[0].Embedding, nil
}

// Provider returns the provider name
func (c *OpenAIClient) Provider() string {
	return "openai"
}

// Models returns available models
func (c *OpenAIClient) Models() []string {
	return []string{"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"}
}

// AnthropicClient is an Anthropic Claude API client
type AnthropicClient struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
}

// AnthropicConfig configures the Anthropic client
type AnthropicConfig struct {
	APIKey  string
	BaseURL string
	Model   string
	Timeout time.Duration
}

// NewAnthropicClient creates a new Anthropic client
func NewAnthropicClient(config *AnthropicConfig) (*AnthropicClient, error) {
	if config == nil {
		config = &AnthropicConfig{}
	}

	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("Anthropic API key is required")
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1"
	}

	model := config.Model
	if model == "" {
		model = "claude-sonnet-4-20250514"
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	return &AnthropicClient{
		apiKey:     apiKey,
		baseURL:    baseURL,
		model:      model,
		httpClient: &http.Client{Timeout: timeout},
	}, nil
}

// Complete generates a completion
func (c *AnthropicClient) Complete(ctx context.Context, request *CompletionRequest) (*CompletionResponse, error) {
	chatReq := &ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: request.Prompt},
		},
		Model:       request.Model,
		MaxTokens:   request.MaxTokens,
		Temperature: request.Temperature,
	}

	resp, err := c.Chat(ctx, chatReq)
	if err != nil {
		return nil, err
	}

	content := ""
	if s, ok := resp.Message.Content.(string); ok {
		content = s
	}

	return &CompletionResponse{
		Text:         content,
		Model:        resp.Model,
		FinishReason: resp.FinishReason,
		Usage:        resp.Usage,
	}, nil
}

// Chat generates a chat completion
func (c *AnthropicClient) Chat(ctx context.Context, request *ChatRequest) (*ChatResponse, error) {
	model := request.Model
	if model == "" {
		model = c.model
	}

	// Convert messages to Anthropic format
	var system string
	var messages []map[string]interface{}

	for _, msg := range request.Messages {
		if msg.Role == "system" {
			if s, ok := msg.Content.(string); ok {
				system = s
			}
			continue
		}

		role := msg.Role
		if role == "assistant" {
			role = "assistant"
		} else {
			role = "user"
		}

		messages = append(messages, map[string]interface{}{
			"role":    role,
			"content": msg.Content,
		})
	}

	reqBody := map[string]interface{}{
		"model":      model,
		"messages":   messages,
		"max_tokens": request.MaxTokens,
	}
	if system != "" {
		reqBody["system"] = system
	}
	if request.Temperature > 0 {
		reqBody["temperature"] = request.Temperature
	}
	if request.MaxTokens == 0 {
		reqBody["max_tokens"] = 4096
	}

	// Convert tools to Anthropic format
	if len(request.Tools) > 0 {
		var tools []map[string]interface{}
		for _, tool := range request.Tools {
			if tool.Function != nil {
				tools = append(tools, map[string]interface{}{
					"name":         tool.Function.Name,
					"description":  tool.Function.Description,
					"input_schema": tool.Function.Parameters,
				})
			}
		}
		reqBody["tools"] = tools
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Anthropic API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		ID         string `json:"id"`
		Type       string `json:"type"`
		Role       string `json:"role"`
		Content    []struct {
			Type  string `json:"type"`
			Text  string `json:"text,omitempty"`
			ID    string `json:"id,omitempty"`
			Name  string `json:"name,omitempty"`
			Input json.RawMessage `json:"input,omitempty"`
		} `json:"content"`
		Model      string `json:"model"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Build response message
	var textContent strings.Builder
	var toolCalls []LLMToolCall

	for _, content := range result.Content {
		switch content.Type {
		case "text":
			textContent.WriteString(content.Text)
		case "tool_use":
			toolCalls = append(toolCalls, LLMToolCall{
				ID:   content.ID,
				Type: "function",
				Function: LLMFunctionCall{
					Name:      content.Name,
					Arguments: string(content.Input),
				},
			})
		}
	}

	return &ChatResponse{
		Message: ChatMessage{
			Role:      result.Role,
			Content:   textContent.String(),
			ToolCalls: toolCalls,
		},
		Model:        result.Model,
		FinishReason: result.StopReason,
		Usage: &TokenUsage{
			PromptTokens:     result.Usage.InputTokens,
			CompletionTokens: result.Usage.OutputTokens,
			TotalTokens:      result.Usage.InputTokens + result.Usage.OutputTokens,
		},
	}, nil
}

// Stream generates a streaming chat completion
func (c *AnthropicClient) Stream(ctx context.Context, request *ChatRequest) (<-chan *ChatStreamEvent, error) {
	// Similar to Chat but with stream: true
	// Implementation would be similar to OpenAI streaming
	return nil, fmt.Errorf("streaming not implemented for Anthropic client")
}

// Embed generates embeddings (Anthropic doesn't have native embeddings)
func (c *AnthropicClient) Embed(ctx context.Context, text string) ([]float64, error) {
	return nil, fmt.Errorf("Anthropic does not provide embedding API")
}

// Provider returns the provider name
func (c *AnthropicClient) Provider() string {
	return "anthropic"
}

// Models returns available models
func (c *AnthropicClient) Models() []string {
	return []string{"claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"}
}

// AzureOpenAIClient is an Azure OpenAI client
type AzureOpenAIClient struct {
	apiKey     string
	endpoint   string
	deployment string
	apiVersion string
	httpClient *http.Client
}

// AzureOpenAIConfig configures the Azure OpenAI client
type AzureOpenAIConfig struct {
	APIKey     string
	Endpoint   string
	Deployment string
	APIVersion string
	Timeout    time.Duration
}

// NewAzureOpenAIClient creates a new Azure OpenAI client
func NewAzureOpenAIClient(config *AzureOpenAIConfig) (*AzureOpenAIClient, error) {
	if config == nil {
		config = &AzureOpenAIConfig{}
	}

	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("AZURE_OPENAI_API_KEY")
	}

	endpoint := config.Endpoint
	if endpoint == "" {
		endpoint = os.Getenv("AZURE_OPENAI_ENDPOINT")
	}

	if apiKey == "" || endpoint == "" {
		return nil, fmt.Errorf("Azure OpenAI API key and endpoint are required")
	}

	apiVersion := config.APIVersion
	if apiVersion == "" {
		apiVersion = "2024-02-15-preview"
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	return &AzureOpenAIClient{
		apiKey:     apiKey,
		endpoint:   strings.TrimSuffix(endpoint, "/"),
		deployment: config.Deployment,
		apiVersion: apiVersion,
		httpClient: &http.Client{Timeout: timeout},
	}, nil
}

// Complete generates a completion
func (c *AzureOpenAIClient) Complete(ctx context.Context, request *CompletionRequest) (*CompletionResponse, error) {
	chatReq := &ChatRequest{
		Messages: []ChatMessage{
			{Role: "user", Content: request.Prompt},
		},
		MaxTokens:   request.MaxTokens,
		Temperature: request.Temperature,
	}

	resp, err := c.Chat(ctx, chatReq)
	if err != nil {
		return nil, err
	}

	content := ""
	if s, ok := resp.Message.Content.(string); ok {
		content = s
	}

	return &CompletionResponse{
		Text:         content,
		Model:        resp.Model,
		FinishReason: resp.FinishReason,
		Usage:        resp.Usage,
	}, nil
}

// Chat generates a chat completion
func (c *AzureOpenAIClient) Chat(ctx context.Context, request *ChatRequest) (*ChatResponse, error) {
	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		c.endpoint, c.deployment, c.apiVersion)

	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Azure OpenAI API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		Choices []struct {
			Message      ChatMessage `json:"message"`
			FinishReason string      `json:"finish_reason"`
		} `json:"choices"`
		Model string      `json:"model"`
		Usage *TokenUsage `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	return &ChatResponse{
		Message:      result.Choices[0].Message,
		Model:        result.Model,
		FinishReason: result.Choices[0].FinishReason,
		Usage:        result.Usage,
	}, nil
}

// Stream generates a streaming chat completion
func (c *AzureOpenAIClient) Stream(ctx context.Context, request *ChatRequest) (<-chan *ChatStreamEvent, error) {
	return nil, fmt.Errorf("streaming not implemented for Azure OpenAI client")
}

// Embed generates embeddings
func (c *AzureOpenAIClient) Embed(ctx context.Context, text string) ([]float64, error) {
	url := fmt.Sprintf("%s/openai/deployments/%s/embeddings?api-version=%s",
		c.endpoint, c.deployment, c.apiVersion)

	body, err := json.Marshal(map[string]interface{}{
		"input": text,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Azure OpenAI API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Data) == 0 {
		return nil, fmt.Errorf("no embeddings in response")
	}

	return result.Data[0].Embedding, nil
}

// Provider returns the provider name
func (c *AzureOpenAIClient) Provider() string {
	return "azure-openai"
}

// Models returns available models (depends on deployment)
func (c *AzureOpenAIClient) Models() []string {
	return []string{c.deployment}
}

// OllamaClient is an Ollama API client
type OllamaClient struct {
	baseURL    string
	model      string
	httpClient *http.Client
}

// OllamaConfig configures the Ollama client
type OllamaConfig struct {
	BaseURL string
	Model   string
	Timeout time.Duration
}

// NewOllamaClient creates a new Ollama client
func NewOllamaClient(config *OllamaConfig) (*OllamaClient, error) {
	if config == nil {
		config = &OllamaConfig{}
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("OLLAMA_HOST")
	}
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	model := config.Model
	if model == "" {
		model = "llama3.2"
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 300 * time.Second // Ollama can be slow
	}

	return &OllamaClient{
		baseURL:    strings.TrimSuffix(baseURL, "/"),
		model:      model,
		httpClient: &http.Client{Timeout: timeout},
	}, nil
}

// Complete generates a completion
func (c *OllamaClient) Complete(ctx context.Context, request *CompletionRequest) (*CompletionResponse, error) {
	model := request.Model
	if model == "" {
		model = c.model
	}

	body, err := json.Marshal(map[string]interface{}{
		"model":  model,
		"prompt": request.Prompt,
		"stream": false,
		"options": map[string]interface{}{
			"temperature": request.Temperature,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Ollama API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		Response string `json:"response"`
		Model    string `json:"model"`
		Done     bool   `json:"done"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &CompletionResponse{
		Text:  result.Response,
		Model: result.Model,
	}, nil
}

// Chat generates a chat completion
func (c *OllamaClient) Chat(ctx context.Context, request *ChatRequest) (*ChatResponse, error) {
	model := request.Model
	if model == "" {
		model = c.model
	}

	// Convert messages to Ollama format
	var messages []map[string]interface{}
	for _, msg := range request.Messages {
		messages = append(messages, map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		})
	}

	reqBody := map[string]interface{}{
		"model":    model,
		"messages": messages,
		"stream":   false,
	}
	if request.Temperature > 0 {
		reqBody["options"] = map[string]interface{}{
			"temperature": request.Temperature,
		}
	}

	// Handle tools
	if len(request.Tools) > 0 {
		var tools []map[string]interface{}
		for _, tool := range request.Tools {
			if tool.Function != nil {
				tools = append(tools, map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name":        tool.Function.Name,
						"description": tool.Function.Description,
						"parameters":  tool.Function.Parameters,
					},
				})
			}
		}
		reqBody["tools"] = tools
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Ollama API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		Message struct {
			Role      string        `json:"role"`
			Content   string        `json:"content"`
			ToolCalls []LLMToolCall `json:"tool_calls,omitempty"`
		} `json:"message"`
		Model string `json:"model"`
		Done  bool   `json:"done"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &ChatResponse{
		Message: ChatMessage{
			Role:      result.Message.Role,
			Content:   result.Message.Content,
			ToolCalls: result.Message.ToolCalls,
		},
		Model: result.Model,
	}, nil
}

// Stream generates a streaming chat completion
func (c *OllamaClient) Stream(ctx context.Context, request *ChatRequest) (<-chan *ChatStreamEvent, error) {
	model := request.Model
	if model == "" {
		model = c.model
	}

	var messages []map[string]interface{}
	for _, msg := range request.Messages {
		messages = append(messages, map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		})
	}

	body, err := json.Marshal(map[string]interface{}{
		"model":    model,
		"messages": messages,
		"stream":   true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("Ollama API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	events := make(chan *ChatStreamEvent)

	go func() {
		defer resp.Body.Close()
		defer close(events)

		decoder := json.NewDecoder(resp.Body)
		for {
			select {
			case <-ctx.Done():
				events <- &ChatStreamEvent{Error: ctx.Err(), Done: true}
				return
			default:
			}

			var chunk struct {
				Message struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"message"`
				Done bool `json:"done"`
			}

			if err := decoder.Decode(&chunk); err != nil {
				if err == io.EOF {
					events <- &ChatStreamEvent{Done: true}
					return
				}
				events <- &ChatStreamEvent{Error: err, Done: true}
				return
			}

			events <- &ChatStreamEvent{
				Delta: &ChatMessage{
					Role:    chunk.Message.Role,
					Content: chunk.Message.Content,
				},
				Done: chunk.Done,
			}

			if chunk.Done {
				return
			}
		}
	}()

	return events, nil
}

// Embed generates embeddings
func (c *OllamaClient) Embed(ctx context.Context, text string) ([]float64, error) {
	body, err := json.Marshal(map[string]interface{}{
		"model":  c.model,
		"prompt": text,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Ollama API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		Embedding []float64 `json:"embedding"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Embedding, nil
}

// Provider returns the provider name
func (c *OllamaClient) Provider() string {
	return "ollama"
}

// Models returns available models
func (c *OllamaClient) Models() []string {
	// Could query /api/tags endpoint
	return []string{c.model}
}

// SSEReader reads Server-Sent Events
type SSEReader struct {
	reader io.Reader
	buf    []byte
}

// SSEEvent represents an SSE event
type SSEEvent struct {
	Event string
	Data  string
	ID    string
}

// NewSSEReader creates a new SSE reader
func NewSSEReader(reader io.Reader) *SSEReader {
	return &SSEReader{reader: reader}
}

// Read reads the next SSE event
func (r *SSEReader) Read() (*SSEEvent, error) {
	buf := make([]byte, 4096)
	var data strings.Builder

	for {
		n, err := r.reader.Read(buf)
		if err != nil {
			return nil, err
		}

		data.Write(buf[:n])
		content := data.String()

		// Check for complete event (ends with \n\n)
		if strings.Contains(content, "\n\n") {
			parts := strings.SplitN(content, "\n\n", 2)
			eventData := parts[0]

			// Parse event
			event := &SSEEvent{}
			for _, line := range strings.Split(eventData, "\n") {
				if strings.HasPrefix(line, "data: ") {
					event.Data = strings.TrimPrefix(line, "data: ")
				} else if strings.HasPrefix(line, "event: ") {
					event.Event = strings.TrimPrefix(line, "event: ")
				} else if strings.HasPrefix(line, "id: ") {
					event.ID = strings.TrimPrefix(line, "id: ")
				}
			}

			// Keep remaining data
			if len(parts) > 1 {
				data.Reset()
				data.WriteString(parts[1])
			} else {
				data.Reset()
			}

			return event, nil
		}
	}
}

// LLMClientFactory creates LLM clients
type LLMClientFactory struct{}

// NewLLMClient creates an LLM client based on provider
func NewLLMClient(provider string, config map[string]interface{}) (LLMClient, error) {
	switch strings.ToLower(provider) {
	case "openai":
		cfg := &OpenAIConfig{}
		if v, ok := config["api_key"].(string); ok {
			cfg.APIKey = v
		}
		if v, ok := config["base_url"].(string); ok {
			cfg.BaseURL = v
		}
		if v, ok := config["model"].(string); ok {
			cfg.Model = v
		}
		return NewOpenAIClient(cfg)

	case "anthropic", "claude":
		cfg := &AnthropicConfig{}
		if v, ok := config["api_key"].(string); ok {
			cfg.APIKey = v
		}
		if v, ok := config["model"].(string); ok {
			cfg.Model = v
		}
		return NewAnthropicClient(cfg)

	case "azure", "azure-openai":
		cfg := &AzureOpenAIConfig{}
		if v, ok := config["api_key"].(string); ok {
			cfg.APIKey = v
		}
		if v, ok := config["endpoint"].(string); ok {
			cfg.Endpoint = v
		}
		if v, ok := config["deployment"].(string); ok {
			cfg.Deployment = v
		}
		return NewAzureOpenAIClient(cfg)

	case "ollama":
		cfg := &OllamaConfig{}
		if v, ok := config["base_url"].(string); ok {
			cfg.BaseURL = v
		}
		if v, ok := config["model"].(string); ok {
			cfg.Model = v
		}
		return NewOllamaClient(cfg)

	default:
		return nil, fmt.Errorf("unsupported LLM provider: %s", provider)
	}
}

// LLMClientTool wraps an LLM client as a tool
type LLMClientTool struct {
	client LLMClient
	name   string
}

// NewLLMClientTool creates a new LLM client tool
func NewLLMClientTool(client LLMClient, name string) *LLMClientTool {
	if name == "" {
		name = "llm_" + client.Provider()
	}
	return &LLMClientTool{
		client: client,
		name:   name,
	}
}

// Name returns the tool name
func (t *LLMClientTool) Name() string {
	return t.name
}

// Description returns the tool description
func (t *LLMClientTool) Description() string {
	return fmt.Sprintf("Query an LLM (%s) for text generation", t.client.Provider())
}

// Schema returns the tool schema
func (t *LLMClientTool) Schema() *ToolSchema {
	return &ToolSchema{
		Type:        "object",
		Description: t.Description(),
		Properties: map[string]PropertySchema{
			"prompt": {
				Type:        "string",
				Description: "The prompt to send to the LLM",
			},
			"max_tokens": {
				Type:        "integer",
				Description: "Maximum tokens in response",
			},
		},
		Required: []string{"prompt"},
	}
}

// Execute runs the tool
func (t *LLMClientTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	prompt, ok := args["prompt"].(string)
	if !ok {
		return nil, fmt.Errorf("prompt must be a string")
	}

	maxTokens := 1000
	if v, ok := args["max_tokens"].(float64); ok {
		maxTokens = int(v)
	}

	resp, err := t.client.Complete(ctx, &CompletionRequest{
		Prompt:    prompt,
		MaxTokens: maxTokens,
	})
	if err != nil {
		return nil, err
	}

	return resp.Text, nil
}
