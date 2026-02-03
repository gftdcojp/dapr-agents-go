package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// ============================================================================
// OpenAI Embedder (Python: OpenAIEmbedder)
// ============================================================================

// OpenAIEmbedder generates embeddings using OpenAI API
type OpenAIEmbedder struct {
	apiKey     string
	baseURL    string
	model      string
	dimensions int
	httpClient *http.Client
}

// OpenAIEmbedderConfig configures the OpenAI embedder
type OpenAIEmbedderConfig struct {
	APIKey     string
	BaseURL    string
	Model      string
	Dimensions int
	Timeout    time.Duration
}

// NewOpenAIEmbedder creates a new OpenAI embedder
func NewOpenAIEmbedder(config *OpenAIEmbedderConfig) (*OpenAIEmbedder, error) {
	if config == nil {
		config = &OpenAIEmbedderConfig{}
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
		model = "text-embedding-3-small"
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	return &OpenAIEmbedder{
		apiKey:     apiKey,
		baseURL:    baseURL,
		model:      model,
		dimensions: config.Dimensions,
		httpClient: &http.Client{Timeout: timeout},
	}, nil
}

// Embed generates embeddings for text
func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	reqBody := map[string]interface{}{
		"input": text,
		"model": e.model,
	}
	if e.dimensions > 0 {
		reqBody["dimensions"] = e.dimensions
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.httpClient.Do(req)
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

// EmbedBatch generates embeddings for multiple texts
func (e *OpenAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	reqBody := map[string]interface{}{
		"input": texts,
		"model": e.model,
	}
	if e.dimensions > 0 {
		reqBody["dimensions"] = e.dimensions
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.httpClient.Do(req)
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
			Index     int       `json:"index"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Sort by index and extract embeddings
	embeddings := make([][]float64, len(texts))
	for _, d := range result.Data {
		if d.Index < len(embeddings) {
			embeddings[d.Index] = d.Embedding
		}
	}

	return embeddings, nil
}

// ============================================================================
// Azure OpenAI Embedder
// ============================================================================

// AzureOpenAIEmbedder generates embeddings using Azure OpenAI
type AzureOpenAIEmbedder struct {
	apiKey     string
	endpoint   string
	deployment string
	apiVersion string
	httpClient *http.Client
}

// AzureOpenAIEmbedderConfig configures the Azure embedder
type AzureOpenAIEmbedderConfig struct {
	APIKey     string
	Endpoint   string
	Deployment string
	APIVersion string
	Timeout    time.Duration
}

// NewAzureOpenAIEmbedder creates a new Azure OpenAI embedder
func NewAzureOpenAIEmbedder(config *AzureOpenAIEmbedderConfig) (*AzureOpenAIEmbedder, error) {
	if config == nil {
		config = &AzureOpenAIEmbedderConfig{}
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
		timeout = 30 * time.Second
	}

	return &AzureOpenAIEmbedder{
		apiKey:     apiKey,
		endpoint:   endpoint,
		deployment: config.Deployment,
		apiVersion: apiVersion,
		httpClient: &http.Client{Timeout: timeout},
	}, nil
}

// Embed generates embeddings for text
func (e *AzureOpenAIEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	url := fmt.Sprintf("%s/openai/deployments/%s/embeddings?api-version=%s",
		e.endpoint, e.deployment, e.apiVersion)

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
	req.Header.Set("api-key", e.apiKey)

	resp, err := e.httpClient.Do(req)
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

// EmbedBatch generates embeddings for multiple texts
func (e *AzureOpenAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	url := fmt.Sprintf("%s/openai/deployments/%s/embeddings?api-version=%s",
		e.endpoint, e.deployment, e.apiVersion)

	body, err := json.Marshal(map[string]interface{}{
		"input": texts,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", e.apiKey)

	resp, err := e.httpClient.Do(req)
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
			Index     int       `json:"index"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Sort by index and extract embeddings
	embeddings := make([][]float64, len(texts))
	for _, d := range result.Data {
		if d.Index < len(embeddings) {
			embeddings[d.Index] = d.Embedding
		}
	}

	return embeddings, nil
}

// ============================================================================
// NVIDIA Embedder (Python: NVIDIAEmbedder)
// ============================================================================

// NVIDIAEmbedder generates embeddings using NVIDIA NIM
type NVIDIAEmbedder struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
}

// NVIDIAEmbedderConfig configures the NVIDIA embedder
type NVIDIAEmbedderConfig struct {
	APIKey  string
	BaseURL string
	Model   string
	Timeout time.Duration
}

// NewNVIDIAEmbedder creates a new NVIDIA embedder
func NewNVIDIAEmbedder(config *NVIDIAEmbedderConfig) (*NVIDIAEmbedder, error) {
	if config == nil {
		config = &NVIDIAEmbedderConfig{}
	}

	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("NVIDIA_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("NVIDIA API key is required")
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "https://integrate.api.nvidia.com/v1"
	}

	model := config.Model
	if model == "" {
		model = "nvidia/nv-embedqa-e5-v5"
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	return &NVIDIAEmbedder{
		apiKey:     apiKey,
		baseURL:    baseURL,
		model:      model,
		httpClient: &http.Client{Timeout: timeout},
	}, nil
}

// Embed generates embeddings for text
func (e *NVIDIAEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	body, err := json.Marshal(map[string]interface{}{
		"input": []string{text},
		"model": e.model,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("NVIDIA API error %d: %s", resp.StatusCode, string(bodyBytes))
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

// EmbedBatch generates embeddings for multiple texts
func (e *NVIDIAEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	body, err := json.Marshal(map[string]interface{}{
		"input": texts,
		"model": e.model,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("NVIDIA API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Sort by index and extract embeddings
	embeddings := make([][]float64, len(texts))
	for _, d := range result.Data {
		if d.Index < len(embeddings) {
			embeddings[d.Index] = d.Embedding
		}
	}

	return embeddings, nil
}

// ============================================================================
// Ollama Embedder
// ============================================================================

// OllamaEmbedder generates embeddings using Ollama
type OllamaEmbedder struct {
	baseURL    string
	model      string
	httpClient *http.Client
}

// OllamaEmbedderConfig configures the Ollama embedder
type OllamaEmbedderConfig struct {
	BaseURL string
	Model   string
	Timeout time.Duration
}

// NewOllamaEmbedder creates a new Ollama embedder
func NewOllamaEmbedder(config *OllamaEmbedderConfig) (*OllamaEmbedder, error) {
	if config == nil {
		config = &OllamaEmbedderConfig{}
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
		model = "nomic-embed-text"
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 60 * time.Second
	}

	return &OllamaEmbedder{
		baseURL:    baseURL,
		model:      model,
		httpClient: &http.Client{Timeout: timeout},
	}, nil
}

// Embed generates embeddings for text
func (e *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	body, err := json.Marshal(map[string]interface{}{
		"model":  e.model,
		"prompt": text,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.baseURL+"/api/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := e.httpClient.Do(req)
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

// EmbedBatch generates embeddings for multiple texts
func (e *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	embeddings := make([][]float64, len(texts))
	for i, text := range texts {
		emb, err := e.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		embeddings[i] = emb
	}
	return embeddings, nil
}

// ============================================================================
// Hugging Face Embedder (Python: SentenceTransformerEmbedder via HF)
// ============================================================================

// HuggingFaceEmbedder generates embeddings using Hugging Face Inference API
type HuggingFaceEmbedder struct {
	apiKey     string
	model      string
	httpClient *http.Client
}

// HuggingFaceEmbedderConfig configures the Hugging Face embedder
type HuggingFaceEmbedderConfig struct {
	APIKey  string
	Model   string
	Timeout time.Duration
}

// NewHuggingFaceEmbedder creates a new Hugging Face embedder
func NewHuggingFaceEmbedder(config *HuggingFaceEmbedderConfig) (*HuggingFaceEmbedder, error) {
	if config == nil {
		config = &HuggingFaceEmbedderConfig{}
	}

	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("HUGGINGFACE_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("Hugging Face API key is required")
	}

	model := config.Model
	if model == "" {
		model = "sentence-transformers/all-MiniLM-L6-v2"
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	return &HuggingFaceEmbedder{
		apiKey:     apiKey,
		model:      model,
		httpClient: &http.Client{Timeout: timeout},
	}, nil
}

// Embed generates embeddings for text
func (e *HuggingFaceEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	url := fmt.Sprintf("https://api-inference.huggingface.co/pipeline/feature-extraction/%s", e.model)

	body, err := json.Marshal(map[string]interface{}{
		"inputs": text,
		"options": map[string]interface{}{
			"wait_for_model": true,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Hugging Face API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var embedding []float64
	if err := json.NewDecoder(resp.Body).Decode(&embedding); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return embedding, nil
}

// EmbedBatch generates embeddings for multiple texts
func (e *HuggingFaceEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	url := fmt.Sprintf("https://api-inference.huggingface.co/pipeline/feature-extraction/%s", e.model)

	body, err := json.Marshal(map[string]interface{}{
		"inputs": texts,
		"options": map[string]interface{}{
			"wait_for_model": true,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Hugging Face API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var embeddings [][]float64
	if err := json.NewDecoder(resp.Body).Decode(&embeddings); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return embeddings, nil
}

// ============================================================================
// Embedder Factory
// ============================================================================

// NewEmbedder creates an embedder by provider name
func NewEmbedder(provider string, config map[string]interface{}) (Embedder, error) {
	switch provider {
	case "openai":
		cfg := &OpenAIEmbedderConfig{}
		if v, ok := config["api_key"].(string); ok {
			cfg.APIKey = v
		}
		if v, ok := config["model"].(string); ok {
			cfg.Model = v
		}
		if v, ok := config["dimensions"].(int); ok {
			cfg.Dimensions = v
		}
		return NewOpenAIEmbedder(cfg)

	case "azure", "azure-openai":
		cfg := &AzureOpenAIEmbedderConfig{}
		if v, ok := config["api_key"].(string); ok {
			cfg.APIKey = v
		}
		if v, ok := config["endpoint"].(string); ok {
			cfg.Endpoint = v
		}
		if v, ok := config["deployment"].(string); ok {
			cfg.Deployment = v
		}
		return NewAzureOpenAIEmbedder(cfg)

	case "nvidia":
		cfg := &NVIDIAEmbedderConfig{}
		if v, ok := config["api_key"].(string); ok {
			cfg.APIKey = v
		}
		if v, ok := config["model"].(string); ok {
			cfg.Model = v
		}
		return NewNVIDIAEmbedder(cfg)

	case "ollama":
		cfg := &OllamaEmbedderConfig{}
		if v, ok := config["base_url"].(string); ok {
			cfg.BaseURL = v
		}
		if v, ok := config["model"].(string); ok {
			cfg.Model = v
		}
		return NewOllamaEmbedder(cfg)

	case "huggingface", "hf":
		cfg := &HuggingFaceEmbedderConfig{}
		if v, ok := config["api_key"].(string); ok {
			cfg.APIKey = v
		}
		if v, ok := config["model"].(string); ok {
			cfg.Model = v
		}
		return NewHuggingFaceEmbedder(cfg)

	default:
		return nil, fmt.Errorf("unsupported embedder provider: %s", provider)
	}
}

// SentenceTransformerEmbedder is an alias for HuggingFaceEmbedder
// Python SDK compatibility
type SentenceTransformerEmbedder = HuggingFaceEmbedder

// NewSentenceTransformerEmbedder creates a SentenceTransformer embedder
func NewSentenceTransformerEmbedder(model string) (*HuggingFaceEmbedder, error) {
	return NewHuggingFaceEmbedder(&HuggingFaceEmbedderConfig{
		Model: model,
	})
}
