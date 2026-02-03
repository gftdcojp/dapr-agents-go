package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/dapr/go-sdk/client"
)

// MemoryBase defines the interface for conversation memory implementations
type MemoryBase interface {
	// AddMessage adds a single message to memory
	AddMessage(ctx context.Context, message Message) error

	// AddMessages adds multiple messages to memory
	AddMessages(ctx context.Context, messages []Message) error

	// AddInteraction adds a user-assistant message pair
	AddInteraction(ctx context.Context, userMsg, assistantMsg Message) error

	// GetMessages retrieves messages from memory
	GetMessages(ctx context.Context) ([]Message, error)

	// ResetMemory clears all messages
	ResetMemory(ctx context.Context) error

	// GetRelevantMessages retrieves messages relevant to a query (for vector memory)
	GetRelevantMessages(ctx context.Context, query string, k int) ([]Message, error)
}

// ConversationListMemory implements in-memory list-based storage
type ConversationListMemory struct {
	messages []Message
	maxSize  int
	mu       sync.RWMutex
}

// NewConversationListMemory creates a new list-based memory
func NewConversationListMemory(maxSize int) *ConversationListMemory {
	if maxSize <= 0 {
		maxSize = 100
	}
	return &ConversationListMemory{
		messages: make([]Message, 0),
		maxSize:  maxSize,
	}
}

func (m *ConversationListMemory) AddMessage(ctx context.Context, message Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if message.Timestamp.IsZero() {
		message.Timestamp = time.Now()
	}
	m.messages = append(m.messages, message)

	// Trim if exceeds max size
	if len(m.messages) > m.maxSize {
		m.messages = m.messages[len(m.messages)-m.maxSize:]
	}
	return nil
}

func (m *ConversationListMemory) AddMessages(ctx context.Context, messages []Message) error {
	for _, msg := range messages {
		if err := m.AddMessage(ctx, msg); err != nil {
			return err
		}
	}
	return nil
}

func (m *ConversationListMemory) AddInteraction(ctx context.Context, userMsg, assistantMsg Message) error {
	if err := m.AddMessage(ctx, userMsg); err != nil {
		return err
	}
	return m.AddMessage(ctx, assistantMsg)
}

func (m *ConversationListMemory) GetMessages(ctx context.Context) ([]Message, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]Message, len(m.messages))
	copy(result, m.messages)
	return result, nil
}

func (m *ConversationListMemory) ResetMemory(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.messages = make([]Message, 0)
	return nil
}

func (m *ConversationListMemory) GetRelevantMessages(ctx context.Context, query string, k int) ([]Message, error) {
	// List memory doesn't support semantic search, return last k messages
	m.mu.RLock()
	defer m.mu.RUnlock()

	if k >= len(m.messages) {
		result := make([]Message, len(m.messages))
		copy(result, m.messages)
		return result, nil
	}

	result := make([]Message, k)
	copy(result, m.messages[len(m.messages)-k:])
	return result, nil
}

// ConversationDaprStateMemory implements Dapr state store backed memory
type ConversationDaprStateMemory struct {
	storeName  string
	key        string
	maxSize    int
	daprClient client.Client
	mu         sync.RWMutex
}

// NewConversationDaprStateMemory creates a new Dapr state store backed memory
func NewConversationDaprStateMemory(storeName, key string, maxSize int) (*ConversationDaprStateMemory, error) {
	c, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	if maxSize <= 0 {
		maxSize = 100
	}

	return &ConversationDaprStateMemory{
		storeName:  storeName,
		key:        key,
		maxSize:    maxSize,
		daprClient: c,
	}, nil
}

func (m *ConversationDaprStateMemory) loadMessages(ctx context.Context) ([]Message, error) {
	item, err := m.daprClient.GetState(ctx, m.storeName, m.key, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get state: %w", err)
	}

	if len(item.Value) == 0 {
		return []Message{}, nil
	}

	var messages []Message
	if err := json.Unmarshal(item.Value, &messages); err != nil {
		return nil, fmt.Errorf("failed to unmarshal messages: %w", err)
	}

	return messages, nil
}

func (m *ConversationDaprStateMemory) saveMessages(ctx context.Context, messages []Message) error {
	data, err := json.Marshal(messages)
	if err != nil {
		return fmt.Errorf("failed to marshal messages: %w", err)
	}

	if err := m.daprClient.SaveState(ctx, m.storeName, m.key, data, nil); err != nil {
		return fmt.Errorf("failed to save state: %w", err)
	}

	return nil
}

func (m *ConversationDaprStateMemory) AddMessage(ctx context.Context, message Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	messages, err := m.loadMessages(ctx)
	if err != nil {
		messages = []Message{}
	}

	if message.Timestamp.IsZero() {
		message.Timestamp = time.Now()
	}
	messages = append(messages, message)

	if len(messages) > m.maxSize {
		messages = messages[len(messages)-m.maxSize:]
	}

	return m.saveMessages(ctx, messages)
}

func (m *ConversationDaprStateMemory) AddMessages(ctx context.Context, messages []Message) error {
	for _, msg := range messages {
		if err := m.AddMessage(ctx, msg); err != nil {
			return err
		}
	}
	return nil
}

func (m *ConversationDaprStateMemory) AddInteraction(ctx context.Context, userMsg, assistantMsg Message) error {
	if err := m.AddMessage(ctx, userMsg); err != nil {
		return err
	}
	return m.AddMessage(ctx, assistantMsg)
}

func (m *ConversationDaprStateMemory) GetMessages(ctx context.Context) ([]Message, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.loadMessages(ctx)
}

func (m *ConversationDaprStateMemory) ResetMemory(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if err := m.daprClient.DeleteState(ctx, m.storeName, m.key, nil); err != nil {
		return fmt.Errorf("failed to delete state: %w", err)
	}
	return nil
}

func (m *ConversationDaprStateMemory) GetRelevantMessages(ctx context.Context, query string, k int) ([]Message, error) {
	messages, err := m.GetMessages(ctx)
	if err != nil {
		return nil, err
	}

	if k >= len(messages) {
		return messages, nil
	}

	return messages[len(messages)-k:], nil
}

// ConversationVectorMemory implements vector store backed memory with semantic search
type ConversationVectorMemory struct {
	messages    []Message
	embeddings  [][]float64
	embedder    Embedder
	maxSize     int
	mu          sync.RWMutex
}

// Embedder defines the interface for text embedding
type Embedder interface {
	// Embed returns the embedding vector for the given text
	Embed(ctx context.Context, text string) ([]float64, error)

	// EmbedBatch returns embedding vectors for multiple texts
	EmbedBatch(ctx context.Context, texts []string) ([][]float64, error)
}

// DaprEmbedder uses Dapr binding to generate embeddings
type DaprEmbedder struct {
	componentName string
	daprClient    client.Client
}

// NewDaprEmbedder creates a new Dapr-based embedder
func NewDaprEmbedder(componentName string) (*DaprEmbedder, error) {
	c, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	return &DaprEmbedder{
		componentName: componentName,
		daprClient:    c,
	}, nil
}

func (e *DaprEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	input := map[string]interface{}{
		"input": text,
	}
	inputBytes, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	resp, err := e.daprClient.InvokeBinding(ctx, &client.InvokeBindingRequest{
		Name:      e.componentName,
		Operation: "embed",
		Data:      inputBytes,
	})
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}

	var result struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.Unmarshal(resp.Data, &result); err != nil {
		return nil, fmt.Errorf("failed to parse embedding: %w", err)
	}

	return result.Embedding, nil
}

func (e *DaprEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	embeddings := make([][]float64, len(texts))
	for i, text := range texts {
		emb, err := e.Embed(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("failed to embed text %d: %w", i, err)
		}
		embeddings[i] = emb
	}
	return embeddings, nil
}

// SimpleEmbedder provides a simple TF-IDF like embedding (for testing without LLM)
type SimpleEmbedder struct {
	vocabulary map[string]int
	idf        map[string]float64
	vocabSize  int
	mu         sync.RWMutex
}

// NewSimpleEmbedder creates a simple embedder for testing
func NewSimpleEmbedder(vocabSize int) *SimpleEmbedder {
	if vocabSize <= 0 {
		vocabSize = 1000
	}
	return &SimpleEmbedder{
		vocabulary: make(map[string]int),
		idf:        make(map[string]float64),
		vocabSize:  vocabSize,
	}
}

func (e *SimpleEmbedder) tokenize(text string) []string {
	// Simple tokenization by splitting on spaces and converting to lowercase
	tokens := make([]string, 0)
	word := ""
	for _, c := range text {
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' {
			if c >= 'A' && c <= 'Z' {
				c = c + 32 // to lowercase
			}
			word += string(c)
		} else if len(word) > 0 {
			tokens = append(tokens, word)
			word = ""
		}
	}
	if len(word) > 0 {
		tokens = append(tokens, word)
	}
	return tokens
}

func (e *SimpleEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	tokens := e.tokenize(text)
	embedding := make([]float64, e.vocabSize)

	// Count term frequency
	tf := make(map[string]int)
	for _, token := range tokens {
		tf[token]++
		// Add to vocabulary if not exists
		if _, exists := e.vocabulary[token]; !exists && len(e.vocabulary) < e.vocabSize {
			e.vocabulary[token] = len(e.vocabulary)
		}
	}

	// Create embedding vector
	for token, count := range tf {
		if idx, exists := e.vocabulary[token]; exists {
			embedding[idx] = float64(count) / float64(len(tokens))
		}
	}

	// Normalize
	norm := 0.0
	for _, v := range embedding {
		norm += v * v
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range embedding {
			embedding[i] /= norm
		}
	}

	return embedding, nil
}

func (e *SimpleEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
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

// NewConversationVectorMemory creates a new vector-based memory
func NewConversationVectorMemory(embedder Embedder, maxSize int) *ConversationVectorMemory {
	if maxSize <= 0 {
		maxSize = 100
	}
	return &ConversationVectorMemory{
		messages:   make([]Message, 0),
		embeddings: make([][]float64, 0),
		embedder:   embedder,
		maxSize:    maxSize,
	}
}

func (m *ConversationVectorMemory) AddMessage(ctx context.Context, message Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if message.Timestamp.IsZero() {
		message.Timestamp = time.Now()
	}

	// Generate embedding for the message
	embedding, err := m.embedder.Embed(ctx, message.Content)
	if err != nil {
		return fmt.Errorf("failed to embed message: %w", err)
	}

	m.messages = append(m.messages, message)
	m.embeddings = append(m.embeddings, embedding)

	// Trim if exceeds max size
	if len(m.messages) > m.maxSize {
		m.messages = m.messages[len(m.messages)-m.maxSize:]
		m.embeddings = m.embeddings[len(m.embeddings)-m.maxSize:]
	}

	return nil
}

func (m *ConversationVectorMemory) AddMessages(ctx context.Context, messages []Message) error {
	for _, msg := range messages {
		if err := m.AddMessage(ctx, msg); err != nil {
			return err
		}
	}
	return nil
}

func (m *ConversationVectorMemory) AddInteraction(ctx context.Context, userMsg, assistantMsg Message) error {
	if err := m.AddMessage(ctx, userMsg); err != nil {
		return err
	}
	return m.AddMessage(ctx, assistantMsg)
}

func (m *ConversationVectorMemory) GetMessages(ctx context.Context) ([]Message, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]Message, len(m.messages))
	copy(result, m.messages)
	return result, nil
}

func (m *ConversationVectorMemory) ResetMemory(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.messages = make([]Message, 0)
	m.embeddings = make([][]float64, 0)
	return nil
}

// cosineSimilarity calculates cosine similarity between two vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// GetRelevantMessages retrieves the k most relevant messages to the query
func (m *ConversationVectorMemory) GetRelevantMessages(ctx context.Context, query string, k int) ([]Message, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.messages) == 0 {
		return []Message{}, nil
	}

	if k >= len(m.messages) {
		result := make([]Message, len(m.messages))
		copy(result, m.messages)
		return result, nil
	}

	// Embed the query
	queryEmbedding, err := m.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	// Calculate similarities
	type scoredMessage struct {
		index      int
		similarity float64
	}

	scores := make([]scoredMessage, len(m.messages))
	for i, emb := range m.embeddings {
		scores[i] = scoredMessage{
			index:      i,
			similarity: cosineSimilarity(queryEmbedding, emb),
		}
	}

	// Sort by similarity (descending)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].similarity > scores[j].similarity
	})

	// Return top k
	result := make([]Message, k)
	for i := 0; i < k; i++ {
		result[i] = m.messages[scores[i].index]
	}

	return result, nil
}

// MemoryConfig configures memory behavior
type MemoryConfig struct {
	Type           string // list, dapr, vector
	MaxSize        int
	StoreName      string // for dapr
	Key            string // for dapr
	EmbedComponent string // for vector (Dapr binding)
	VocabSize      int    // for simple embedder
}

// NewMemory creates a memory instance based on config
func NewMemory(config *MemoryConfig) (MemoryBase, error) {
	if config == nil {
		return NewConversationListMemory(100), nil
	}

	switch config.Type {
	case "list", "":
		return NewConversationListMemory(config.MaxSize), nil

	case "dapr":
		if config.StoreName == "" {
			config.StoreName = "statestore"
		}
		if config.Key == "" {
			config.Key = "conversation-memory"
		}
		return NewConversationDaprStateMemory(config.StoreName, config.Key, config.MaxSize)

	case "vector":
		var embedder Embedder
		if config.EmbedComponent != "" {
			var err error
			embedder, err = NewDaprEmbedder(config.EmbedComponent)
			if err != nil {
				return nil, fmt.Errorf("failed to create Dapr embedder: %w", err)
			}
		} else {
			embedder = NewSimpleEmbedder(config.VocabSize)
		}
		return NewConversationVectorMemory(embedder, config.MaxSize), nil

	default:
		return nil, fmt.Errorf("unknown memory type: %s", config.Type)
	}
}
