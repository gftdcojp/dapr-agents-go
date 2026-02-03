package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/dapr/go-sdk/client"
)

// FuncTool wraps a simple function as a Tool
type FuncTool struct {
	name        string
	description string
	schema      *ToolSchema
	fn          func(ctx context.Context, params map[string]interface{}) (interface{}, error)
}

// NewFuncTool creates a new function-based tool
func NewFuncTool(name, description string, schema *ToolSchema, fn func(ctx context.Context, params map[string]interface{}) (interface{}, error)) *FuncTool {
	return &FuncTool{
		name:        name,
		description: description,
		schema:      schema,
		fn:          fn,
	}
}

func (t *FuncTool) Name() string               { return t.name }
func (t *FuncTool) Description() string        { return t.description }
func (t *FuncTool) Schema() *ToolSchema        { return t.schema }

func (t *FuncTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	return t.fn(ctx, params)
}

// HTTPTool calls an HTTP endpoint as a tool
type HTTPTool struct {
	name        string
	description string
	schema      *ToolSchema
	method      string
	url         string
	headers     map[string]string
	client      *http.Client
}

// HTTPToolConfig configures an HTTP-based tool
type HTTPToolConfig struct {
	Name        string
	Description string
	Schema      *ToolSchema
	Method      string // GET, POST, etc.
	URL         string
	Headers     map[string]string
	Timeout     time.Duration
}

// NewHTTPTool creates a new HTTP-based tool
func NewHTTPTool(config HTTPToolConfig) *HTTPTool {
	timeout := config.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	method := config.Method
	if method == "" {
		method = "POST"
	}

	return &HTTPTool{
		name:        config.Name,
		description: config.Description,
		schema:      config.Schema,
		method:      method,
		url:         config.URL,
		headers:     config.Headers,
		client:      &http.Client{Timeout: timeout},
	}
}

func (t *HTTPTool) Name() string               { return t.name }
func (t *HTTPTool) Description() string        { return t.description }
func (t *HTTPTool) Schema() *ToolSchema        { return t.schema }

func (t *HTTPTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	// This is a simplified implementation
	// In production, you'd want proper request building, error handling, etc.
	return map[string]interface{}{
		"url":    t.url,
		"method": t.method,
		"params": params,
	}, nil
}

// DaprServiceTool invokes a Dapr service as a tool
type DaprServiceTool struct {
	name        string
	description string
	schema      *ToolSchema
	appID       string
	methodName  string
	daprClient  client.Client
}

// DaprServiceToolConfig configures a Dapr service invocation tool
type DaprServiceToolConfig struct {
	Name        string
	Description string
	Schema      *ToolSchema
	AppID       string // Target Dapr app ID
	MethodName  string // Method to invoke
}

// NewDaprServiceTool creates a tool that invokes a Dapr service
func NewDaprServiceTool(config DaprServiceToolConfig) (*DaprServiceTool, error) {
	daprClient, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	return &DaprServiceTool{
		name:        config.Name,
		description: config.Description,
		schema:      config.Schema,
		appID:       config.AppID,
		methodName:  config.MethodName,
		daprClient:  daprClient,
	}, nil
}

func (t *DaprServiceTool) Name() string               { return t.name }
func (t *DaprServiceTool) Description() string        { return t.description }
func (t *DaprServiceTool) Schema() *ToolSchema        { return t.schema }

func (t *DaprServiceTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	data, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal params: %w", err)
	}

	content := &client.DataContent{
		ContentType: "application/json",
		Data:        data,
	}

	resp, err := t.daprClient.InvokeMethodWithContent(ctx, t.appID, t.methodName, "POST", content)
	if err != nil {
		return nil, fmt.Errorf("service invocation failed: %w", err)
	}

	var result interface{}
	if err := json.Unmarshal(resp, &result); err != nil {
		// Return raw response if not JSON
		return string(resp), nil
	}

	return result, nil
}

// DaprActorTool invokes a Dapr Actor method as a tool
type DaprActorTool struct {
	name        string
	description string
	schema      *ToolSchema
	actorType   string
	actorID     string
	methodName  string
	daprClient  client.Client
}

// DaprActorToolConfig configures a Dapr Actor invocation tool
type DaprActorToolConfig struct {
	Name        string
	Description string
	Schema      *ToolSchema
	ActorType   string
	ActorID     string // Can be empty to use dynamic ID from params
	MethodName  string
}

// NewDaprActorTool creates a tool that invokes a Dapr Actor method
func NewDaprActorTool(config DaprActorToolConfig) (*DaprActorTool, error) {
	daprClient, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	return &DaprActorTool{
		name:        config.Name,
		description: config.Description,
		schema:      config.Schema,
		actorType:   config.ActorType,
		actorID:     config.ActorID,
		methodName:  config.MethodName,
		daprClient:  daprClient,
	}, nil
}

func (t *DaprActorTool) Name() string               { return t.name }
func (t *DaprActorTool) Description() string        { return t.description }
func (t *DaprActorTool) Schema() *ToolSchema        { return t.schema }

func (t *DaprActorTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	// Get actor ID from params or use configured ID
	actorID := t.actorID
	if idParam, ok := params["actorId"].(string); ok && idParam != "" {
		actorID = idParam
		delete(params, "actorId")
	}
	if actorID == "" {
		return nil, fmt.Errorf("actor ID is required")
	}

	data, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal params: %w", err)
	}

	resp, err := t.daprClient.InvokeActor(ctx, &client.InvokeActorRequest{
		ActorType: t.actorType,
		ActorID:   actorID,
		Method:    t.methodName,
		Data:      data,
	})
	if err != nil {
		return nil, fmt.Errorf("actor invocation failed: %w", err)
	}

	var result interface{}
	if err := json.Unmarshal(resp.Data, &result); err != nil {
		return string(resp.Data), nil
	}

	return result, nil
}

// DaprStateTool reads/writes to Dapr state store as a tool
type DaprStateTool struct {
	name        string
	description string
	schema      *ToolSchema
	storeName   string
	operation   string // get, set, delete
	daprClient  client.Client
}

// DaprStateToolConfig configures a Dapr state store tool
type DaprStateToolConfig struct {
	Name        string
	Description string
	Schema      *ToolSchema
	StoreName   string
	Operation   string // get, set, delete
}

// NewDaprStateTool creates a tool for Dapr state operations
func NewDaprStateTool(config DaprStateToolConfig) (*DaprStateTool, error) {
	daprClient, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	return &DaprStateTool{
		name:        config.Name,
		description: config.Description,
		schema:      config.Schema,
		storeName:   config.StoreName,
		operation:   config.Operation,
		daprClient:  daprClient,
	}, nil
}

func (t *DaprStateTool) Name() string               { return t.name }
func (t *DaprStateTool) Description() string        { return t.description }
func (t *DaprStateTool) Schema() *ToolSchema        { return t.schema }

func (t *DaprStateTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	key, ok := params["key"].(string)
	if !ok {
		return nil, fmt.Errorf("key parameter is required")
	}

	switch t.operation {
	case "get":
		item, err := t.daprClient.GetState(ctx, t.storeName, key, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to get state: %w", err)
		}
		var result interface{}
		if err := json.Unmarshal(item.Value, &result); err != nil {
			return string(item.Value), nil
		}
		return result, nil

	case "set":
		value, ok := params["value"]
		if !ok {
			return nil, fmt.Errorf("value parameter is required for set operation")
		}
		data, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal value: %w", err)
		}
		if err := t.daprClient.SaveState(ctx, t.storeName, key, data, nil); err != nil {
			return nil, fmt.Errorf("failed to save state: %w", err)
		}
		return map[string]string{"status": "saved"}, nil

	case "delete":
		if err := t.daprClient.DeleteState(ctx, t.storeName, key, nil); err != nil {
			return nil, fmt.Errorf("failed to delete state: %w", err)
		}
		return map[string]string{"status": "deleted"}, nil

	default:
		return nil, fmt.Errorf("unsupported operation: %s", t.operation)
	}
}

// DaprPubSubTool publishes messages to Dapr pub/sub as a tool
type DaprPubSubTool struct {
	name        string
	description string
	schema      *ToolSchema
	pubsubName  string
	topic       string
	daprClient  client.Client
}

// DaprPubSubToolConfig configures a Dapr pub/sub tool
type DaprPubSubToolConfig struct {
	Name        string
	Description string
	Schema      *ToolSchema
	PubSubName  string
	Topic       string // Can be empty to use topic from params
}

// NewDaprPubSubTool creates a tool for publishing to Dapr pub/sub
func NewDaprPubSubTool(config DaprPubSubToolConfig) (*DaprPubSubTool, error) {
	daprClient, err := client.NewClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Dapr client: %w", err)
	}

	return &DaprPubSubTool{
		name:        config.Name,
		description: config.Description,
		schema:      config.Schema,
		pubsubName:  config.PubSubName,
		topic:       config.Topic,
		daprClient:  daprClient,
	}, nil
}

func (t *DaprPubSubTool) Name() string               { return t.name }
func (t *DaprPubSubTool) Description() string        { return t.description }
func (t *DaprPubSubTool) Schema() *ToolSchema        { return t.schema }

func (t *DaprPubSubTool) Execute(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	topic := t.topic
	if topicParam, ok := params["topic"].(string); ok && topicParam != "" {
		topic = topicParam
		delete(params, "topic")
	}
	if topic == "" {
		return nil, fmt.Errorf("topic is required")
	}

	message, ok := params["message"]
	if !ok {
		message = params
	}

	data, err := json.Marshal(message)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal message: %w", err)
	}

	if err := t.daprClient.PublishEvent(ctx, t.pubsubName, topic, data); err != nil {
		return nil, fmt.Errorf("failed to publish event: %w", err)
	}

	return map[string]interface{}{
		"status": "published",
		"topic":  topic,
	}, nil
}

// ToolBuilder provides a fluent API for building tools
type ToolBuilder struct {
	name        string
	description string
	schema      *ToolSchema
}

// NewToolBuilder creates a new tool builder
func NewToolBuilder(name string) *ToolBuilder {
	return &ToolBuilder{
		name: name,
		schema: &ToolSchema{
			Type:       "object",
			Properties: make(map[string]PropertySchema),
		},
	}
}

// Description sets the tool description
func (b *ToolBuilder) Description(desc string) *ToolBuilder {
	b.description = desc
	return b
}

// AddParam adds a parameter to the tool schema
func (b *ToolBuilder) AddParam(name, paramType, description string, required bool) *ToolBuilder {
	b.schema.Properties[name] = PropertySchema{
		Type:        paramType,
		Description: description,
	}
	if required {
		b.schema.Required = append(b.schema.Required, name)
	}
	return b
}

// AddStringParam adds a string parameter
func (b *ToolBuilder) AddStringParam(name, description string, required bool) *ToolBuilder {
	return b.AddParam(name, "string", description, required)
}

// AddNumberParam adds a number parameter
func (b *ToolBuilder) AddNumberParam(name, description string, required bool) *ToolBuilder {
	return b.AddParam(name, "number", description, required)
}

// AddBoolParam adds a boolean parameter
func (b *ToolBuilder) AddBoolParam(name, description string, required bool) *ToolBuilder {
	return b.AddParam(name, "boolean", description, required)
}

// AddEnumParam adds an enum parameter
func (b *ToolBuilder) AddEnumParam(name, description string, values []string, required bool) *ToolBuilder {
	b.schema.Properties[name] = PropertySchema{
		Type:        "string",
		Description: description,
		Enum:        values,
	}
	if required {
		b.schema.Required = append(b.schema.Required, name)
	}
	return b
}

// BuildFunc creates a FuncTool with the given handler
func (b *ToolBuilder) BuildFunc(fn func(ctx context.Context, params map[string]interface{}) (interface{}, error)) *FuncTool {
	return NewFuncTool(b.name, b.description, b.schema, fn)
}

// BuildHTTP creates an HTTPTool
func (b *ToolBuilder) BuildHTTP(method, url string) *HTTPTool {
	return NewHTTPTool(HTTPToolConfig{
		Name:        b.name,
		Description: b.description,
		Schema:      b.schema,
		Method:      method,
		URL:         url,
	})
}

// BuildDaprService creates a DaprServiceTool
func (b *ToolBuilder) BuildDaprService(appID, methodName string) (*DaprServiceTool, error) {
	return NewDaprServiceTool(DaprServiceToolConfig{
		Name:        b.name,
		Description: b.description,
		Schema:      b.schema,
		AppID:       appID,
		MethodName:  methodName,
	})
}

// BuildDaprActor creates a DaprActorTool
func (b *ToolBuilder) BuildDaprActor(actorType, methodName string) (*DaprActorTool, error) {
	return NewDaprActorTool(DaprActorToolConfig{
		Name:        b.name,
		Description: b.description,
		Schema:      b.schema,
		ActorType:   actorType,
		MethodName:  methodName,
	})
}

// Schema returns the current tool schema
func (b *ToolBuilder) Schema() *ToolSchema {
	return b.schema
}
