package agent

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/exporters/zipkin"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"go.opentelemetry.io/otel/trace"
)

// Instrumentor provides OpenTelemetry instrumentation for Dapr Agents
type Instrumentor struct {
	config         *InstrumentorConfig
	tracerProvider *sdktrace.TracerProvider
	tracer         trace.Tracer
	shutdown       func(context.Context) error
}

// InstrumentorConfig configures the instrumentor
type InstrumentorConfig struct {
	// ServiceName is the name of the service
	ServiceName string

	// ServiceVersion is the version of the service
	ServiceVersion string

	// Environment is the deployment environment (production, staging, etc.)
	Environment string

	// ExporterType specifies the exporter (otlp, zipkin, console)
	ExporterType string

	// OTLPEndpoint is the OTLP collector endpoint
	OTLPEndpoint string

	// OTLPProtocol is the OTLP protocol (grpc or http)
	OTLPProtocol string

	// ZipkinEndpoint is the Zipkin collector endpoint
	ZipkinEndpoint string

	// SampleRate is the trace sampling rate (0.0 to 1.0)
	SampleRate float64

	// EnableConsoleExporter also exports to console (for debugging)
	EnableConsoleExporter bool
}

// DefaultInstrumentorConfig returns sensible defaults
func DefaultInstrumentorConfig() *InstrumentorConfig {
	serviceName := os.Getenv("OTEL_SERVICE_NAME")
	if serviceName == "" {
		serviceName = "dapr-agent"
	}

	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		endpoint = "localhost:4317"
	}

	return &InstrumentorConfig{
		ServiceName:    serviceName,
		ServiceVersion: "1.0.0",
		Environment:    os.Getenv("OTEL_ENVIRONMENT"),
		ExporterType:   "otlp",
		OTLPEndpoint:   endpoint,
		OTLPProtocol:   "grpc",
		SampleRate:     1.0,
	}
}

// NewInstrumentor creates a new instrumentor
func NewInstrumentor(config *InstrumentorConfig) (*Instrumentor, error) {
	if config == nil {
		config = DefaultInstrumentorConfig()
	}

	// Create resource
	res, err := resource.New(context.Background(),
		resource.WithAttributes(
			semconv.ServiceName(config.ServiceName),
			semconv.ServiceVersion(config.ServiceVersion),
			attribute.String("environment", config.Environment),
			attribute.String("library.name", "dapr-agents-go"),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Create exporters
	var exporters []sdktrace.SpanExporter

	switch config.ExporterType {
	case "otlp":
		exporter, err := createOTLPExporter(config)
		if err != nil {
			return nil, fmt.Errorf("failed to create OTLP exporter: %w", err)
		}
		exporters = append(exporters, exporter)

	case "zipkin":
		exporter, err := createZipkinExporter(config)
		if err != nil {
			return nil, fmt.Errorf("failed to create Zipkin exporter: %w", err)
		}
		exporters = append(exporters, exporter)

	case "console":
		exporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
		if err != nil {
			return nil, fmt.Errorf("failed to create console exporter: %w", err)
		}
		exporters = append(exporters, exporter)
	}

	if config.EnableConsoleExporter && config.ExporterType != "console" {
		exporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
		if err != nil {
			log.Printf("failed to create console exporter: %v", err)
		} else {
			exporters = append(exporters, exporter)
		}
	}

	// Create tracer provider options
	opts := []sdktrace.TracerProviderOption{
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.TraceIDRatioBased(config.SampleRate)),
	}

	for _, exporter := range exporters {
		opts = append(opts, sdktrace.WithBatcher(exporter))
	}

	// Create tracer provider
	tp := sdktrace.NewTracerProvider(opts...)

	// Set global tracer provider
	otel.SetTracerProvider(tp)

	// Set global propagator
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	inst := &Instrumentor{
		config:         config,
		tracerProvider: tp,
		tracer:         tp.Tracer("dapr-agents-go"),
		shutdown: func(ctx context.Context) error {
			return tp.Shutdown(ctx)
		},
	}

	return inst, nil
}

func createOTLPExporter(config *InstrumentorConfig) (sdktrace.SpanExporter, error) {
	ctx := context.Background()

	switch config.OTLPProtocol {
	case "grpc":
		client := otlptracegrpc.NewClient(
			otlptracegrpc.WithEndpoint(config.OTLPEndpoint),
			otlptracegrpc.WithInsecure(), // TODO: make configurable
		)
		return otlptrace.New(ctx, client)

	case "http":
		client := otlptracehttp.NewClient(
			otlptracehttp.WithEndpoint(config.OTLPEndpoint),
			otlptracehttp.WithInsecure(), // TODO: make configurable
		)
		return otlptrace.New(ctx, client)

	default:
		return nil, fmt.Errorf("unsupported OTLP protocol: %s", config.OTLPProtocol)
	}
}

func createZipkinExporter(config *InstrumentorConfig) (sdktrace.SpanExporter, error) {
	endpoint := config.ZipkinEndpoint
	if endpoint == "" {
		endpoint = "http://localhost:9411/api/v2/spans"
	}
	return zipkin.New(endpoint)
}

// Shutdown shuts down the instrumentor
func (i *Instrumentor) Shutdown(ctx context.Context) error {
	if i.shutdown != nil {
		return i.shutdown(ctx)
	}
	return nil
}

// Tracer returns the tracer
func (i *Instrumentor) Tracer() trace.Tracer {
	return i.tracer
}

// StartSpan starts a new span
func (i *Instrumentor) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	return i.tracer.Start(ctx, name, opts...)
}

// SpanAttributes returns common span attributes for agents
func SpanAttributes(agentType, agentID string) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String("agent.type", agentType),
		attribute.String("agent.id", agentID),
	}
}

// Traced wraps a function with tracing
func (i *Instrumentor) Traced(ctx context.Context, name string, fn func(context.Context) error) error {
	ctx, span := i.StartSpan(ctx, name)
	defer span.End()

	err := fn(ctx)
	if err != nil {
		span.RecordError(err)
	}
	return err
}

// AgentSpan represents a traced agent operation
type AgentSpan struct {
	span      trace.Span
	startTime time.Time
}

// NewAgentSpan creates a new agent span
func (i *Instrumentor) NewAgentSpan(ctx context.Context, operation string, agentType, agentID string) (context.Context, *AgentSpan) {
	ctx, span := i.tracer.Start(ctx, fmt.Sprintf("agent.%s", operation),
		trace.WithAttributes(SpanAttributes(agentType, agentID)...),
	)

	return ctx, &AgentSpan{
		span:      span,
		startTime: time.Now(),
	}
}

// SetInput sets the input attribute on the span
func (s *AgentSpan) SetInput(input string) {
	s.span.SetAttributes(attribute.String("agent.input", truncate(input, 1000)))
}

// SetOutput sets the output attribute on the span
func (s *AgentSpan) SetOutput(output string) {
	s.span.SetAttributes(attribute.String("agent.output", truncate(output, 1000)))
}

// SetStepCount sets the step count attribute
func (s *AgentSpan) SetStepCount(count int) {
	s.span.SetAttributes(attribute.Int("agent.steps", count))
}

// RecordToolCall records a tool call event
func (s *AgentSpan) RecordToolCall(toolName string, args map[string]interface{}) {
	s.span.AddEvent("tool_call", trace.WithAttributes(
		attribute.String("tool.name", toolName),
	))
}

// RecordLLMCall records an LLM call event
func (s *AgentSpan) RecordLLMCall(model string, tokenCount int) {
	s.span.AddEvent("llm_call", trace.WithAttributes(
		attribute.String("llm.model", model),
		attribute.Int("llm.tokens", tokenCount),
	))
}

// RecordError records an error on the span
func (s *AgentSpan) RecordError(err error) {
	s.span.RecordError(err)
}

// End ends the span
func (s *AgentSpan) End() {
	duration := time.Since(s.startTime)
	s.span.SetAttributes(attribute.Int64("agent.duration_ms", duration.Milliseconds()))
	s.span.End()
}

// ToolSpan represents a traced tool operation
type ToolSpan struct {
	span      trace.Span
	startTime time.Time
}

// NewToolSpan creates a new tool span
func (i *Instrumentor) NewToolSpan(ctx context.Context, toolName string) (context.Context, *ToolSpan) {
	ctx, span := i.tracer.Start(ctx, fmt.Sprintf("tool.%s", toolName),
		trace.WithAttributes(attribute.String("tool.name", toolName)),
	)

	return ctx, &ToolSpan{
		span:      span,
		startTime: time.Now(),
	}
}

// SetArgs sets the tool arguments
func (s *ToolSpan) SetArgs(args map[string]interface{}) {
	// Don't include sensitive data in traces
	s.span.SetAttributes(attribute.Int("tool.args_count", len(args)))
}

// SetResult sets the tool result
func (s *ToolSpan) SetResult(success bool) {
	s.span.SetAttributes(attribute.Bool("tool.success", success))
}

// RecordError records an error on the span
func (s *ToolSpan) RecordError(err error) {
	s.span.RecordError(err)
}

// End ends the span
func (s *ToolSpan) End() {
	duration := time.Since(s.startTime)
	s.span.SetAttributes(attribute.Int64("tool.duration_ms", duration.Milliseconds()))
	s.span.End()
}

// LLMSpan represents a traced LLM operation
type LLMSpan struct {
	span      trace.Span
	startTime time.Time
}

// NewLLMSpan creates a new LLM span
func (i *Instrumentor) NewLLMSpan(ctx context.Context, model, provider string) (context.Context, *LLMSpan) {
	ctx, span := i.tracer.Start(ctx, "llm.completion",
		trace.WithAttributes(
			attribute.String("llm.model", model),
			attribute.String("llm.provider", provider),
		),
	)

	return ctx, &LLMSpan{
		span:      span,
		startTime: time.Now(),
	}
}

// SetTokens sets token counts
func (s *LLMSpan) SetTokens(input, output int) {
	s.span.SetAttributes(
		attribute.Int("llm.input_tokens", input),
		attribute.Int("llm.output_tokens", output),
		attribute.Int("llm.total_tokens", input+output),
	)
}

// SetToolCalls sets the number of tool calls
func (s *LLMSpan) SetToolCalls(count int) {
	s.span.SetAttributes(attribute.Int("llm.tool_calls", count))
}

// RecordError records an error on the span
func (s *LLMSpan) RecordError(err error) {
	s.span.RecordError(err)
}

// End ends the span
func (s *LLMSpan) End() {
	duration := time.Since(s.startTime)
	s.span.SetAttributes(attribute.Int64("llm.duration_ms", duration.Milliseconds()))
	s.span.End()
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// GlobalInstrumentor is the global instrumentor instance
var globalInstrumentor *Instrumentor

// InitGlobalInstrumentor initializes the global instrumentor
func InitGlobalInstrumentor(config *InstrumentorConfig) error {
	inst, err := NewInstrumentor(config)
	if err != nil {
		return err
	}
	globalInstrumentor = inst
	return nil
}

// GetGlobalInstrumentor returns the global instrumentor
func GetGlobalInstrumentor() *Instrumentor {
	return globalInstrumentor
}

// ShutdownGlobalInstrumentor shuts down the global instrumentor
func ShutdownGlobalInstrumentor(ctx context.Context) error {
	if globalInstrumentor != nil {
		return globalInstrumentor.Shutdown(ctx)
	}
	return nil
}
