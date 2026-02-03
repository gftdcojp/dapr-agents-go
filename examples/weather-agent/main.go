// Example: Weather Agent
//
// This example demonstrates a simple Dapr Agent that can check weather
// using custom tools. It exposes the agent via gRPC, HTTP, and MCP.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	agent "github.com/gftdcojp/dapr-agents-go"
)

func main() {
	// Configure the agent
	config := &agent.AgentConfig{
		Name:            "WeatherAgent",
		LLMComponent:    getEnv("LLM_COMPONENT", "llm"),
		LLMModel:        getEnv("LLM_MODEL", "gpt-4-turbo"),
		MemoryStore:     "conversation-statestore",
		WorkflowStore:   "workflow-statestore",
		MaxSteps:        10,
		StepTimeout:     30 * time.Second,
		MemoryMaxTokens: 4000,
		SystemPrompt: `You are a helpful weather assistant. You can check the current weather
and forecast for any city using the available tools. Always be polite and provide accurate information.
When asked about weather, use the appropriate tool to get the data.`,
	}

	// Create tools
	weatherTool := agent.NewToolBuilder("get_weather").
		Description("Get the current weather for a city").
		AddStringParam("city", "The city name (e.g., 'Tokyo', 'New York')", true).
		AddEnumParam("units", "Temperature units", []string{"celsius", "fahrenheit"}, false).
		BuildFunc(getWeather)

	forecastTool := agent.NewToolBuilder("get_forecast").
		Description("Get the weather forecast for a city").
		AddStringParam("city", "The city name", true).
		AddNumberParam("days", "Number of days (1-7)", true).
		BuildFunc(getForecast)

	// Create the agent factory
	agentFactory := func() agent.Agent {
		a := agent.NewBaseAgent(config)
		a.RegisterTool(weatherTool)
		a.RegisterTool(forecastTool)
		return a
	}

	// Create server
	server := agent.NewServer(&agent.ServerConfig{
		Port:                getEnv("APP_PORT", "50051"),
		Protocol:            "grpc",
		EnableHTTPEndpoints: true,
		HTTPPort:            getEnv("HTTP_PORT", "8080"),
	})
	server.RegisterAgent(agentFactory)

	// Create MCP server
	mcpServer := agent.NewMCPServer(&agent.MCPServerConfig{
		Port:        8081,
		Name:        "weather-agent-mcp",
		Version:     "1.0.0",
		Description: "Weather Agent - Get weather information for any city",
	})
	mcpServer.RegisterAgent("WeatherAgent", agentFactory)
	mcpServer.RegisterTool(weatherTool)
	mcpServer.RegisterTool(forecastTool)

	// Start MCP server in background
	go func() {
		log.Printf("MCP server starting on :8081")
		if err := mcpServer.Start(); err != nil {
			log.Printf("MCP server error: %v", err)
		}
	}()

	// Start main server
	log.Printf("Weather Agent starting on :%s (gRPC), :%s (HTTP)",
		getEnv("APP_PORT", "50051"), getEnv("HTTP_PORT", "8080"))
	if err := server.Start(); err != nil {
		log.Fatal(err)
	}
}

func getWeather(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	city, ok := params["city"].(string)
	if !ok || city == "" {
		return nil, fmt.Errorf("city is required")
	}

	units := "celsius"
	if u, ok := params["units"].(string); ok {
		units = u
	}

	// In production, call a real weather API
	temp := 22
	if units == "fahrenheit" {
		temp = 72
	}

	return map[string]interface{}{
		"city":        city,
		"temperature": temp,
		"units":       units,
		"condition":   "partly cloudy",
		"humidity":    65,
		"wind_speed":  12,
		"timestamp":   time.Now().Format(time.RFC3339),
	}, nil
}

func getForecast(ctx context.Context, params map[string]interface{}) (interface{}, error) {
	city, ok := params["city"].(string)
	if !ok || city == "" {
		return nil, fmt.Errorf("city is required")
	}

	days := 3
	if d, ok := params["days"].(float64); ok {
		days = int(d)
		if days < 1 || days > 7 {
			days = 3
		}
	}

	forecast := make([]map[string]interface{}, days)
	conditions := []string{"sunny", "partly cloudy", "cloudy", "rainy"}

	for i := 0; i < days; i++ {
		date := time.Now().AddDate(0, 0, i+1)
		forecast[i] = map[string]interface{}{
			"date":      date.Format("2006-01-02"),
			"high":      22 + (i % 5),
			"low":       15 - (i % 3),
			"condition": conditions[i%len(conditions)],
		}
	}

	return map[string]interface{}{
		"city":     city,
		"forecast": forecast,
	}, nil
}

func getEnv(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}
