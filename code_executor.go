package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// CodeExecutor executes code in various environments
type CodeExecutor interface {
	// Execute runs code and returns the result
	Execute(ctx context.Context, code string, language string) (*ExecutionResult, error)

	// ExecuteFile runs a file and returns the result
	ExecuteFile(ctx context.Context, filepath string) (*ExecutionResult, error)

	// Supported returns supported languages
	Supported() []string

	// Cleanup cleans up any resources
	Cleanup(ctx context.Context) error
}

// ExecutionResult represents the result of code execution
type ExecutionResult struct {
	// Output is the stdout from execution
	Output string `json:"output"`

	// Error is the stderr from execution
	Error string `json:"error"`

	// ExitCode is the exit code
	ExitCode int `json:"exit_code"`

	// Duration is how long execution took
	Duration time.Duration `json:"duration"`

	// Language is the language that was executed
	Language string `json:"language"`

	// Success indicates if execution was successful
	Success bool `json:"success"`

	// Files contains any output files generated
	Files map[string][]byte `json:"files,omitempty"`
}

// LanguageConfig configures a language for execution
type LanguageConfig struct {
	// Name is the language name
	Name string

	// Extensions are file extensions for this language
	Extensions []string

	// Command is the execution command
	Command string

	// Args are additional arguments
	Args []string

	// FileArg is how to pass the file (empty means as last arg)
	FileArg string

	// DockerImage is the Docker image for this language
	DockerImage string

	// Setup is any setup command to run first
	Setup string
}

// DefaultLanguages returns default language configurations
func DefaultLanguages() map[string]*LanguageConfig {
	return map[string]*LanguageConfig{
		"python": {
			Name:        "python",
			Extensions:  []string{".py"},
			Command:     "python3",
			DockerImage: "python:3.11-slim",
		},
		"python3": {
			Name:        "python",
			Extensions:  []string{".py"},
			Command:     "python3",
			DockerImage: "python:3.11-slim",
		},
		"javascript": {
			Name:        "javascript",
			Extensions:  []string{".js"},
			Command:     "node",
			DockerImage: "node:20-slim",
		},
		"node": {
			Name:        "javascript",
			Extensions:  []string{".js"},
			Command:     "node",
			DockerImage: "node:20-slim",
		},
		"typescript": {
			Name:        "typescript",
			Extensions:  []string{".ts"},
			Command:     "npx",
			Args:        []string{"ts-node"},
			DockerImage: "node:20-slim",
			Setup:       "npm install -g ts-node typescript",
		},
		"go": {
			Name:        "go",
			Extensions:  []string{".go"},
			Command:     "go",
			Args:        []string{"run"},
			DockerImage: "golang:1.21-alpine",
		},
		"golang": {
			Name:        "go",
			Extensions:  []string{".go"},
			Command:     "go",
			Args:        []string{"run"},
			DockerImage: "golang:1.21-alpine",
		},
		"ruby": {
			Name:        "ruby",
			Extensions:  []string{".rb"},
			Command:     "ruby",
			DockerImage: "ruby:3.2-slim",
		},
		"bash": {
			Name:        "bash",
			Extensions:  []string{".sh"},
			Command:     "bash",
			DockerImage: "bash:5",
		},
		"shell": {
			Name:        "bash",
			Extensions:  []string{".sh"},
			Command:     "bash",
			DockerImage: "bash:5",
		},
		"rust": {
			Name:        "rust",
			Extensions:  []string{".rs"},
			Command:     "rustc",
			Args:        []string{"-o", "/tmp/rust_out"},
			DockerImage: "rust:1.74-slim",
		},
		"java": {
			Name:        "java",
			Extensions:  []string{".java"},
			Command:     "java",
			DockerImage: "openjdk:21-slim",
		},
		"c": {
			Name:        "c",
			Extensions:  []string{".c"},
			Command:     "gcc",
			Args:        []string{"-o", "/tmp/c_out"},
			DockerImage: "gcc:13",
		},
		"cpp": {
			Name:        "cpp",
			Extensions:  []string{".cpp", ".cc"},
			Command:     "g++",
			Args:        []string{"-o", "/tmp/cpp_out"},
			DockerImage: "gcc:13",
		},
	}
}

// LocalCodeExecutor executes code locally
type LocalCodeExecutor struct {
	// WorkDir is the working directory
	WorkDir string

	// Languages are supported language configs
	Languages map[string]*LanguageConfig

	// Timeout is the default execution timeout
	Timeout time.Duration

	// AllowedLanguages restricts which languages can be executed
	AllowedLanguages []string

	// Environment variables to set
	Environment map[string]string

	mu sync.Mutex
}

// LocalCodeExecutorConfig configures the local executor
type LocalCodeExecutorConfig struct {
	WorkDir          string
	Timeout          time.Duration
	AllowedLanguages []string
	Environment      map[string]string
	CustomLanguages  map[string]*LanguageConfig
}

// NewLocalCodeExecutor creates a new local code executor
func NewLocalCodeExecutor(config *LocalCodeExecutorConfig) (*LocalCodeExecutor, error) {
	if config == nil {
		config = &LocalCodeExecutorConfig{}
	}

	workDir := config.WorkDir
	if workDir == "" {
		var err error
		workDir, err = os.MkdirTemp("", "code-executor-*")
		if err != nil {
			return nil, fmt.Errorf("failed to create work directory: %w", err)
		}
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	languages := DefaultLanguages()
	for name, lang := range config.CustomLanguages {
		languages[name] = lang
	}

	return &LocalCodeExecutor{
		WorkDir:          workDir,
		Languages:        languages,
		Timeout:          timeout,
		AllowedLanguages: config.AllowedLanguages,
		Environment:      config.Environment,
	}, nil
}

// Execute runs code locally
func (e *LocalCodeExecutor) Execute(ctx context.Context, code string, language string) (*ExecutionResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Validate language
	langConfig, ok := e.Languages[strings.ToLower(language)]
	if !ok {
		return nil, fmt.Errorf("unsupported language: %s", language)
	}

	// Check if language is allowed
	if len(e.AllowedLanguages) > 0 {
		allowed := false
		for _, l := range e.AllowedLanguages {
			if strings.EqualFold(l, language) || strings.EqualFold(l, langConfig.Name) {
				allowed = true
				break
			}
		}
		if !allowed {
			return nil, fmt.Errorf("language not allowed: %s", language)
		}
	}

	// Create temp file
	ext := ".txt"
	if len(langConfig.Extensions) > 0 {
		ext = langConfig.Extensions[0]
	}
	tmpFile, err := os.CreateTemp(e.WorkDir, fmt.Sprintf("code-*%s", ext))
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(code); err != nil {
		tmpFile.Close()
		return nil, fmt.Errorf("failed to write code: %w", err)
	}
	tmpFile.Close()

	return e.executeFile(ctx, tmpFile.Name(), langConfig)
}

// ExecuteFile runs a file locally
func (e *LocalCodeExecutor) ExecuteFile(ctx context.Context, filepath string) (*ExecutionResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Determine language from extension
	ext := strings.ToLower(strings.TrimPrefix(filepath, "."))
	var langConfig *LanguageConfig
	for _, lang := range e.Languages {
		for _, langExt := range lang.Extensions {
			if langExt == "."+ext || langExt == ext {
				langConfig = lang
				break
			}
		}
		if langConfig != nil {
			break
		}
	}

	if langConfig == nil {
		return nil, fmt.Errorf("could not determine language for file: %s", filepath)
	}

	return e.executeFile(ctx, filepath, langConfig)
}

func (e *LocalCodeExecutor) executeFile(ctx context.Context, filepath string, langConfig *LanguageConfig) (*ExecutionResult, error) {
	start := time.Now()

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, e.Timeout)
	defer cancel()

	// Build command
	args := append([]string{}, langConfig.Args...)
	if langConfig.FileArg != "" {
		args = append(args, langConfig.FileArg, filepath)
	} else {
		args = append(args, filepath)
	}

	cmd := exec.CommandContext(ctx, langConfig.Command, args...)
	cmd.Dir = e.WorkDir

	// Set environment
	cmd.Env = os.Environ()
	for k, v := range e.Environment {
		cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", k, v))
	}

	// Capture output
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Run command
	err := cmd.Run()

	result := &ExecutionResult{
		Output:   stdout.String(),
		Error:    stderr.String(),
		Duration: time.Since(start),
		Language: langConfig.Name,
	}

	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exitErr.ExitCode()
		} else {
			result.ExitCode = -1
			result.Error = err.Error()
		}
	} else {
		result.Success = true
	}

	return result, nil
}

// Supported returns supported languages
func (e *LocalCodeExecutor) Supported() []string {
	seen := make(map[string]bool)
	var languages []string
	for _, lang := range e.Languages {
		if !seen[lang.Name] {
			seen[lang.Name] = true
			languages = append(languages, lang.Name)
		}
	}
	return languages
}

// Cleanup cleans up resources
func (e *LocalCodeExecutor) Cleanup(ctx context.Context) error {
	return os.RemoveAll(e.WorkDir)
}

// DockerCodeExecutor executes code in Docker containers
type DockerCodeExecutor struct {
	// Languages are supported language configs
	Languages map[string]*LanguageConfig

	// Timeout is the default execution timeout
	Timeout time.Duration

	// AllowedLanguages restricts which languages can be executed
	AllowedLanguages []string

	// Network is the Docker network to use
	Network string

	// Memory limit (e.g., "256m")
	MemoryLimit string

	// CPU limit (e.g., "1.0")
	CPULimit string

	// EnableNetwork enables network in containers
	EnableNetwork bool

	// Volumes are additional volumes to mount
	Volumes []string

	mu sync.Mutex
}

// DockerCodeExecutorConfig configures the Docker executor
type DockerCodeExecutorConfig struct {
	Timeout          time.Duration
	AllowedLanguages []string
	Network          string
	MemoryLimit      string
	CPULimit         string
	EnableNetwork    bool
	Volumes          []string
	CustomLanguages  map[string]*LanguageConfig
}

// NewDockerCodeExecutor creates a new Docker code executor
func NewDockerCodeExecutor(config *DockerCodeExecutorConfig) (*DockerCodeExecutor, error) {
	if config == nil {
		config = &DockerCodeExecutorConfig{}
	}

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 60 * time.Second
	}

	memLimit := config.MemoryLimit
	if memLimit == "" {
		memLimit = "256m"
	}

	cpuLimit := config.CPULimit
	if cpuLimit == "" {
		cpuLimit = "1.0"
	}

	languages := DefaultLanguages()
	for name, lang := range config.CustomLanguages {
		languages[name] = lang
	}

	return &DockerCodeExecutor{
		Languages:        languages,
		Timeout:          timeout,
		AllowedLanguages: config.AllowedLanguages,
		Network:          config.Network,
		MemoryLimit:      memLimit,
		CPULimit:         cpuLimit,
		EnableNetwork:    config.EnableNetwork,
		Volumes:          config.Volumes,
	}, nil
}

// Execute runs code in a Docker container
func (e *DockerCodeExecutor) Execute(ctx context.Context, code string, language string) (*ExecutionResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Validate language
	langConfig, ok := e.Languages[strings.ToLower(language)]
	if !ok {
		return nil, fmt.Errorf("unsupported language: %s", language)
	}

	// Check if language is allowed
	if len(e.AllowedLanguages) > 0 {
		allowed := false
		for _, l := range e.AllowedLanguages {
			if strings.EqualFold(l, language) || strings.EqualFold(l, langConfig.Name) {
				allowed = true
				break
			}
		}
		if !allowed {
			return nil, fmt.Errorf("language not allowed: %s", language)
		}
	}

	// Check Docker image
	if langConfig.DockerImage == "" {
		return nil, fmt.Errorf("no Docker image configured for language: %s", language)
	}

	return e.executeInDocker(ctx, code, langConfig)
}

// ExecuteFile runs a file in a Docker container
func (e *DockerCodeExecutor) ExecuteFile(ctx context.Context, filepath string) (*ExecutionResult, error) {
	// Read file content
	content, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Determine language from extension
	ext := strings.ToLower(strings.TrimPrefix(filepath, "."))
	var langConfig *LanguageConfig
	for _, lang := range e.Languages {
		for _, langExt := range lang.Extensions {
			if langExt == "."+ext || langExt == ext {
				langConfig = lang
				break
			}
		}
		if langConfig != nil {
			break
		}
	}

	if langConfig == nil {
		return nil, fmt.Errorf("could not determine language for file: %s", filepath)
	}

	return e.executeInDocker(ctx, string(content), langConfig)
}

func (e *DockerCodeExecutor) executeInDocker(ctx context.Context, code string, langConfig *LanguageConfig) (*ExecutionResult, error) {
	start := time.Now()

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, e.Timeout)
	defer cancel()

	// Create temp directory for code
	tmpDir, err := os.MkdirTemp("", "docker-code-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// Write code to file
	ext := ".txt"
	if len(langConfig.Extensions) > 0 {
		ext = langConfig.Extensions[0]
	}
	codeFile := filepath.Join(tmpDir, "code"+ext)
	if err := os.WriteFile(codeFile, []byte(code), 0644); err != nil {
		return nil, fmt.Errorf("failed to write code file: %w", err)
	}

	// Build docker command
	args := []string{
		"run",
		"--rm",
		"-m", e.MemoryLimit,
		"--cpus", e.CPULimit,
		"-v", fmt.Sprintf("%s:/code:ro", tmpDir),
		"-w", "/code",
	}

	if !e.EnableNetwork {
		args = append(args, "--network", "none")
	} else if e.Network != "" {
		args = append(args, "--network", e.Network)
	}

	for _, vol := range e.Volumes {
		args = append(args, "-v", vol)
	}

	args = append(args, langConfig.DockerImage)

	// Add execution command
	if langConfig.Setup != "" {
		// If setup is needed, use sh -c
		execCmd := fmt.Sprintf("%s && %s", langConfig.Setup, buildExecCommand(langConfig, "/code/code"+ext))
		args = append(args, "sh", "-c", execCmd)
	} else {
		args = append(args, langConfig.Command)
		args = append(args, langConfig.Args...)
		args = append(args, "/code/code"+ext)
	}

	cmd := exec.CommandContext(ctx, "docker", args...)

	// Capture output
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Run command
	err = cmd.Run()

	result := &ExecutionResult{
		Output:   stdout.String(),
		Error:    stderr.String(),
		Duration: time.Since(start),
		Language: langConfig.Name,
	}

	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exitErr.ExitCode()
		} else {
			result.ExitCode = -1
			result.Error = err.Error()
		}
	} else {
		result.Success = true
	}

	return result, nil
}

func buildExecCommand(langConfig *LanguageConfig, filepath string) string {
	parts := []string{langConfig.Command}
	parts = append(parts, langConfig.Args...)
	parts = append(parts, filepath)
	return strings.Join(parts, " ")
}

// Supported returns supported languages
func (e *DockerCodeExecutor) Supported() []string {
	seen := make(map[string]bool)
	var languages []string
	for _, lang := range e.Languages {
		if lang.DockerImage != "" && !seen[lang.Name] {
			seen[lang.Name] = true
			languages = append(languages, lang.Name)
		}
	}
	return languages
}

// Cleanup cleans up resources
func (e *DockerCodeExecutor) Cleanup(ctx context.Context) error {
	// Docker containers are removed with --rm flag
	return nil
}

// SandboxedCodeExecutor wraps a code executor with additional security
type SandboxedCodeExecutor struct {
	executor     CodeExecutor
	maxCodeSize  int
	maxOutputLen int
	blockedCode  []string
}

// SandboxConfig configures the sandbox
type SandboxConfig struct {
	Executor     CodeExecutor
	MaxCodeSize  int
	MaxOutputLen int
	BlockedCode  []string
}

// NewSandboxedCodeExecutor creates a sandboxed executor
func NewSandboxedCodeExecutor(config *SandboxConfig) *SandboxedCodeExecutor {
	if config.MaxCodeSize == 0 {
		config.MaxCodeSize = 1024 * 1024 // 1MB
	}
	if config.MaxOutputLen == 0 {
		config.MaxOutputLen = 1024 * 1024 // 1MB
	}

	return &SandboxedCodeExecutor{
		executor:     config.Executor,
		maxCodeSize:  config.MaxCodeSize,
		maxOutputLen: config.MaxOutputLen,
		blockedCode:  config.BlockedCode,
	}
}

// Execute runs code with sandbox restrictions
func (s *SandboxedCodeExecutor) Execute(ctx context.Context, code string, language string) (*ExecutionResult, error) {
	// Check code size
	if len(code) > s.maxCodeSize {
		return nil, fmt.Errorf("code exceeds maximum size of %d bytes", s.maxCodeSize)
	}

	// Check for blocked patterns
	for _, blocked := range s.blockedCode {
		if strings.Contains(code, blocked) {
			return nil, fmt.Errorf("code contains blocked pattern: %s", blocked)
		}
	}

	// Execute
	result, err := s.executor.Execute(ctx, code, language)
	if err != nil {
		return nil, err
	}

	// Truncate output if needed
	if len(result.Output) > s.maxOutputLen {
		result.Output = result.Output[:s.maxOutputLen] + "\n... (output truncated)"
	}
	if len(result.Error) > s.maxOutputLen {
		result.Error = result.Error[:s.maxOutputLen] + "\n... (error truncated)"
	}

	return result, nil
}

// ExecuteFile runs a file with sandbox restrictions
func (s *SandboxedCodeExecutor) ExecuteFile(ctx context.Context, filepath string) (*ExecutionResult, error) {
	// Read file to check size and content
	content, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	if len(content) > s.maxCodeSize {
		return nil, fmt.Errorf("file exceeds maximum size of %d bytes", s.maxCodeSize)
	}

	for _, blocked := range s.blockedCode {
		if strings.Contains(string(content), blocked) {
			return nil, fmt.Errorf("file contains blocked pattern: %s", blocked)
		}
	}

	return s.executor.ExecuteFile(ctx, filepath)
}

// Supported returns supported languages
func (s *SandboxedCodeExecutor) Supported() []string {
	return s.executor.Supported()
}

// Cleanup cleans up resources
func (s *SandboxedCodeExecutor) Cleanup(ctx context.Context) error {
	return s.executor.Cleanup(ctx)
}

// CodeExecutorTool creates a tool from a code executor
type CodeExecutorTool struct {
	executor CodeExecutor
	name     string
}

// NewCodeExecutorTool creates a new code executor tool
func NewCodeExecutorTool(executor CodeExecutor, name string) *CodeExecutorTool {
	if name == "" {
		name = "execute_code"
	}
	return &CodeExecutorTool{
		executor: executor,
		name:     name,
	}
}

// Name returns the tool name
func (t *CodeExecutorTool) Name() string {
	return t.name
}

// Description returns the tool description
func (t *CodeExecutorTool) Description() string {
	return fmt.Sprintf("Execute code in supported languages: %s", strings.Join(t.executor.Supported(), ", "))
}

// Schema returns the tool schema
func (t *CodeExecutorTool) Schema() *ToolSchema {
	return &ToolSchema{
		Type:        "object",
		Description: t.Description(),
		Properties: map[string]PropertySchema{
			"code": {
				Type:        "string",
				Description: "The code to execute",
			},
			"language": {
				Type:        "string",
				Description: "The programming language",
				Enum:        t.executor.Supported(),
			},
		},
		Required: []string{"code", "language"},
	}
}

// Execute runs the tool
func (t *CodeExecutorTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	code, ok := args["code"].(string)
	if !ok {
		return nil, fmt.Errorf("code must be a string")
	}

	language, ok := args["language"].(string)
	if !ok {
		return nil, fmt.Errorf("language must be a string")
	}

	result, err := t.executor.Execute(ctx, code, language)
	if err != nil {
		return nil, err
	}

	// Return as JSON-friendly map
	return map[string]interface{}{
		"output":    result.Output,
		"error":     result.Error,
		"exit_code": result.ExitCode,
		"success":   result.Success,
		"duration":  result.Duration.String(),
		"language":  result.Language,
	}, nil
}

// JupyterKernelExecutor executes code through Jupyter kernels
type JupyterKernelExecutor struct {
	// KernelManager manages Jupyter kernels
	endpoint string
	client   interface{} // HTTP client for Jupyter API
	kernels  map[string]string
	mu       sync.Mutex
}

// JupyterKernelConfig configures the Jupyter executor
type JupyterKernelConfig struct {
	Endpoint string // e.g., "http://localhost:8888"
	Token    string
}

// NewJupyterKernelExecutor creates a new Jupyter kernel executor
func NewJupyterKernelExecutor(config *JupyterKernelConfig) (*JupyterKernelExecutor, error) {
	return &JupyterKernelExecutor{
		endpoint: config.Endpoint,
		kernels:  make(map[string]string),
	}, nil
}

// Execute runs code through Jupyter kernel
func (e *JupyterKernelExecutor) Execute(ctx context.Context, code string, language string) (*ExecutionResult, error) {
	// This is a stub - full implementation would use Jupyter kernel protocol
	return nil, fmt.Errorf("Jupyter kernel execution not yet implemented")
}

// ExecuteFile runs a file through Jupyter kernel
func (e *JupyterKernelExecutor) ExecuteFile(ctx context.Context, filepath string) (*ExecutionResult, error) {
	content, err := os.ReadFile(filepath)
	if err != nil {
		return nil, err
	}
	return e.Execute(ctx, string(content), "python")
}

// Supported returns supported languages
func (e *JupyterKernelExecutor) Supported() []string {
	return []string{"python", "python3"}
}

// Cleanup cleans up resources
func (e *JupyterKernelExecutor) Cleanup(ctx context.Context) error {
	// Shutdown kernels
	return nil
}

// REPLExecutor provides an interactive REPL experience
type REPLExecutor struct {
	executor    CodeExecutor
	sessionCode map[string][]string
	mu          sync.Mutex
}

// NewREPLExecutor creates a new REPL executor
func NewREPLExecutor(executor CodeExecutor) *REPLExecutor {
	return &REPLExecutor{
		executor:    executor,
		sessionCode: make(map[string][]string),
	}
}

// Execute runs code in REPL mode, maintaining session state
func (r *REPLExecutor) Execute(ctx context.Context, sessionID, code, language string) (*ExecutionResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Get existing session code
	existingCode := r.sessionCode[sessionID]

	// Combine with new code
	allCode := strings.Join(append(existingCode, code), "\n")

	// Execute combined code
	result, err := r.executor.Execute(ctx, allCode, language)
	if err != nil {
		return nil, err
	}

	// If successful, add code to session
	if result.Success {
		r.sessionCode[sessionID] = append(existingCode, code)
	}

	return result, nil
}

// Reset resets a session
func (r *REPLExecutor) Reset(sessionID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.sessionCode, sessionID)
}

// GetSessionCode returns all code in a session
func (r *REPLExecutor) GetSessionCode(sessionID string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return strings.Join(r.sessionCode[sessionID], "\n")
}

// ExecuteFile is not supported in REPL mode
func (r *REPLExecutor) ExecuteFile(ctx context.Context, filepath string) (*ExecutionResult, error) {
	return nil, fmt.Errorf("ExecuteFile not supported in REPL mode")
}

// Supported returns supported languages
func (r *REPLExecutor) Supported() []string {
	return r.executor.Supported()
}

// Cleanup cleans up resources
func (r *REPLExecutor) Cleanup(ctx context.Context) error {
	r.mu.Lock()
	r.sessionCode = make(map[string][]string)
	r.mu.Unlock()
	return r.executor.Cleanup(ctx)
}

// MultiExecutorResult contains results from multiple executors
type MultiExecutorResult struct {
	Results map[string]*ExecutionResult `json:"results"`
}

// ExecutionOutputParser parses execution output
type ExecutionOutputParser struct{}

// ParsePythonOutput parses Python execution output
func (p *ExecutionOutputParser) ParsePythonOutput(output string) (interface{}, error) {
	// Try to parse as JSON
	var result interface{}
	if err := json.Unmarshal([]byte(strings.TrimSpace(output)), &result); err == nil {
		return result, nil
	}
	return output, nil
}

// CodeBlock represents a code block from a message
type CodeBlock struct {
	Language string
	Code     string
}

// ExtractCodeBlocks extracts code blocks from markdown-style text
func ExtractCodeBlocks(text string) []CodeBlock {
	var blocks []CodeBlock

	lines := strings.Split(text, "\n")
	inBlock := false
	var currentBlock CodeBlock
	var codeLines []string

	for _, line := range lines {
		if strings.HasPrefix(line, "```") {
			if inBlock {
				// End of block
				currentBlock.Code = strings.Join(codeLines, "\n")
				blocks = append(blocks, currentBlock)
				inBlock = false
				codeLines = nil
			} else {
				// Start of block
				inBlock = true
				currentBlock = CodeBlock{
					Language: strings.TrimPrefix(line, "```"),
				}
			}
		} else if inBlock {
			codeLines = append(codeLines, line)
		}
	}

	return blocks
}

// ExecuteCodeBlocks executes all code blocks from text
func ExecuteCodeBlocks(ctx context.Context, executor CodeExecutor, text string) ([]*ExecutionResult, error) {
	blocks := ExtractCodeBlocks(text)
	var results []*ExecutionResult

	for _, block := range blocks {
		if block.Language == "" {
			continue
		}

		result, err := executor.Execute(ctx, block.Code, block.Language)
		if err != nil {
			// Continue with other blocks even if one fails
			results = append(results, &ExecutionResult{
				Error:    err.Error(),
				Language: block.Language,
			})
			continue
		}
		results = append(results, result)
	}

	return results, nil
}

// StreamingCodeExecutor provides streaming output during execution
type StreamingCodeExecutor struct {
	executor CodeExecutor
}

// NewStreamingCodeExecutor creates a new streaming executor
func NewStreamingCodeExecutor(executor CodeExecutor) *StreamingCodeExecutor {
	return &StreamingCodeExecutor{executor: executor}
}

// ExecuteWithStream executes code and streams output to a writer
func (s *StreamingCodeExecutor) ExecuteWithStream(ctx context.Context, code, language string, stdout, stderr io.Writer) (*ExecutionResult, error) {
	// For now, delegate to the underlying executor
	// A full implementation would stream output in real-time
	result, err := s.executor.Execute(ctx, code, language)
	if err != nil {
		return nil, err
	}

	if stdout != nil && result.Output != "" {
		stdout.Write([]byte(result.Output))
	}
	if stderr != nil && result.Error != "" {
		stderr.Write([]byte(result.Error))
	}

	return result, nil
}
