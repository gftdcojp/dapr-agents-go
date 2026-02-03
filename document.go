package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"golang.org/x/net/html"
)

// Document represents a processed document
type Document struct {
	// ID is the document identifier
	ID string `json:"id"`

	// Content is the document text content
	Content string `json:"content"`

	// Metadata contains document metadata
	Metadata map[string]interface{} `json:"metadata,omitempty"`

	// Source is the document source (file path, URL, etc.)
	Source string `json:"source,omitempty"`

	// MimeType is the document MIME type
	MimeType string `json:"mime_type,omitempty"`

	// Chunks are document chunks if split
	Chunks []*DocumentChunk `json:"chunks,omitempty"`

	// Embeddings are document embeddings if computed
	Embeddings []float64 `json:"embeddings,omitempty"`

	// CreatedAt is when the document was created
	CreatedAt time.Time `json:"created_at"`
}

// DocumentChunk represents a chunk of a document
type DocumentChunk struct {
	// ID is the chunk identifier
	ID string `json:"id"`

	// Content is the chunk text content
	Content string `json:"content"`

	// Index is the chunk index in the document
	Index int `json:"index"`

	// Metadata contains chunk-specific metadata
	Metadata map[string]interface{} `json:"metadata,omitempty"`

	// Embeddings are chunk embeddings
	Embeddings []float64 `json:"embeddings,omitempty"`

	// StartOffset is the start offset in the original document
	StartOffset int `json:"start_offset,omitempty"`

	// EndOffset is the end offset in the original document
	EndOffset int `json:"end_offset,omitempty"`
}

// DocumentLoader loads documents from various sources
type DocumentLoader interface {
	// Load loads documents from a source
	Load(ctx context.Context, source string) ([]*Document, error)

	// SupportedTypes returns supported MIME types or file extensions
	SupportedTypes() []string
}

// TextDocumentLoader loads plain text documents
type TextDocumentLoader struct {
	// Encoding is the text encoding (default UTF-8)
	Encoding string
}

// NewTextDocumentLoader creates a new text document loader
func NewTextDocumentLoader() *TextDocumentLoader {
	return &TextDocumentLoader{Encoding: "utf-8"}
}

// Load loads a text document
func (l *TextDocumentLoader) Load(ctx context.Context, source string) ([]*Document, error) {
	content, err := os.ReadFile(source)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	if !utf8.Valid(content) {
		return nil, fmt.Errorf("file is not valid UTF-8")
	}

	doc := &Document{
		ID:        filepath.Base(source),
		Content:   string(content),
		Source:    source,
		MimeType:  "text/plain",
		CreatedAt: time.Now(),
		Metadata: map[string]interface{}{
			"filename": filepath.Base(source),
			"size":     len(content),
		},
	}

	return []*Document{doc}, nil
}

// SupportedTypes returns supported types
func (l *TextDocumentLoader) SupportedTypes() []string {
	return []string{".txt", ".text", "text/plain"}
}

// MarkdownDocumentLoader loads Markdown documents
type MarkdownDocumentLoader struct{}

// NewMarkdownDocumentLoader creates a new Markdown loader
func NewMarkdownDocumentLoader() *MarkdownDocumentLoader {
	return &MarkdownDocumentLoader{}
}

// Load loads a Markdown document
func (l *MarkdownDocumentLoader) Load(ctx context.Context, source string) ([]*Document, error) {
	content, err := os.ReadFile(source)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Extract metadata from frontmatter if present
	text := string(content)
	metadata := make(map[string]interface{})

	if strings.HasPrefix(text, "---") {
		parts := strings.SplitN(text[3:], "---", 2)
		if len(parts) == 2 {
			// Parse YAML frontmatter (simplified)
			frontmatter := strings.TrimSpace(parts[0])
			for _, line := range strings.Split(frontmatter, "\n") {
				if idx := strings.Index(line, ":"); idx > 0 {
					key := strings.TrimSpace(line[:idx])
					value := strings.TrimSpace(line[idx+1:])
					metadata[key] = value
				}
			}
			text = strings.TrimSpace(parts[1])
		}
	}

	metadata["filename"] = filepath.Base(source)
	metadata["size"] = len(content)

	doc := &Document{
		ID:        filepath.Base(source),
		Content:   text,
		Source:    source,
		MimeType:  "text/markdown",
		Metadata:  metadata,
		CreatedAt: time.Now(),
	}

	return []*Document{doc}, nil
}

// SupportedTypes returns supported types
func (l *MarkdownDocumentLoader) SupportedTypes() []string {
	return []string{".md", ".markdown", "text/markdown"}
}

// HTMLDocumentLoader loads HTML documents
type HTMLDocumentLoader struct {
	// ExtractText extracts plain text from HTML
	ExtractText bool
}

// NewHTMLDocumentLoader creates a new HTML loader
func NewHTMLDocumentLoader(extractText bool) *HTMLDocumentLoader {
	return &HTMLDocumentLoader{ExtractText: extractText}
}

// Load loads an HTML document
func (l *HTMLDocumentLoader) Load(ctx context.Context, source string) ([]*Document, error) {
	var content []byte
	var err error

	// Check if source is URL or file
	if strings.HasPrefix(source, "http://") || strings.HasPrefix(source, "https://") {
		resp, err := http.Get(source)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch URL: %w", err)
		}
		defer resp.Body.Close()
		content, err = io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response: %w", err)
		}
	} else {
		content, err = os.ReadFile(source)
		if err != nil {
			return nil, fmt.Errorf("failed to read file: %w", err)
		}
	}

	text := string(content)
	metadata := make(map[string]interface{})

	if l.ExtractText {
		// Extract plain text from HTML
		text, err = extractTextFromHTML(content)
		if err != nil {
			return nil, fmt.Errorf("failed to extract text: %w", err)
		}
		metadata["original_size"] = len(content)
	}

	metadata["source"] = source
	metadata["size"] = len(text)

	doc := &Document{
		ID:        filepath.Base(source),
		Content:   text,
		Source:    source,
		MimeType:  "text/html",
		Metadata:  metadata,
		CreatedAt: time.Now(),
	}

	return []*Document{doc}, nil
}

// extractTextFromHTML extracts plain text from HTML content
func extractTextFromHTML(content []byte) (string, error) {
	doc, err := html.Parse(bytes.NewReader(content))
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	var extract func(*html.Node)
	extract = func(n *html.Node) {
		if n.Type == html.TextNode {
			text := strings.TrimSpace(n.Data)
			if text != "" {
				buf.WriteString(text)
				buf.WriteString(" ")
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			// Skip script and style tags
			if c.Type == html.ElementNode && (c.Data == "script" || c.Data == "style") {
				continue
			}
			extract(c)
		}
	}
	extract(doc)

	return strings.TrimSpace(buf.String()), nil
}

// SupportedTypes returns supported types
func (l *HTMLDocumentLoader) SupportedTypes() []string {
	return []string{".html", ".htm", "text/html"}
}

// JSONDocumentLoader loads JSON documents
type JSONDocumentLoader struct {
	// ContentField is the field to extract as content
	ContentField string
}

// NewJSONDocumentLoader creates a new JSON loader
func NewJSONDocumentLoader(contentField string) *JSONDocumentLoader {
	return &JSONDocumentLoader{ContentField: contentField}
}

// Load loads a JSON document
func (l *JSONDocumentLoader) Load(ctx context.Context, source string) ([]*Document, error) {
	content, err := os.ReadFile(source)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	var data interface{}
	if err := json.Unmarshal(content, &data); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Handle array of documents
	if arr, ok := data.([]interface{}); ok {
		var docs []*Document
		for i, item := range arr {
			doc, err := l.createDocument(item, source, i)
			if err != nil {
				return nil, err
			}
			docs = append(docs, doc)
		}
		return docs, nil
	}

	// Handle single document
	doc, err := l.createDocument(data, source, 0)
	if err != nil {
		return nil, err
	}
	return []*Document{doc}, nil
}

func (l *JSONDocumentLoader) createDocument(data interface{}, source string, index int) (*Document, error) {
	var content string
	metadata := make(map[string]interface{})

	if obj, ok := data.(map[string]interface{}); ok {
		// Extract content from specified field or stringify
		if l.ContentField != "" {
			if val, exists := obj[l.ContentField]; exists {
				content = fmt.Sprintf("%v", val)
			}
		}
		if content == "" {
			jsonBytes, _ := json.Marshal(obj)
			content = string(jsonBytes)
		}
		// Use remaining fields as metadata
		for k, v := range obj {
			if k != l.ContentField {
				metadata[k] = v
			}
		}
	} else {
		content = fmt.Sprintf("%v", data)
	}

	return &Document{
		ID:        fmt.Sprintf("%s_%d", filepath.Base(source), index),
		Content:   content,
		Source:    source,
		MimeType:  "application/json",
		Metadata:  metadata,
		CreatedAt: time.Now(),
	}, nil
}

// SupportedTypes returns supported types
func (l *JSONDocumentLoader) SupportedTypes() []string {
	return []string{".json", "application/json"}
}

// CSVDocumentLoader loads CSV documents
type CSVDocumentLoader struct {
	// Delimiter is the field delimiter
	Delimiter rune
	// ContentColumns are columns to include in content
	ContentColumns []string
	// CreatePerRow creates one document per row
	CreatePerRow bool
}

// NewCSVDocumentLoader creates a new CSV loader
func NewCSVDocumentLoader(createPerRow bool) *CSVDocumentLoader {
	return &CSVDocumentLoader{
		Delimiter:    ',',
		CreatePerRow: createPerRow,
	}
}

// Load loads a CSV document
func (l *CSVDocumentLoader) Load(ctx context.Context, source string) ([]*Document, error) {
	content, err := os.ReadFile(source)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	lines := strings.Split(string(content), "\n")
	if len(lines) == 0 {
		return nil, fmt.Errorf("empty CSV file")
	}

	// Parse header
	header := strings.Split(lines[0], string(l.Delimiter))
	for i := range header {
		header[i] = strings.TrimSpace(header[i])
	}

	if !l.CreatePerRow {
		// Return single document with full CSV content
		return []*Document{{
			ID:       filepath.Base(source),
			Content:  string(content),
			Source:   source,
			MimeType: "text/csv",
			Metadata: map[string]interface{}{
				"columns":   header,
				"row_count": len(lines) - 1,
			},
			CreatedAt: time.Now(),
		}}, nil
	}

	// Create one document per row
	var docs []*Document
	for i, line := range lines[1:] {
		if strings.TrimSpace(line) == "" {
			continue
		}

		fields := strings.Split(line, string(l.Delimiter))
		metadata := make(map[string]interface{})
		var contentParts []string

		for j, field := range fields {
			field = strings.TrimSpace(field)
			if j < len(header) {
				metadata[header[j]] = field

				// Include in content if no specific columns specified or if column is specified
				if len(l.ContentColumns) == 0 || containsString(l.ContentColumns, header[j]) {
					contentParts = append(contentParts, fmt.Sprintf("%s: %s", header[j], field))
				}
			}
		}

		docs = append(docs, &Document{
			ID:        fmt.Sprintf("%s_row%d", filepath.Base(source), i+1),
			Content:   strings.Join(contentParts, "\n"),
			Source:    source,
			MimeType:  "text/csv",
			Metadata:  metadata,
			CreatedAt: time.Now(),
		})
	}

	return docs, nil
}

func containsString(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// SupportedTypes returns supported types
func (l *CSVDocumentLoader) SupportedTypes() []string {
	return []string{".csv", "text/csv"}
}

// PDFDocumentLoader loads PDF documents (stub - requires external library)
type PDFDocumentLoader struct{}

// NewPDFDocumentLoader creates a new PDF loader
func NewPDFDocumentLoader() *PDFDocumentLoader {
	return &PDFDocumentLoader{}
}

// Load loads a PDF document
func (l *PDFDocumentLoader) Load(ctx context.Context, source string) ([]*Document, error) {
	// This is a stub - full implementation would use a PDF parsing library
	// such as pdfcpu, gopdf, or call external tools like pdftotext
	return nil, fmt.Errorf("PDF loading not implemented - requires external PDF library")
}

// SupportedTypes returns supported types
func (l *PDFDocumentLoader) SupportedTypes() []string {
	return []string{".pdf", "application/pdf"}
}

// DirectoryLoader loads all documents from a directory
type DirectoryLoader struct {
	// Loaders maps extensions to loaders
	Loaders map[string]DocumentLoader
	// Recursive enables recursive directory traversal
	Recursive bool
	// FilePattern is a glob pattern for files
	FilePattern string
	// ExcludePatterns are patterns to exclude
	ExcludePatterns []string
}

// NewDirectoryLoader creates a new directory loader
func NewDirectoryLoader(recursive bool) *DirectoryLoader {
	return &DirectoryLoader{
		Loaders: map[string]DocumentLoader{
			".txt":      NewTextDocumentLoader(),
			".md":       NewMarkdownDocumentLoader(),
			".markdown": NewMarkdownDocumentLoader(),
			".html":     NewHTMLDocumentLoader(true),
			".htm":      NewHTMLDocumentLoader(true),
			".json":     NewJSONDocumentLoader("content"),
			".csv":      NewCSVDocumentLoader(false),
		},
		Recursive: recursive,
	}
}

// RegisterLoader registers a loader for an extension
func (l *DirectoryLoader) RegisterLoader(ext string, loader DocumentLoader) {
	l.Loaders[ext] = loader
}

// Load loads all documents from a directory
func (l *DirectoryLoader) Load(ctx context.Context, source string) ([]*Document, error) {
	var docs []*Document

	walkFn := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories unless recursive
		if info.IsDir() {
			if !l.Recursive && path != source {
				return filepath.SkipDir
			}
			return nil
		}

		// Check exclude patterns
		for _, pattern := range l.ExcludePatterns {
			matched, _ := filepath.Match(pattern, filepath.Base(path))
			if matched {
				return nil
			}
		}

		// Check file pattern
		if l.FilePattern != "" {
			matched, _ := filepath.Match(l.FilePattern, filepath.Base(path))
			if !matched {
				return nil
			}
		}

		// Find appropriate loader
		ext := strings.ToLower(filepath.Ext(path))
		loader, ok := l.Loaders[ext]
		if !ok {
			return nil // Skip unsupported files
		}

		// Load document
		loadedDocs, err := loader.Load(ctx, path)
		if err != nil {
			return nil // Skip files that fail to load
		}

		docs = append(docs, loadedDocs...)
		return nil
	}

	if err := filepath.Walk(source, walkFn); err != nil {
		return nil, fmt.Errorf("failed to walk directory: %w", err)
	}

	return docs, nil
}

// SupportedTypes returns supported types
func (l *DirectoryLoader) SupportedTypes() []string {
	var types []string
	for ext := range l.Loaders {
		types = append(types, ext)
	}
	return types
}

// TextSplitter splits text into chunks
type TextSplitter interface {
	// Split splits a document into chunks
	Split(doc *Document) []*DocumentChunk
}

// CharacterTextSplitter splits text by character count
type CharacterTextSplitter struct {
	// ChunkSize is the maximum chunk size
	ChunkSize int
	// ChunkOverlap is the overlap between chunks
	ChunkOverlap int
	// Separator is the separator to split on
	Separator string
}

// NewCharacterTextSplitter creates a new character text splitter
func NewCharacterTextSplitter(chunkSize, chunkOverlap int) *CharacterTextSplitter {
	return &CharacterTextSplitter{
		ChunkSize:    chunkSize,
		ChunkOverlap: chunkOverlap,
		Separator:    "\n\n",
	}
}

// Split splits a document into chunks
func (s *CharacterTextSplitter) Split(doc *Document) []*DocumentChunk {
	text := doc.Content
	var chunks []*DocumentChunk

	// Split by separator first
	parts := strings.Split(text, s.Separator)

	var currentChunk strings.Builder
	var currentOffset int
	chunkIndex := 0

	for _, part := range parts {
		if currentChunk.Len()+len(part)+len(s.Separator) > s.ChunkSize && currentChunk.Len() > 0 {
			// Create chunk
			chunks = append(chunks, &DocumentChunk{
				ID:          fmt.Sprintf("%s_chunk%d", doc.ID, chunkIndex),
				Content:     currentChunk.String(),
				Index:       chunkIndex,
				StartOffset: currentOffset,
				EndOffset:   currentOffset + currentChunk.Len(),
				Metadata:    copyMetadata(doc.Metadata),
			})

			// Handle overlap
			if s.ChunkOverlap > 0 && currentChunk.Len() > s.ChunkOverlap {
				overlapText := currentChunk.String()[currentChunk.Len()-s.ChunkOverlap:]
				currentOffset += currentChunk.Len() - s.ChunkOverlap
				currentChunk.Reset()
				currentChunk.WriteString(overlapText)
			} else {
				currentOffset += currentChunk.Len()
				currentChunk.Reset()
			}
			chunkIndex++
		}

		if currentChunk.Len() > 0 {
			currentChunk.WriteString(s.Separator)
		}
		currentChunk.WriteString(part)
	}

	// Add remaining chunk
	if currentChunk.Len() > 0 {
		chunks = append(chunks, &DocumentChunk{
			ID:          fmt.Sprintf("%s_chunk%d", doc.ID, chunkIndex),
			Content:     currentChunk.String(),
			Index:       chunkIndex,
			StartOffset: currentOffset,
			EndOffset:   currentOffset + currentChunk.Len(),
			Metadata:    copyMetadata(doc.Metadata),
		})
	}

	return chunks
}

// RecursiveCharacterTextSplitter recursively splits text
type RecursiveCharacterTextSplitter struct {
	// ChunkSize is the maximum chunk size
	ChunkSize int
	// ChunkOverlap is the overlap between chunks
	ChunkOverlap int
	// Separators is the list of separators to try
	Separators []string
}

// NewRecursiveCharacterTextSplitter creates a new recursive text splitter
func NewRecursiveCharacterTextSplitter(chunkSize, chunkOverlap int) *RecursiveCharacterTextSplitter {
	return &RecursiveCharacterTextSplitter{
		ChunkSize:    chunkSize,
		ChunkOverlap: chunkOverlap,
		Separators:   []string{"\n\n", "\n", ". ", " ", ""},
	}
}

// Split splits a document into chunks
func (s *RecursiveCharacterTextSplitter) Split(doc *Document) []*DocumentChunk {
	return s.splitText(doc.Content, doc.ID, doc.Metadata, s.Separators)
}

func (s *RecursiveCharacterTextSplitter) splitText(text, docID string, metadata map[string]interface{}, separators []string) []*DocumentChunk {
	var chunks []*DocumentChunk

	if len(text) <= s.ChunkSize {
		return []*DocumentChunk{{
			ID:       fmt.Sprintf("%s_chunk0", docID),
			Content:  text,
			Index:    0,
			Metadata: copyMetadata(metadata),
		}}
	}

	separator := separators[0]
	remainingSeps := separators[1:]

	parts := strings.Split(text, separator)
	var goodSplits []string

	for _, part := range parts {
		if len(part) < s.ChunkSize {
			goodSplits = append(goodSplits, part)
		} else if len(remainingSeps) > 0 {
			// Recursively split
			subChunks := s.splitText(part, docID, metadata, remainingSeps)
			for _, chunk := range subChunks {
				goodSplits = append(goodSplits, chunk.Content)
			}
		} else {
			// Just truncate
			goodSplits = append(goodSplits, part[:s.ChunkSize])
		}
	}

	// Merge splits into chunks
	var currentChunk strings.Builder
	chunkIndex := 0

	for _, split := range goodSplits {
		if currentChunk.Len()+len(split)+len(separator) > s.ChunkSize && currentChunk.Len() > 0 {
			chunks = append(chunks, &DocumentChunk{
				ID:       fmt.Sprintf("%s_chunk%d", docID, chunkIndex),
				Content:  strings.TrimSpace(currentChunk.String()),
				Index:    chunkIndex,
				Metadata: copyMetadata(metadata),
			})
			chunkIndex++

			// Handle overlap
			if s.ChunkOverlap > 0 && currentChunk.Len() > s.ChunkOverlap {
				overlapText := currentChunk.String()[currentChunk.Len()-s.ChunkOverlap:]
				currentChunk.Reset()
				currentChunk.WriteString(overlapText)
			} else {
				currentChunk.Reset()
			}
		}

		if currentChunk.Len() > 0 {
			currentChunk.WriteString(separator)
		}
		currentChunk.WriteString(split)
	}

	if currentChunk.Len() > 0 {
		chunks = append(chunks, &DocumentChunk{
			ID:       fmt.Sprintf("%s_chunk%d", docID, chunkIndex),
			Content:  strings.TrimSpace(currentChunk.String()),
			Index:    chunkIndex,
			Metadata: copyMetadata(metadata),
		})
	}

	return chunks
}

// SentenceTextSplitter splits text by sentences
type SentenceTextSplitter struct {
	// ChunkSize is the maximum number of sentences per chunk
	ChunkSize int
	// ChunkOverlap is the overlap in sentences
	ChunkOverlap int
}

// NewSentenceTextSplitter creates a new sentence text splitter
func NewSentenceTextSplitter(chunkSize, chunkOverlap int) *SentenceTextSplitter {
	return &SentenceTextSplitter{
		ChunkSize:    chunkSize,
		ChunkOverlap: chunkOverlap,
	}
}

// Split splits a document into chunks by sentences
func (s *SentenceTextSplitter) Split(doc *Document) []*DocumentChunk {
	// Simple sentence splitting using regex
	sentencePattern := regexp.MustCompile(`[.!?]+\s+`)
	sentences := sentencePattern.Split(doc.Content, -1)

	var chunks []*DocumentChunk
	chunkIndex := 0

	for i := 0; i < len(sentences); i += s.ChunkSize - s.ChunkOverlap {
		end := i + s.ChunkSize
		if end > len(sentences) {
			end = len(sentences)
		}

		chunkSentences := sentences[i:end]
		content := strings.Join(chunkSentences, ". ")

		chunks = append(chunks, &DocumentChunk{
			ID:       fmt.Sprintf("%s_chunk%d", doc.ID, chunkIndex),
			Content:  content,
			Index:    chunkIndex,
			Metadata: copyMetadata(doc.Metadata),
		})
		chunkIndex++

		if end == len(sentences) {
			break
		}
	}

	return chunks
}

// TokenTextSplitter splits text by token count (approximate)
type TokenTextSplitter struct {
	// ChunkSize is the maximum number of tokens per chunk
	ChunkSize int
	// ChunkOverlap is the overlap in tokens
	ChunkOverlap int
	// TokensPerWord is the approximate tokens per word
	TokensPerWord float64
}

// NewTokenTextSplitter creates a new token text splitter
func NewTokenTextSplitter(chunkSize, chunkOverlap int) *TokenTextSplitter {
	return &TokenTextSplitter{
		ChunkSize:     chunkSize,
		ChunkOverlap:  chunkOverlap,
		TokensPerWord: 1.3, // Approximate for English
	}
}

// Split splits a document into chunks by approximate token count
func (s *TokenTextSplitter) Split(doc *Document) []*DocumentChunk {
	words := strings.Fields(doc.Content)
	wordsPerChunk := int(float64(s.ChunkSize) / s.TokensPerWord)
	overlapWords := int(float64(s.ChunkOverlap) / s.TokensPerWord)

	var chunks []*DocumentChunk
	chunkIndex := 0

	for i := 0; i < len(words); i += wordsPerChunk - overlapWords {
		end := i + wordsPerChunk
		if end > len(words) {
			end = len(words)
		}

		content := strings.Join(words[i:end], " ")

		chunks = append(chunks, &DocumentChunk{
			ID:       fmt.Sprintf("%s_chunk%d", doc.ID, chunkIndex),
			Content:  content,
			Index:    chunkIndex,
			Metadata: copyMetadata(doc.Metadata),
		})
		chunkIndex++

		if end == len(words) {
			break
		}
	}

	return chunks
}

// MarkdownTextSplitter splits Markdown by headers
type MarkdownTextSplitter struct {
	// ChunkSize is the maximum chunk size
	ChunkSize int
	// ChunkOverlap is the overlap between chunks
	ChunkOverlap int
	// HeadersToSplitOn are header levels to split on
	HeadersToSplitOn []int
}

// NewMarkdownTextSplitter creates a new Markdown text splitter
func NewMarkdownTextSplitter(chunkSize, chunkOverlap int) *MarkdownTextSplitter {
	return &MarkdownTextSplitter{
		ChunkSize:        chunkSize,
		ChunkOverlap:     chunkOverlap,
		HeadersToSplitOn: []int{1, 2, 3},
	}
}

// Split splits a Markdown document by headers
func (s *MarkdownTextSplitter) Split(doc *Document) []*DocumentChunk {
	lines := strings.Split(doc.Content, "\n")
	var chunks []*DocumentChunk
	var currentChunk strings.Builder
	var currentHeader string
	chunkIndex := 0

	for _, line := range lines {
		// Check if line is a header
		isHeader := false
		for _, level := range s.HeadersToSplitOn {
			prefix := strings.Repeat("#", level) + " "
			if strings.HasPrefix(line, prefix) {
				// Save current chunk
				if currentChunk.Len() > 0 {
					metadata := copyMetadata(doc.Metadata)
					if currentHeader != "" {
						metadata["header"] = currentHeader
					}
					chunks = append(chunks, &DocumentChunk{
						ID:       fmt.Sprintf("%s_chunk%d", doc.ID, chunkIndex),
						Content:  strings.TrimSpace(currentChunk.String()),
						Index:    chunkIndex,
						Metadata: metadata,
					})
					chunkIndex++
					currentChunk.Reset()
				}

				currentHeader = strings.TrimPrefix(line, prefix)
				isHeader = true
				break
			}
		}

		if currentChunk.Len()+len(line) > s.ChunkSize && currentChunk.Len() > 0 {
			// Save chunk due to size
			metadata := copyMetadata(doc.Metadata)
			if currentHeader != "" {
				metadata["header"] = currentHeader
			}
			chunks = append(chunks, &DocumentChunk{
				ID:       fmt.Sprintf("%s_chunk%d", doc.ID, chunkIndex),
				Content:  strings.TrimSpace(currentChunk.String()),
				Index:    chunkIndex,
				Metadata: metadata,
			})
			chunkIndex++
			currentChunk.Reset()
		}

		if !isHeader {
			if currentChunk.Len() > 0 {
				currentChunk.WriteString("\n")
			}
			currentChunk.WriteString(line)
		}
	}

	// Add final chunk
	if currentChunk.Len() > 0 {
		metadata := copyMetadata(doc.Metadata)
		if currentHeader != "" {
			metadata["header"] = currentHeader
		}
		chunks = append(chunks, &DocumentChunk{
			ID:       fmt.Sprintf("%s_chunk%d", doc.ID, chunkIndex),
			Content:  strings.TrimSpace(currentChunk.String()),
			Index:    chunkIndex,
			Metadata: metadata,
		})
	}

	return chunks
}

func copyMetadata(m map[string]interface{}) map[string]interface{} {
	if m == nil {
		return make(map[string]interface{})
	}
	copy := make(map[string]interface{})
	for k, v := range m {
		copy[k] = v
	}
	return copy
}

// DocumentProcessor combines loading, splitting, and embedding
type DocumentProcessor struct {
	// Loaders maps extensions to loaders
	Loaders map[string]DocumentLoader

	// Splitter is the text splitter
	Splitter TextSplitter

	// Embedder generates embeddings
	Embedder Embedder

	mu sync.Mutex
}

// DocumentProcessorConfig configures the processor
type DocumentProcessorConfig struct {
	ChunkSize    int
	ChunkOverlap int
	Embedder     Embedder
}

// NewDocumentProcessor creates a new document processor
func NewDocumentProcessor(config *DocumentProcessorConfig) *DocumentProcessor {
	if config == nil {
		config = &DocumentProcessorConfig{
			ChunkSize:    1000,
			ChunkOverlap: 200,
		}
	}

	return &DocumentProcessor{
		Loaders: map[string]DocumentLoader{
			".txt":      NewTextDocumentLoader(),
			".md":       NewMarkdownDocumentLoader(),
			".markdown": NewMarkdownDocumentLoader(),
			".html":     NewHTMLDocumentLoader(true),
			".htm":      NewHTMLDocumentLoader(true),
			".json":     NewJSONDocumentLoader("content"),
			".csv":      NewCSVDocumentLoader(false),
		},
		Splitter: NewRecursiveCharacterTextSplitter(config.ChunkSize, config.ChunkOverlap),
		Embedder: config.Embedder,
	}
}

// Process loads, splits, and optionally embeds documents
func (p *DocumentProcessor) Process(ctx context.Context, source string) ([]*Document, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Determine loader
	info, err := os.Stat(source)
	if err != nil {
		return nil, fmt.Errorf("failed to stat source: %w", err)
	}

	var docs []*Document

	if info.IsDir() {
		dirLoader := NewDirectoryLoader(true)
		dirLoader.Loaders = p.Loaders
		docs, err = dirLoader.Load(ctx, source)
	} else {
		ext := strings.ToLower(filepath.Ext(source))
		loader, ok := p.Loaders[ext]
		if !ok {
			return nil, fmt.Errorf("no loader for extension: %s", ext)
		}
		docs, err = loader.Load(ctx, source)
	}

	if err != nil {
		return nil, err
	}

	// Split documents
	for _, doc := range docs {
		doc.Chunks = p.Splitter.Split(doc)

		// Generate embeddings if embedder is available
		if p.Embedder != nil {
			for _, chunk := range doc.Chunks {
				embedding, err := p.Embedder.Embed(ctx, chunk.Content)
				if err == nil {
					chunk.Embeddings = embedding
				}
			}
		}
	}

	return docs, nil
}

// RegisterLoader registers a loader for an extension
func (p *DocumentProcessor) RegisterLoader(ext string, loader DocumentLoader) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.Loaders[ext] = loader
}

// SetSplitter sets the text splitter
func (p *DocumentProcessor) SetSplitter(splitter TextSplitter) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.Splitter = splitter
}

// SetEmbedder sets the embedder
func (p *DocumentProcessor) SetEmbedder(embedder Embedder) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.Embedder = embedder
}

// WebLoader loads documents from URLs
type WebLoader struct {
	// Client is the HTTP client
	Client *http.Client
	// ExtractText extracts plain text from HTML
	ExtractText bool
	// UserAgent is the user agent string
	UserAgent string
}

// NewWebLoader creates a new web loader
func NewWebLoader(extractText bool) *WebLoader {
	return &WebLoader{
		Client:      &http.Client{Timeout: 30 * time.Second},
		ExtractText: extractText,
		UserAgent:   "DaprAgents/1.0",
	}
}

// Load loads a document from a URL
func (l *WebLoader) Load(ctx context.Context, source string) ([]*Document, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", source, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("User-Agent", l.UserAgent)

	resp, err := l.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch URL: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP error: %d", resp.StatusCode)
	}

	content, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	contentType := resp.Header.Get("Content-Type")
	text := string(content)

	if l.ExtractText && strings.Contains(contentType, "text/html") {
		text, _ = extractTextFromHTML(content)
	}

	return []*Document{{
		ID:       source,
		Content:  text,
		Source:   source,
		MimeType: contentType,
		Metadata: map[string]interface{}{
			"url":          source,
			"content_type": contentType,
			"status_code":  resp.StatusCode,
		},
		CreatedAt: time.Now(),
	}}, nil
}

// SupportedTypes returns supported types
func (l *WebLoader) SupportedTypes() []string {
	return []string{"http://", "https://"}
}

// SitemapLoader loads URLs from a sitemap
type SitemapLoader struct {
	// WebLoader loads individual pages
	WebLoader *WebLoader
	// MaxPages limits the number of pages to load
	MaxPages int
}

// NewSitemapLoader creates a new sitemap loader
func NewSitemapLoader(maxPages int) *SitemapLoader {
	return &SitemapLoader{
		WebLoader: NewWebLoader(true),
		MaxPages:  maxPages,
	}
}

// Load loads documents from a sitemap URL
func (l *SitemapLoader) Load(ctx context.Context, sitemapURL string) ([]*Document, error) {
	// Fetch sitemap
	resp, err := http.Get(sitemapURL)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch sitemap: %w", err)
	}
	defer resp.Body.Close()

	content, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read sitemap: %w", err)
	}

	// Extract URLs (simple regex-based extraction)
	urlPattern := regexp.MustCompile(`<loc>(.*?)</loc>`)
	matches := urlPattern.FindAllStringSubmatch(string(content), -1)

	var docs []*Document
	count := 0

	for _, match := range matches {
		if l.MaxPages > 0 && count >= l.MaxPages {
			break
		}

		url := match[1]
		pageDocs, err := l.WebLoader.Load(ctx, url)
		if err != nil {
			continue // Skip failed pages
		}
		docs = append(docs, pageDocs...)
		count++
	}

	return docs, nil
}

// SupportedTypes returns supported types
func (l *SitemapLoader) SupportedTypes() []string {
	return []string{"sitemap.xml"}
}
