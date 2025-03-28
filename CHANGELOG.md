# Changelog

All notable changes to this project will be documented in this file.

## [1.7.0] - 2025-03-28

### Added
- Advanced GitHub features (Phase 4)
  - GitHub Actions integration
    - Workflow listing, retrieval, and execution
    - Build/test status checking
    - Workflow visualization
  - PR management
    - PR creation and analysis
    - Code review features
    - PR visualization with metrics
  - Repository analysis
    - Dependency scanning
    - Code quality metrics
    - Repository visualization (commits, contributors, languages)

### Changed
- Enhanced GitHub tools with more comprehensive capabilities
- Added visualization components for repository data
- Updated requirements.txt with new dependencies for graph visualization

## [1.6.0] - 2025-03-28

### Added
- Web interface
  - FastAPI backend with comprehensive API endpoints
  - User authentication with API keys
  - WebSocket support for streaming responses
  - File upload capabilities for documents and images
  - React frontend with chat UI and tool visualization
  - Markdown rendering and syntax highlighting for code
- Enhanced RAG capabilities
  - Support for multiple document formats (PDF, DOCX, Excel, CSV, HTML)
  - Document collections and namespaces
  - Hybrid search combining vector and keyword search
  - Metadata filtering and advanced search options
  - Document chunking strategies (tokens, paragraphs, sentences)
- Comprehensive test suite
  - Unit tests for all major components
  - Integration tests for API endpoints
  - Test coverage reporting
  - Automated test runner

### Changed
- Improved error handling throughout the codebase
- Enhanced documentation with detailed comments
- Updated requirements.txt with new dependencies
- Optimized performance for RAG operations with large document collections

## [1.5.0] - 2025-03-27

### Added
- Support for image inputs with Claude 3 models
  - Image processing utilities for encoding and optimization
  - Image analysis tools using Claude's multimodal capabilities
  - OCR (text extraction) from images
  - Image description and detailed analysis
  - Image downloading and management
- CLI commands for image operations:
  - `image analyze` - Analyze images with custom prompts
  - `image ocr` - Extract text from images
  - `image describe` - Generate image descriptions
  - `image optimize` - Resize and compress images for Claude
  - `image download` - Download images from URLs
  - `image send` - Send messages with images
- Command-line options for image analysis:
  - `--image` - Specify image file to analyze
  - `--image-prompt` - Custom prompt for image analysis

### Changed
- Updated `Agent` class with multimodal input handling
- Enhanced CLI interface with image commands
- Improved main.py with image-specific command-line options

## [1.4.0] - 2025-03-27

### Added
- Retrieval-Augmented Generation (RAG) system
  - Vector store for document embeddings
  - Document processing and chunking strategies
  - Embedding generation using Claude API
  - Retrieval mechanisms and context injection
  - RAG integration with agent responses
- RAG command-line interface
  - Document management (add, delete, list)
  - RAG system querying
  - Statistics and reporting
- RAG tools for the agent
  - Document addition and retrieval
  - Context enhancement for queries
  - Metadata filtering and search
- Command-line options for RAG operations

### Changed
- Updated `Agent` class to support RAG context enhancement
- Improved CLI with RAG command support
- Enhanced main.py with command-line options for RAG

## [1.3.0] - 2025-03-27

### Added
- JSON mode support for structured outputs from Claude
- Predefined schemas for common outputs (text analysis, code analysis, etc.)
- Structured output CLI commands:
  - `structured list` - List available schemas
  - `structured analyze` - Process query with structured output
  - `structured custom` - Create and use custom schemas
- Command-line options for structured output processing
- Custom schema creation capabilities
- Enhanced anthropic_cookbook with structured output tools

### Changed
- Updated `Agent` class with methods for structured responses
- Improved CLI to support working with structured data
- Enhanced error handling for JSON parsing and validation

## [1.2.0] - 2025-03-27

### Added
- Model configuration and selection system with detailed model information
- Usage tracking and cost estimation for Claude API calls
- Model recommendation system based on task descriptions
- New CLI commands for model management:
  - `models list` - List all available models
  - `models info` - Show detailed model information
  - `models recommend` - Get model recommendations for tasks
  - `models current` - Show current model information
- `usage` command to show API usage statistics and costs
- Enhanced conversation saving with model information and usage stats

## [1.1.0] - 2025-03-27

### Added
- Streaming response capability for more responsive conversation
- Command-line argument support for model selection and streaming configuration
- `streaming` command to toggle streaming mode in CLI
- Enhanced error handling for streaming responses

### Changed
- Updated `Agent` class to support both streaming and non-streaming responses
- Modified CLI to display streamed responses in real-time
- Improved tool execution feedback during streaming

## [1.0.0] - 2025-03-27

### Added
- Initial release of the Anthropic-powered Agent with GitHub integration
- Core agent implementation with memory system and tool execution
- GitHub integration tools (repository info, code analysis, issue management)
- Claude-powered language tools (summarization, translation, code completion)
- System utility tools (file operations, data analysis, command execution)
- Advanced techniques from Anthropic's cookbook
- Command-line interface for interacting with the agent
- Comprehensive error handling and validation