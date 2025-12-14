# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a PromptCraft debugging tool built with Streamlit. The application helps users iteratively debug and optimize LLM prompts through automated evaluation and optimization cycles. There are three main Python files implementing different versions of the tool:

1. `app.py` - Basic version with mock LLM services
2. `prompt_debugger.py` - Enhanced version with improved UI and features
3. `app_api.py` - Production-ready version with actual OpenAI API integration

## Project Structure

- `app.py`, `prompt_debugger.py`, `app_api.py`: Main application files with different implementations
- `README.md`: Project documentation in Chinese
- Data is stored locally in JSON format in the `prompt_sessions` directory
- No external database is required

## Key Features

- Prompt templating with Jinja2/{{variable}} syntax
- Iterative debugging with automatic or manual optimization paths
- LLM evaluation and scoring system
- Smart rollback mechanism when quality decreases
- Visual diff comparison between prompt versions
- Export functionality for debugging reports
- Support for multiple LLM providers (OpenAI, Qwen, DeepSeek)

## Development Commands

### Running the Application

To run any version of the application:
```bash
streamlit run app_api.py
```

Or for the other versions:
```bash
streamlit run app.py
streamlit run prompt_debugger.py
```

### Dependencies

Install required packages:
```bash
pip install streamlit openai jinja2 pandas textdistance
```

Note: The project can run in mock mode without API keys for testing UI functionality.

### Testing

There are no formal tests in this repository. Testing is done manually through the Streamlit interface.

## Architecture

### Core Components

1. **UI Layer**: Streamlit-based interface with multiple tabs for different views
2. **Logic Layer**:
   - LLMEngine/MockLLMService for LLM interactions
   - SessionManager/DataManager for data persistence
3. **Data Layer**: Local JSON file storage for session history

### Key Classes

- `LLMEngine` (app_api.py): Handles real API calls to LLM providers
- `MockLLMService` (prompt_debugger.py): Simulates LLM responses for testing
- `SessionManager`/`DataManager`: Manages data persistence
- Various helper functions for rendering, diff generation, and logging

### Data Flow

1. User defines prompt template with variables
2. System renders prompt with test data
3. LLM executes the prompt
4. Results are evaluated by another LLM or mock service
5. Scores are compared and optimization suggestions are generated
6. Process repeats for specified iterations or until user intervention

## Common Development Tasks

### Adding New LLM Providers

Modify the LLMEngine class in app_api.py to add new provider configurations and API call patterns.

### Modifying Evaluation Criteria

Update the evaluation prompt in the `evaluate` method of LLMEngine to change how prompts are scored.

### Changing Optimization Strategy

Modify the optimization prompt in the `optimize` method of LLMEngine to alter how prompts are improved.

### UI Customization

Streamlit components can be modified directly in the main application files. Custom CSS is included in the header of app_api.py.