# YouTube Comment Analysis 

## Overview

This project analyzes YouTube video comments using GPT-based and keyword-based methods, providing **interactive web visualizations** and detailed results. It is designed to be cross-platform (macOS, Linux, Windows) and easy to set up and run.

## Features

- **Interactive Web App**: Real-time progress tracking and task-by-task visualization
- **GPT-Powered Analysis**: AI-generated search specifications and sentiment analysis
- **Cross-Platform**: Works on macOS, Linux, and Windows
- **Real-time Progress**: Live updates during analysis
- **Task Flow Visualization**: Step-by-step results for each analysis task

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-folder>
```

### 2. Install Dependencies

We recommend using a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Your API Key

You need an OpenAI API key (or compatible key for your chosen model). You can provide it in two ways:

- **Environment variable:**
  ```bash
  export OPENAI_API_KEY=sk-...
  # On Windows (cmd):
  set OPENAI_API_KEY=sk-...
  # On Windows (PowerShell):
  $env:OPENAI_API_KEY="sk-..."
  ```
- **Or create a .env file:**
  ```bash
  echo "OPENAI_API_KEY=sk-..." > .env
  ```

### 4. Run the Server

```bash
python run_server.py
```

- The server will start at [http://localhost:8000](http://localhost:8000)
- Upload your dataset (CSV or XLSX) via the web interface
- The analysis will run and show **real-time progress** with live task updates
- **Interactive web app** will display results for each task

### 5. Web App Features

The interactive web app provides:
- **Real-time Progress Tracking**: Live updates during analysis
- **Task-by-Task Results**: Detailed breakdown of each analysis step
- **Interactive Task Navigation**: Click between different analysis tasks
- **Visual Data Presentation**: Charts and summaries for each task
- **Download Results**: Save analysis results as JSON files

### 6. Output

Each run creates a new directory in `output/run-[unique-run-id]/` containing:
- `results.json` (detailed results)
- `task_results.json` (task-by-task breakdown)
- `interactive_visualization.html` (server-based web app)
- `standalone_visualization.html` (standalone web app - no server required)

**For standalone viewing:** Open `standalone_visualization.html` in any web browser to view the complete analysis results with interactive task navigation. This file contains all data embedded and works without any server.

**For server-based viewing:** Use `interactive_visualization.html` when running the server at localhost:8000.

### 7. Custom Models

If you wish to use a non-OpenAI model, update the `gpt_call` function in `challenge.py` to use your preferred provider. Document any additional setup in this README.

## Dataset Format

Your Excel/CSV file should contain these columns:
- `id` - Comment/video ID
- `parent_id` - Parent comment ID (null for videos)
- `content` - Comment text
- `author_id` - Author identifier

## Analysis Tasks

The system performs 6 main tasks:
1. **Data Cleaning & Separation** - Identify videos and comments
2. **Dynamic Search Specifications** - AI-generated search criteria
3. **Static Search Specifications** - Universal search criteria
4. **Data Summary** - Cleaning statistics
5. **Comment Search Results** - Matched comments for each spec
6. **Generalizations & Insights** - Sentiment analysis, topics, questions

## Notes
- No API key is committed to the repo. Use `.env.example` as a template if you wish.
- The `output/` directory is generated and should be added to `.gitignore`.
- The solution is cross-platform and requires only Python 3.8+ and pip.
- **Web app visualization** is automatically generated and served via the FastAPI server.

## Troubleshooting
- If you see an error about the API key, ensure it is set in your environment or .env file.
- For large datasets, analysis may take several minutes.
- The web app requires JavaScript to be enabled in your browser.