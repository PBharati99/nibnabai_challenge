# YouTube Comment Analysis

## Overview

This project analyzes YouTube video comments using GPT-based and keyword-based methods, providing interactive web visualizations and detailed results. It is designed to be cross-platform (macOS, Linux, Windows) and easy to set up and run.

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
- **Or as an argument:**
  You can pass the API key directly to the analysis functions if running from Python.

### 4. Run the Server

```bash
python run_server.py
```

- The server will start at [http://localhost:8000](http://localhost:8000)
- Upload your dataset (CSV or XLSX) via the web interface
- The analysis will run and show real-time progress

### 5. Output

Each run creates a new directory in `output/run-[unique-run-id]/` containing:
- `results.json` (detailed results)
- `task_results.json` (task-by-task breakdown)

### 6. Custom Models

If you wish to use a non-OpenAI model, update the `gpt_call` function in `challenge_.py` to use your preferred provider. Document any additional setup in this README.

## Notes
- No API key is committed to the repo. Use `.env.example` as a template if you wish.
- The `output/` directory is generated and should be added to `.gitignore`.
- The solution is cross-platform and requires only Python 3.8+ and pip.

## Troubleshooting
- If you see an error about the API key, ensure it is set in your environment or passed as an argument.
- For large datasets, analysis may take several minutes.

## License
MIT 