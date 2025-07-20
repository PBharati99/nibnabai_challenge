#!/usr/bin/env python3
"""
Enhanced FastAPI Server Runner with Real-time Progress
"""

import os
import uvicorn
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        print("📄 Loading .env file...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ .env file loaded successfully!")
    else:
        print("⚠️  No .env file found. Please create one with your OPENAI_API_KEY")

def check_api_key():
    """Check if OpenAI API key is available"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        return False
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: API key doesn't start with 'sk-'. Please check your key.")
    
    print("✅ OpenAI API key found!")
    return True

if __name__ == "__main__":
    print("🚀 YouTube Comment Analysis - Interactive Server")
    print("=" * 60)
    
    # Load .env file
    load_env_file()
    
    # Only check for the API key in the __main__ block, not at import/module level
    # Check API key
    if not check_api_key():
        print("\n❌ Cannot start server without API key")
        print("   Please create a .env file with your OPENAI_API_KEY")
        exit(1)
    
    print("\n🌐 Starting FastAPI server...")
    print("📁 Open http://localhost:8000 in your browser")
    print("✨ New Features:")
    print("   • Real-time progress tracking")
    print("   • Live task output display")
    print("   • Background processing")
    print("   • Enhanced error handling")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Import and run the server
    from api_server import app
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 