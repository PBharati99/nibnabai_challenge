from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import tempfile
import json
import asyncio
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime
import math

# Import our analysis functions
from challenge import (
    create_interactive_visualization_enhanced,
    create_standalone_visualization,
    clean_data,
    separate_videos_and_comments,
    generate_dynamic_spec,
    search_comments,
    sentiment_analysis,
    extract_topics,
    extract_questions,
    STATIC_SPECS
)

app = FastAPI(title="YouTube Comment Analysis API", version="2.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_for_json(obj):
    """
    Recursively clean data structure to ensure JSON compatibility.
    Replaces inf, -inf, and NaN values with safe alternatives.
    """
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return 0.0
        elif math.isinf(obj):
            return 1.0 if obj > 0 else 0.0
        else:
            return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        return str(obj)

# Global storage for analysis progress
analysis_progress = {}
analysis_results = {}

class ProgressTracker:
    def __init__(self, analysis_id: str):
        self.analysis_id = analysis_id
        self.progress = {
            "status": "starting",
            "current_task": "",
            "current_step": "",
            "progress_percentage": 0,
            "logs": [],
            "results": {},
            "error": None
        }
        analysis_progress[analysis_id] = self.progress
    
    def update(self, task: str, step: str, percentage: int, log_message: str = ""):
        self.progress["current_task"] = task
        self.progress["current_step"] = step
        self.progress["progress_percentage"] = percentage
        if log_message:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.progress["logs"].append(f"[{timestamp}] {log_message}")
        analysis_progress[self.analysis_id] = self.progress
    
    def set_results(self, results: Dict[str, Any]):
        # Clean results for JSON safety before storing
        cleaned_results = clean_for_json(results)
        self.progress["results"] = cleaned_results
        self.progress["status"] = "completed"
        analysis_progress[self.analysis_id] = self.progress
    
    def set_error(self, error: str):
        self.progress["error"] = error
        self.progress["status"] = "error"
        analysis_progress[self.analysis_id] = self.progress

def analyze_dataset_with_progress(file_path: str, analysis_id: str, use_gpt_search: bool, use_all_comments: bool, use_sampling: bool):
    """Enhanced analysis function with progress tracking"""
    tracker = ProgressTracker(analysis_id)
    
    try:
        # Task 1: Load and Clean Data
        tracker.update("Task 1", "Loading dataset", 5, "Loading dataset file...")
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            try:
                df = pd.read_csv(file_path, sep="\t")
            except Exception as e2:
                raise ValueError(f"Cannot read file as Excel or CSV: {e2}")
        
        tracker.update("Task 1", "Cleaning data", 10, f"Cleaning dataset with {len(df)} rows...")
        df = clean_data(df)
        tracker.update("Task 1", "Separating videos and comments", 15, f"✅ Cleaned dataset: {len(df)} rows")
        
        videos, comments = separate_videos_and_comments(df)
        tracker.update("Task 1", "Completed", 20, f"✅ Found {len(videos)} videos and {len(comments)} comments")
        
        # Initialize results structure
        results = {}
        task_results = {
            "task1": {
                "videos_count": len(videos),
                "comments_count": len(comments),
                "videos": videos.to_dict(orient="records")[:5],  # Show first 5 videos
                "status": "completed"
            },
            "task2": {},
            "task3": {"static_specs": STATIC_SPECS},
            "task4": {
                "cleaned_rows": len(df),
                "status": "completed"
            },
            "task5": {},
            "task6": {}
        }
        
        total_videos = len(videos)
        tracker.update("Task 2", "Starting", 25, f"🚀 Starting analysis of {total_videos} videos...")
        
        # Process each video
        for idx, (_, video) in enumerate(videos.iterrows()):
            video_id = video['id']
            video_comments = comments[comments['parent_id'] == video_id]
            
            progress = 25 + int((idx / total_videos) * 70)  # 25% to 95%
            
            # Task 2: Dynamic Spec Generation
            tracker.update("Task 2", f"Video {idx+1}/{total_videos}", progress, 
                         f"🔍 Processing video {idx+1}/{total_videos}: {video_id} ({len(video_comments)} comments)")
            
            try:
                dynamic_spec = generate_dynamic_spec(video)
                task_results["task2"][video_id] = dynamic_spec
                tracker.update("Task 2", f"Video {idx+1}/{total_videos}", progress, 
                             f"✅ Dynamic spec generated: {', '.join(dynamic_spec['keywords'][:3])}...")
            except Exception as e:
                tracker.update("Task 2", f"Video {idx+1}/{total_videos}", progress, 
                             f"⚠️ GPT failed, using fallback keywords")
                task_results["task2"][video_id] = {
                    "keywords": ["general feedback", "questions", "criticism"],
                    "description": "Fallback spec due to API error"
                }
            
            # Task 3: Static Specs
            tracker.update("Task 3", f"Video {idx+1}/{total_videos}", progress, 
                         f"🔍 Applying static search specs...")
            static_results = {}
            for spec in STATIC_SPECS:
                static_results[spec["name"]] = search_comments(video_comments, spec, use_gpt_search, use_all_comments)
            task_results["task3"][video_id] = static_results
            
            # Task 5: Search Comments
            tracker.update("Task 5", f"Video {idx+1}/{total_videos}", progress, 
                         f"🔍 Searching comments with dynamic spec...")
            matched_comments = {
                "dynamic": search_comments(video_comments, task_results["task2"][video_id], use_gpt_search, use_all_comments),
                "static": static_results
            }
            task_results["task5"][video_id] = matched_comments
            tracker.update("Task 5", f"Video {idx+1}/{total_videos}", progress, 
                         f"✅ Found {len(matched_comments['dynamic'])} dynamic matches")
            
            # Task 6: Generalizations
            tracker.update("Task 6", f"Video {idx+1}/{total_videos}", progress, 
                         f"🔍 Generating generalizations...")
            comment_list = video_comments.to_dict(orient="records")
            
            try:
                if use_all_comments and not use_sampling:
                    sentiment = sentiment_analysis(comment_list, use_all_comments=True)
                    topics = extract_topics(comment_list, use_all_comments=True)
                    questions = extract_questions(comment_list, use_all_comments=True)
                else:
                    sentiment = sentiment_analysis(comment_list, use_all_comments=False)
                    topics = extract_topics(comment_list, use_all_comments=False)
                    questions = extract_questions(comment_list, use_all_comments=False)
                
                # Ensure sentiment is a valid float
                try:
                    sentiment = float(sentiment)
                    # Check for infinite or NaN values
                    if not math.isfinite(sentiment):
                        sentiment = 0.5
                    sentiment = max(0.0, min(1.0, sentiment))  # Clamp to 0-1 range
                except (ValueError, TypeError):
                    sentiment = 0.5  # Default fallback
                
                task_results["task6"][video_id] = {
                    "sentiment": sentiment,
                    "topics": topics,
                    "questions": questions,
                    "total_comments_analyzed": len(comment_list)
                }
                
                tracker.update("Task 6", f"Video {idx+1}/{total_videos}", progress, 
                             f"✅ Sentiment: {sentiment:.3f}, Topics: {len(topics)}, Questions: {len(questions)}")
                
            except Exception as e:
                tracker.update("Task 6", f"Video {idx+1}/{total_videos}", progress, 
                             f"⚠️ GPT failed, using default values")
                task_results["task6"][video_id] = {
                    "sentiment": 0.5,
                    "topics": ["general feedback"],
                    "questions": ["Sample question"],
                    "total_comments_analyzed": len(comment_list)
                }
            
            # Store video results
            # Ensure sentiment is valid before storing
            sentiment_value = task_results["task6"][video_id]["sentiment"]
            try:
                sentiment_value = float(sentiment_value)
                # Check for infinite or NaN values
                if not math.isfinite(sentiment_value):
                    sentiment_value = 0.5
                sentiment_value = max(0.0, min(1.0, sentiment_value))  # Clamp to 0-1 range
            except (ValueError, TypeError):
                sentiment_value = 0.5  # Default fallback
            
            results[video_id] = {
                "video_author": video['author_id'] if video['author_id'] and str(video['author_id']).lower() != 'nan' else 'Unknown Author',
                "video_content": video['content'][:200] + "..." if len(video['content']) > 200 else video['content'],
                "video_url": video.get('url', ''),
                "dynamic_spec": task_results["task2"][video_id],
                "matched_comments": matched_comments,
                "generalizations": {
                    "avg_sentiment": sentiment_value,
                    "topics": task_results["task6"][video_id]["topics"],
                    "questions": task_results["task6"][video_id]["questions"]
                }
            }
        
        # Final steps
        tracker.update("Finalizing", "Creating visualization", 95, "🎨 Analysis complete! Creating visualization...")
        
        # Create visualization with embedded data for standalone use
        html_content = create_interactive_visualization_enhanced(results, task_results)
        
        # Create a standalone version that embeds the data directly
        standalone_html = create_standalone_visualization(results, task_results)
        
        # Save results
        run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        output_dir = f"output/{run_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean data for JSON serialization
        cleaned_results = clean_for_json(results)
        cleaned_task_results = clean_for_json(task_results)
        
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(cleaned_results, f, indent=2)
        with open(f"{output_dir}/task_results.json", "w") as f:
            json.dump(cleaned_task_results, f, indent=2)
        
        # Save the interactive HTML visualization (server version)
        with open(f"{output_dir}/interactive_visualization.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Save the standalone HTML visualization (with embedded data)
        with open(f"{output_dir}/standalone_visualization.html", "w", encoding="utf-8") as f:
            f.write(standalone_html)
        
        tracker.update("Completed", "Done", 100, f"✅ Results and visualization saved to {output_dir}/")
        
        # Set final results
        final_results = {
            "results": cleaned_results,
            "task_results": cleaned_task_results,
            "html_content": html_content,
            "output_dir": output_dir
        }
        tracker.set_results(final_results)
        
    except Exception as e:
        tracker.set_error(str(e))
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the enhanced interactive visualization"""
    try:
        html_content = create_interactive_visualization_enhanced()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading visualization: {str(e)}</h1>", status_code=500)

@app.post("/analyze")
async def analyze_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_gpt_search: bool = Form(True),
    use_all_comments: bool = Form(False),
    use_sampling: bool = Form(True)
):
    """Start analysis in background and return analysis ID"""
    try:
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.csv')):
            raise HTTPException(status_code=400, detail="Only Excel (.xlsx) and CSV files are supported")
        
        # Generate analysis ID
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Start analysis in background
        background_tasks.add_task(
            analyze_dataset_with_progress,
            tmp_file_path,
            analysis_id,
            use_gpt_search,
            use_all_comments,
            use_sampling
        )
        
        return {
            "status": "started",
            "analysis_id": analysis_id,
            "message": "Analysis started successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.get("/progress/{analysis_id}")
async def get_progress(analysis_id: str):
    """Get analysis progress"""
    if analysis_id not in analysis_progress:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_progress[analysis_id]

@app.get("/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Get analysis results"""
    if analysis_id not in analysis_progress:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    progress = analysis_progress[analysis_id]
    if progress["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed yet")
    
    return progress["results"]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Enhanced API is running"}

if __name__ == "__main__":
    print("🚀 Starting YouTube Comment Analysis API Server...")
    print("📁 Open http://localhost:8000 in your browser")
    print("✨ Features: Real-time progress tracking, live task updates")
    uvicorn.run(app, host="0.0.0.0", port=8000) 