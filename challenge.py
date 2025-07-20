import os
import json
import uuid
import pandas as pd
import re
from datetime import datetime
from openai import OpenAI
from collections import Counter
import base64
import io
import math

# ==================================
# Utility: JSON Parsing Helper
# ==================================
def extract_json_from_response(response, default_value):
    """
    Extract JSON from GPT response, handling cases where the response
    contains extra text or formatting.
    """
    try:
        # First, try direct JSON parsing
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON array from the response
        json_match = re.search(r'\[.*?\]', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to extract JSON from markdown code blocks
        markdown_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if markdown_match:
            try:
                return json.loads(markdown_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to extract array content without brackets
        array_content = re.search(r'\[(.*?)\]', response, re.DOTALL)
        if array_content:
            try:
                # Clean up the content and reconstruct as JSON
                content = array_content.group(1)
                # Remove quotes and split by comma
                items = [item.strip().strip('"\'') for item in content.split(',')]
                items = [item for item in items if item]
                return items
            except:
                pass
        
        print(f"Failed to parse JSON response: {response}")
        return default_value

def extract_number_from_response(response, default_value=0.5):
    """
    Extract a number from GPT response, handling cases where the response
    contains extra text or formatting. For sentiment analysis, ensures
    the result is always between 0 and 1 and is JSON-safe.
    """
    try:
        # First, try direct float conversion
        number = float(response)
        # Check for infinite or NaN values
        if not math.isfinite(number):
            return default_value
        # Clamp to 0-1 range for sentiment analysis
        return max(0.0, min(1.0, number))
    except (ValueError, TypeError):
        # Try to extract number from the response
        number_match = re.search(r'0\.\d+|\d+\.\d+|\d+', response)
        if number_match:
            try:
                number = float(number_match.group())
                # Check for infinite or NaN values
                if not math.isfinite(number):
                    return default_value
                # Clamp to 0-1 range for sentiment analysis
                return max(0.0, min(1.0, number))
            except (ValueError, TypeError):
                pass
        
        print(f"Failed to parse number from response: {response}")
        return default_value

# ==================================
# Utility: Call OpenAI GPT
# ==================================
def get_openai_client(api_key=None):
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please provide the OpenAI API key as an argument or set the OPENAI_API_KEY environment variable")
    return OpenAI(api_key=api_key)

def gpt_call(prompt, model="gpt-4o", max_tokens=1000, temperature=0.8, max_retries=2, api_key=None, client=None):
    """
    Enhanced GPT call with retry logic and better error handling
    """
    if client is None:
        client = get_openai_client(api_key)
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries:
                print(f"GPT call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                import time
                time.sleep(1)  # Wait before retry
            else:
                print(f"GPT call failed after {max_retries + 1} attempts: {e}")
                return ""
    return ""

# ==================================
# Task 1: Separate Videos & Comments
# ==================================
def separate_videos_and_comments(df):
    """
    Separate videos and comments, including replies to comments
    """
    videos = df[df['parent_id'].isna()].reset_index(drop=True)
    comments = df[~df['parent_id'].isna()].reset_index(drop=True)
    
    # Handle missing author IDs
    videos['author_id'] = videos['author_id'].fillna('Unknown Author')
    
    # Include replies to comments (comments that have parent_id that is not a video)
    # This ensures we capture the full conversation thread
    all_comments = comments.copy()
    
    print(f"Found {len(videos)} videos and {len(all_comments)} comments (including replies)")
    
    return videos, all_comments

# ==================================
# Task 2: Dynamic CommentSearchSpec
# ==================================
def extract_channel_id_from_url(video_url):
    """
    Extract channel ID from YouTube video URL
    """
    try:
        if 'youtube.com/watch?v=' in video_url:
            # Extract video ID first
            video_id = video_url.split('v=')[1].split('&')[0]
            
            # For now, we'll use the video ID as a proxy for channel analysis
            # In a full implementation, you'd fetch the channel ID from YouTube API
            return video_id
        elif 'youtu.be/' in video_url:
            video_id = video_url.split('youtu.be/')[1].split('?')[0]
            return video_id
        else:
            return None
    except:
        return None

def get_author_content_categories(channel_id, video_url, video_content):
    """
    Analyze the author's content categories based on video URL and content
    """
    prompt = f"""
    You are analyzing a YouTube creator's content to understand their typical categories and audience.
    
    Video URL: {video_url}
    Video Content: {video_content}
    Channel/Video ID: {channel_id}
    
    Based on this information, identify the creator's typical content categories and what types of comments their audience typically leaves.
    
    Return ONLY a JSON object with this structure:
    {{
        "content_categories": ["category1", "category2", "category3"],
        "audience_interests": ["interest1", "interest2", "interest3"],
        "typical_comment_topics": ["topic1", "topic2", "topic3", "topic4", "topic5"]
    }}
    
    Focus on specific, actionable categories that would help find relevant comments.
    """
    
    response = gpt_call(prompt)
    try:
        return json.loads(response)
    except:
        # Fallback if JSON parsing fails
        return {
            "content_categories": ["general content"],
            "audience_interests": ["general feedback"],
            "typical_comment_topics": ["general feedback", "questions", "opinions"]
        }

def generate_dynamic_spec(video_row):
    """
    Enhanced dynamic spec generation with randomization for varied keywords
    """
    import random
    import time
    
    video_url = video_row.get('url', '')
    video_content = video_row.get('content', '')
    author_id = video_row.get('author_id', 'Unknown Author')
    
    # Extract channel ID from URL
    channel_id = extract_channel_id_from_url(video_url)
    
    # Get author's content categories and typical comment topics
    author_analysis = get_author_content_categories(channel_id, video_url, video_content)
    
    # Add randomization to make prompts different each time
    random_focus_areas = [
        "Focus on technical specifications and performance aspects",
        "Focus on user experience and usability features", 
        "Focus on price, value proposition, and cost considerations",
        "Focus on comparisons with alternatives and competitors",
        "Focus on community feedback, opinions, and user reactions",
        "Focus on specific features, capabilities, and unique selling points",
        "Focus on potential issues, concerns, and problem areas",
        "Focus on recommendations, suggestions, and improvement ideas",
        "Focus on brand reputation and market positioning",
        "Focus on innovation, future potential, and cutting-edge features"
    ]
    
    random_perspectives = [
        "from a technical enthusiast's perspective",
        "from a casual user's perspective", 
        "from a professional reviewer's perspective",
        "from a price-conscious consumer's perspective",
        "from a power user's perspective",
        "from a beginner's perspective",
        "from an industry expert's perspective"
    ]
    
    random_focus = random.choice(random_focus_areas)
    random_perspective = random.choice(random_perspectives)
    
    # Add timestamp for additional randomness
    timestamp = int(time.time())
    
    prompt = f"""
    You are analyzing a YouTube video to find relevant comment keywords.
    
    Video URL: {video_url}
    Video Content: {video_content}
    Author: {author_id}
    Channel ID: {channel_id}
    Analysis Time: {timestamp}
    
    Author's Content Categories: {author_analysis.get('content_categories', [])}
    Audience Interests: {author_analysis.get('audience_interests', [])}
    Typical Comment Topics: {author_analysis.get('typical_comment_topics', [])}
    
    {random_focus}
    
    Based on this author's typical content and audience, suggest 5-8 specific keywords {random_perspective} that would help find relevant comments for THIS specific video.
    
    Focus on:
    - Keywords specific to this author's content style
    - Terms their audience typically uses
    - Product names, features, or topics mentioned in this video
    - Common discussion points for this type of content
    
    IMPORTANT: Respond ONLY with a valid JSON array of strings. No explanations, no extra text.
    Example format: ["price", "build quality", "Linux support", "modularity", "upgrade"]
    """
    
    # Use higher temperature for more varied responses
    response = gpt_call(prompt, temperature=0.9)
    keywords = extract_json_from_response(response, ["general feedback", "questions", "criticism"])
    
    return {
        "name": "DynamicSpec",
        "description": f"Author-specific search criteria for {author_id} (Channel: {channel_id})",
        "keywords": keywords,
        "search_type": "keyword_based",
        "priority": "high",
        "author_analysis": author_analysis,
        "channel_id": channel_id,
        "focus_area": random_focus,
        "perspective": random_perspective,
        "timestamp": timestamp
    }

# ==================================
# Task 3: Static CommentSearchSpecs
# ==================================
STATIC_SPECS = [
    {
        "name": "Positive Feedback",
        "description": "Comments expressing positive sentiment and satisfaction",
        "keywords": ["love", "amazing", "great", "awesome", "excellent", "perfect", "best", "fantastic", "good", "nice", "wonderful", "outstanding", "superb", "brilliant", "incredible"],
        "search_type": "sentiment_positive",
        "priority": "medium"
    },
    {
        "name": "Questions",
        "description": "Comments asking questions or seeking information",
        "keywords": ["?", "how", "why", "what", "when", "where", "which", "can you", "does it", "is it", "are you", "do you", "would you", "could you", "explain", "tell me", "help"],
        "search_type": "question_detection",
        "priority": "high"
    }
]

# ==================================
# Task 4: Clean Data
# ==================================
def clean_data(df):
    df['content'] = df['content'].astype(str).str.replace(r"http\S+", "", regex=True)  # remove URLs
    df['content'] = df['content'].str.strip()
    return df[df['content'] != ""]

# ==================================
# Task 5: Enhanced Search Algorithm (GPT-based)
# ==================================
def search_comments_gpt(comments, spec, sample_size=50):
    """
    Enhanced search using GPT for semantic understanding
    """
    if len(comments) == 0:
        return []
    
    # Sample comments to reduce API costs
    sample_comments = comments[:sample_size] if len(comments) > sample_size else comments
    
    # Convert to list if it's a DataFrame
    if hasattr(sample_comments, 'to_dict'):
        sample_list = sample_comments.to_dict(orient='records')
    else:
        sample_list = list(sample_comments)
    
    text_block = "\n".join([f"{i+1}. {comment.get('content', str(comment))}" for i, comment in enumerate(sample_list)])
    
    prompt = f"""
    Search for comments that match this criteria: {spec['description']}
    Keywords to look for: {', '.join(spec['keywords'])}
    
    Comments to analyze:
    {text_block}
    
    Return ONLY a JSON array of comment numbers that match the criteria.
    Example: [1, 3, 7, 12]
    """
    
    response = gpt_call(prompt)
    try:
        matched_indices = json.loads(response)
        return [sample_list[i-1] for i in matched_indices if 1 <= i <= len(sample_list)]
    except:
        # Fallback to keyword search if GPT fails
        return search_comments_keyword(sample_list, spec)

def search_comments_keyword(comments, spec):
    """
    Enhanced keyword-based search with better matching
    """
    keywords = spec.get("keywords", [])
    if not keywords:
        return []
    
    matched = []
    
    # Convert to list if it's a DataFrame
    if hasattr(comments, 'iterrows'):
        comments_list = comments.to_dict(orient='records')
    else:
        comments_list = list(comments)
    
    for comment in comments_list:
        if isinstance(comment, dict):
            text = comment.get('content', '').lower()
        else:
            text = str(comment).lower()
        
        # Skip empty content
        if not text.strip():
            continue
        
        # Check for keyword matches with multiple strategies
        for keyword in keywords:
            kw_lower = keyword.lower().strip()
            if not kw_lower:
                continue
            
            # Direct match
            if kw_lower in text:
                matched.append(comment)
                break
            
            # Word boundary match (more precise)
            if re.search(r'\b' + re.escape(kw_lower) + r'\b', text):
                matched.append(comment)
                break
            
            # Partial word match (for compound words)
            if len(kw_lower) > 3 and kw_lower in text:
                matched.append(comment)
                break
            
            # Handle common variations
            variations = [
                kw_lower + 's',      # plural
                kw_lower + 'ing',    # gerund
                kw_lower + 'ed',     # past tense
                kw_lower + 'er',     # comparative
                kw_lower + 'est',    # superlative
                kw_lower[:-1] if kw_lower.endswith('s') else None,  # singular
                kw_lower[:-3] if kw_lower.endswith('ing') else None,  # base form
            ]
            
            for variation in variations:
                if variation and variation in text:
                    matched.append(comment)
                    break
            else:
                continue
            break
    
    return matched

def search_comments_hybrid(comments, spec, use_gpt=True):
    """
    Hybrid search approach: Start with keyword search, then enhance with GPT
    """
    if len(comments) == 0:
        return []
    
    print(f"   🔍 Starting hybrid search for {len(comments)} comments...")
    
    # Step 1: Always do keyword search first (fast and reliable)
    keyword_matches = search_comments_keyword(comments, spec)
    print(f"   ✅ Keyword search found {len(keyword_matches)} matches")
    
    # Step 2: If GPT is enabled and we have comments, try to find additional matches
    if use_gpt and len(comments) > 0:
        try:
            # Convert comments to list if it's a DataFrame
            if hasattr(comments, 'to_dict'):
                comments_list = comments.to_dict(orient='records')
            else:
                comments_list = list(comments)
            
            # Sample a subset for GPT analysis (to avoid token limits)
            sample_size = min(50, len(comments_list))
            sample_comments = comments_list[:sample_size]
            
            # Create text block for GPT
            text_block = "\n".join([f"{j+1}. {comment.get('content', str(comment))}" 
                                   for j, comment in enumerate(sample_comments)])
            
            prompt = f"""
            Find comments that match this criteria: {spec['description']}
            Keywords to look for: {', '.join(spec['keywords'])}
            
            Comments to analyze (sample of {len(sample_comments)}):
            {text_block}
            
            Return ONLY a JSON array of comment numbers that match the criteria.
            Example: [1, 3, 7, 12]
            If no matches, return: []
            """
            
            response = gpt_call(prompt, temperature=0.7)
            gpt_matched_indices = extract_json_from_response(response, [])
            
            # Add GPT matches that weren't already found by keyword search
            gpt_matches = []
            for idx in gpt_matched_indices:
                if 1 <= idx <= len(sample_comments):
                    gpt_match = sample_comments[idx-1]
                    # Check if this match wasn't already found by keyword search
                    if gpt_match not in keyword_matches:
                        gpt_matches.append(gpt_match)
            
            print(f"   ✅ GPT found {len(gpt_matches)} additional matches")
            
            # Combine keyword and GPT matches
            all_matches = keyword_matches + gpt_matches
            
        except Exception as e:
            print(f"   ⚠️ GPT search failed: {e}, using keyword results only")
            all_matches = keyword_matches
    else:
        all_matches = keyword_matches
    
    print(f"   🎯 Total matches found: {len(all_matches)}")
    return all_matches

def search_comments(comments, spec, use_gpt=True, use_all_comments=False):
    """
    Main search function with hybrid approach for better reliability
    """
    if len(comments) == 0:
        return []
    
    # Always use hybrid approach for better results
    # Keyword search is fast and reliable, GPT adds semantic understanding
    return search_comments_hybrid(comments, spec, use_gpt)

# ==================================
# Task 6: Enhanced Generalizations (All Comments)
# ==================================

# 6a: Enhanced Sentiment Analysis (All Comments)
def sentiment_analysis_all(comments):
    """
    Analyze sentiment of all comments using batch processing
    """
    if not comments:
        return 0.5
    
    # Process all comments in batches
    batch_size = 50  # GPT token limit consideration
    all_scores = []
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]
        text_block = "\n".join([c['content'] for c in batch])
        
        prompt = f"""
        Analyze the sentiment of these comments (0=negative, 1=positive):
        {text_block}
        Return only a number between 0 and 1.
        """
        
        response = gpt_call(prompt)
        score = extract_number_from_response(response, 0.5)
        # Additional safety check for finite values
        if not math.isfinite(score):
            score = 0.5
        # Additional safety check
        score = max(0.0, min(1.0, score))
        all_scores.append(score)
    
    # Average all batch scores with additional safety check
    if not all_scores:
        return 0.5
    
    final_score = sum(all_scores) / len(all_scores)
    # Ensure the final score is finite and in range
    if not math.isfinite(final_score):
        final_score = 0.5
    return round(max(0.0, min(1.0, final_score)), 3)

def sentiment_analysis(comments, use_all_comments=False):
    """
    Main sentiment analysis function with option for all comments
    """
    if use_all_comments:
        return sentiment_analysis_all(comments)
    else:
        # Original sampling approach
        if not comments:
            return 0.5
        
        text_block = "\n".join([c['content'] for c in comments[:20]])
        prompt = f"""
        Analyze the overall sentiment of the following YouTube comments 
        (scale 0 = very negative, 1 = very positive):
        ---
        {text_block}
        ---
        
        IMPORTANT: Respond ONLY with a single number between 0 and 1. No explanations, no extra text.
        Example: 0.75
        """
        response = gpt_call(prompt)
        score = extract_number_from_response(response, 0.5)
        # Additional safety check for finite values
        if not math.isfinite(score):
            score = 0.5
        # Additional safety check to ensure value is between 0 and 1
        score = max(0.0, min(1.0, score))
        return round(score, 3)

# 6b: Enhanced Topics Extraction (All Comments)
def extract_topics_all(comments, top_n=5):
    """
    Extract topics from all comments using batch processing
    """
    if not comments:
        return ["general feedback"]
    
    # Process in larger batches for topic analysis
    batch_size = 100
    all_topics = []
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]
        text_block = "\n".join([c['content'] for c in batch])
        
        prompt = f"""
        Identify the main topics in these comments:
        {text_block}
        Return only a JSON array of topic strings.
        """
        
        response = gpt_call(prompt)
        batch_topics = extract_json_from_response(response, [])
        all_topics.extend(batch_topics)
    
    # Count and return top topics
    topic_counts = Counter(all_topics)
    return [topic for topic, count in topic_counts.most_common(top_n)]

def extract_topics(comments, top_n=5, use_all_comments=False):
    """
    Main topics extraction function with option for all comments
    """
    if use_all_comments:
        return extract_topics_all(comments, top_n)
    else:
        # Original sampling approach
        if not comments:
            return ["general feedback"]
        
        text_block = "\n".join([c['content'] for c in comments[:30]])
        prompt = f"""
        Identify the top {top_n} topics discussed in the following YouTube comments:
        ---
        {text_block}
        ---
        
        IMPORTANT: Respond ONLY with a valid JSON array of strings. No explanations, no extra text.
        Example format: ["topic1", "topic2", "topic3"]
        """
        response = gpt_call(prompt)
        return extract_json_from_response(response, ["general feedback"])

# 6c: Enhanced Question Extraction (All Comments)
def extract_questions_all(comments, top_n=5):
    """
    Extract questions from all comments using pattern matching
    """
    if not comments:
        return []
    
    # Pattern-based question detection for all comments
    question_patterns = [
        r'\?',  # Question mark
        r'\b(how|why|what|when|where|which|who|can you|could you|would you|do you|does it|is it|are you)\b',
        r'\b(explain|tell me|show me|help me|advice|suggestion)\b'
    ]
    
    questions = []
    for comment in comments:
        content = comment['content'].lower()
        if any(re.search(pattern, content) for pattern in question_patterns):
            questions.append(comment['content'])
    
    # Return top N questions
    return questions[:top_n]

def extract_questions(comments, top_n=5, use_all_comments=False):
    """
    Main questions extraction function with option for all comments
    """
    if use_all_comments:
        return extract_questions_all(comments, top_n)
    else:
        # Original hybrid approach
        if not comments:
            return []
        
        # Pattern-based question detection
        question_patterns = [
            r'\?',  # Question mark
            r'\b(how|why|what|when|where|which|who|can you|could you|would you|do you|does it|is it|are you)\b',
            r'\b(explain|tell me|show me|help me|advice|suggestion)\b'
        ]
        
        questions = []
        for comment in comments:
            content = comment['content'].lower()
            if any(re.search(pattern, content) for pattern in question_patterns):
                questions.append(comment['content'])
        
        # If we have enough pattern-based questions, return them
        if len(questions) >= top_n:
            return questions[:top_n]
        
        # Use LLM to identify additional questions
        text_block = "\n".join([c['content'] for c in comments[:20]])
        prompt = f"""
        Identify questions in the following YouTube comments. Look for:
        1. Direct questions with question marks
        2. Implicit questions (asking for advice, explanations, etc.)
        3. Requests for information
        
        Comments:
        ---
        {text_block}
        ---
        
        IMPORTANT: Respond ONLY with a valid JSON array of question strings. No explanations.
        Example: ["What is the price?", "How does it work?"]
        """
        
        response = gpt_call(prompt)
        llm_questions = extract_json_from_response(response, [])
        
        # Combine and deduplicate
        all_questions = list(set(questions + llm_questions))
        return all_questions[:top_n]

# ==================================
# Enhanced Web Visualization with Dataset Loading
# ==================================
def create_interactive_visualization(results=None, task_results=None):
    """
    Create an interactive web visualization with step-by-step task flow and file upload
    """
    # Use real data if provided, otherwise create demo structure
    if results is None or task_results is None:
        # Create demo structure for file upload
            results = {}
            task_results = {
            "task1": {"videos_count": 0, "comments_count": 0, "videos": []},
                "task2": {},
                "task3": {"static_specs": STATIC_SPECS},
            "task4": {"cleaned_rows": 0, "status": "pending"},
                "task5": {},
                "task6": {}
            }
    
    def generateTask1Content(task_results, results):
        """Generate HTML content for Task 1"""
        html = ""
        if 'videos' in task_results['task1'] and task_results['task1']['videos']:
            for video in task_results['task1']['videos']:
                video_id = video.get('id', 'Unknown ID')
                author = video.get('author_id', 'Unknown Author') if video.get('author_id') else 'Unknown Author'
                content = video.get('content', 'No content available')
                url = video.get('url', '')
                
                html += f"""
                <div class="video-item">
                    <div class="video-header">
                        <h5>Video ID: {video_id}</h5>
                        <p><strong>Author:</strong> {author}</p>
                        {f'<p><strong>URL:</strong> <a href="{url}" target="_blank">{url}</a></p>' if url else ''}
                    </div>
                    <div class="video-content">
                        <p><strong>Content:</strong> {content[:200]}{'...' if len(content) > 200 else ''}</p>
                    </div>
                </div>
                """
        else:
            html = "<p>No video details available</p>"
        return html
    
    def generateTask2Content(task_results, results):
        """Generate HTML content for Task 2"""
        html = ""
        for video_id, video_data in results.items():
            if video_id in task_results['task2']:
                spec = task_results['task2'][video_id]
                author = video_data.get('video_author', 'Unknown Author')
                keywords = spec.get('keywords', [])
                description = spec.get('description', '')
                
                html += f"""
                <div class="video-spec-item">
                    <h4>Video: {video_id}</h4>
                    <p><strong>Author:</strong> {author}</p>
                    <p><strong>Generated Keywords:</strong> {', '.join(keywords)}</p>
                    <p><strong>Description:</strong> {description}</p>
                </div>
                """
        return html if html else "<p>No dynamic specs available</p>"
    
    def generateTask3Content(task_results):
        """Generate HTML content for Task 3"""
        if 'static_specs' in task_results['task3']:
            specs = task_results['task3']['static_specs']
            html = "<div class='static-specs-list'>"
            for spec in specs:
                html += f"""
                <div class="static-spec-item">
                    <h4>{spec['name']}</h4>
                    <p><strong>Keywords:</strong> {', '.join(spec['keywords'])}</p>
                    <p><strong>Description:</strong> {spec.get('description', 'Universal search criteria')}</p>
                </div>
                """
            html += "</div>"
            return html
        return "<p>No static specs available</p>"
    
    def generateTask4Content(task_results):
        """Generate HTML content for Task 4"""
        cleaned_rows = task_results['task4'].get('cleaned_rows', 0)
        status = task_results['task4'].get('status', 'completed')
        
        return f"""
        <div class="cleaning-summary">
            <div class="summary-card">
                <h4>Data Cleaning Status</h4>
                <div class="status-badge {status}">{status.upper()}</div>
            </div>
            <div class="summary-card">
                <h4>Cleaned Rows</h4>
                <div class="summary-number">{cleaned_rows}</div>
            </div>
            <div class="cleaning-details">
                <h4>Cleaning Actions Performed:</h4>
                <ul>
                    <li>✅ Removed URLs from content</li>
                    <li>✅ Removed empty content entries</li>
                    <li>✅ Stripped whitespace</li>
                    <li>✅ Handled missing author IDs</li>
                </ul>
            </div>
        </div>
        """
    
    def generateTask5Content(task_results, results):
        """Generate HTML content for Task 5"""
        html = ""
        for video_id, video_data in results.items():
            if video_id in task_results['task5']:
                search_results = task_results['task5'][video_id]
                author = video_data.get('video_author', 'Unknown Author')
                dynamicMatches = len(search_results.get('dynamic', []))
                staticMatches = sum(len(matches) for matches in search_results.get('static', {}).values())
                
                html += f"""
                <div class="search-results-item">
                    <h4>Video: {video_id}</h4>
                    <p><strong>Author:</strong> {author}</p>
                    <div class="search-stats">
                        <div class="stat-item">
                            <span class="stat-label">Dynamic Matches:</span>
                            <span class="stat-value">{dynamicMatches}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Static Matches:</span>
                            <span class="stat-value">{staticMatches}</span>
                        </div>
                    </div>
                </div>
                """
        return html if html else "<p>No search results available</p>"
    
    def generateTask6Content(task_results, results):
        """Generate HTML content for Task 6"""
        html = ""
        for video_id, video_data in results.items():
            if video_id in task_results['task6']:
                task6_data = task_results['task6'][video_id]
                author = video_data.get('video_author', 'Unknown Author')
                sentiment = task6_data.get('sentiment', 0.5)
                topics = task6_data.get('topics', [])
                questions = task6_data.get('questions', [])
                totalComments = task6_data.get('total_comments_analyzed', 0)
                
                sentimentColor = getSentimentColor(sentiment)
                
                html += f"""
                <div class="generalization-item">
                    <h4>Video: {video_id}</h4>
                    <p><strong>Author:</strong> {author}</p>
                    <p><strong>Comments Analyzed:</strong> {totalComments}</p>
                    
                    <div class="generalization-metrics">
                        <div class="metric-card">
                            <h5>Sentiment Analysis</h5>
                            <div class="sentiment-score" style="color: {sentimentColor}">
                                {sentiment:.3f}
                            </div>
                            <p>Overall sentiment (0 = negative, 1 = positive)</p>
                        </div>
                        
                        <div class="metric-card">
                            <h5>Top 5 Topics</h5>
                            <ul class="topics-list">
                                {''.join([f'<li>{topic}</li>' for topic in topics[:5]])}
                            </ul>
                        </div>
                        
                        <div class="metric-card">
                            <h5>Top 5 Questions</h5>
                            <ul class="questions-list">
                                {''.join([f'<li>{q[:100]}{"..." if len(q) > 100 else ""}</li>' for q in questions[:5]])}
                            </ul>
                        </div>
                    </div>
                </div>
                """
        return html if html else "<p>No generalizations available</p>"
    
    def generateTask7Content(task_results, results):
        """Generate content for Task 7: Analysis Summary"""
        if not results:
            return "<p>No results available yet.</p>"
        
        # Calculate summary statistics
        total_videos = len(results)
        total_comments = task_results.get('task1', {}).get('comments_count', 0)
        
        # Calculate average sentiment
        sentiments = []
        total_topics = set()
        total_questions = []
        
        for video_data in results.values():
            generalizations = video_data.get('generalizations', {})
            sentiment = generalizations.get('avg_sentiment', 0.5)
            sentiments.append(sentiment)
            
            topics = generalizations.get('topics', [])
            total_topics.update(topics)
            
            questions = generalizations.get('questions', [])
            total_questions.extend(questions)
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5
        sentiment_color = '#28a745' if avg_sentiment >= 0.7 else '#ffc107' if avg_sentiment >= 0.4 else '#dc3545'
        
        html = f"""
        <div class="analysis-summary">
            <div class="summary-overview">
                <h4>📊 Analysis Overview</h4>
                <div class="summary-stats">
                    <div class="summary-stat">
                        <div class="stat-number">{total_videos}</div>
                        <div class="stat-label">Videos Analyzed</div>
                    </div>
                    <div class="summary-stat">
                        <div class="stat-number">{total_comments}</div>
                        <div class="stat-label">Comments Processed</div>
                    </div>
                    <div class="summary-stat">
                        <div class="stat-number" style="color: {sentiment_color}">{avg_sentiment:.3f}</div>
                        <div class="stat-label">Average Sentiment</div>
                    </div>
                    <div class="summary-stat">
                        <div class="stat-number">{len(total_topics)}</div>
                        <div class="stat-label">Unique Topics</div>
                    </div>
                </div>
            </div>
            
            <div class="key-findings">
                <h4>🔍 Key Findings</h4>
                <div class="findings-grid">
                    <div class="finding-card">
                        <h5>Sentiment Distribution</h5>
                        <p>Overall sentiment across all videos: <strong style="color: {sentiment_color}">{avg_sentiment:.3f}</strong></p>
                        <p>This indicates {'positive' if avg_sentiment >= 0.6 else 'neutral' if avg_sentiment >= 0.4 else 'negative'} community sentiment.</p>
                    </div>
                    
                    <div class="finding-card">
                        <h5>Top Topics</h5>
                        <ul>
                            {''.join([f'<li>{topic}</li>' for topic in list(total_topics)[:10]])}
                        </ul>
                    </div>
                    
                    <div class="finding-card">
                        <h5>Community Questions</h5>
                        <p>Total questions identified: <strong>{len(total_questions)}</strong></p>
                        <ul>
                            {''.join([f'<li>{q[:80] + "..." if len(q) > 80 else q}</li>' for q in total_questions[:5]])}
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="analysis-completion">
                <h4>✅ Analysis Complete</h4>
                <p>All tasks have been successfully completed:</p>
                <ul>
                    <li>✅ Data cleaning and separation</li>
                    <li>✅ Dynamic search specification generation</li>
                    <li>✅ Static search criteria application</li>
                    <li>✅ Comment search and matching</li>
                    <li>✅ Sentiment analysis and topic extraction</li>
                    <li>✅ Question identification and summarization</li>
                </ul>
            </div>
        </div>
        """
        return html
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Comment Analysis - Interactive</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .content {{
            padding: 30px;
        }}
        .upload-section {{
            background-color: #f8f9fa;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
            border: 2px dashed #667eea;
        }}
        .upload-section h2 {{
            margin: 0 0 20px 0;
            color: #333;
        }}
        .file-input {{
            display: none;
        }}
        .file-label {{
            display: inline-block;
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 25px;
            cursor: pointer;
            transition: transform 0.2s;
            margin: 10px;
        }}
        .file-label:hover {{
            transform: translateY(-2px);
        }}
        .analyze-button {{
            display: inline-block;
            padding: 15px 30px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.1em;
            margin: 10px;
            transition: background-color 0.2s;
        }}
        .analyze-button:hover {{
            background: #218838;
        }}
        .analyze-button:disabled {{
            background: #6c757d;
            cursor: not-allowed;
        }}
        .options-section {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid #e0e0e0;
        }}
        .option-group {{
            margin: 15px 0;
        }}
        .option-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }}
        .option-group input[type="checkbox"] {{
            margin-right: 10px;
        }}
        .progress {{
            display: none;
            margin: 20px 0;
            text-align: center;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s;
        }}
        .task-flow-section {{
            background-color: #e3f2fd;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #2196f3;
            display: none;
        }}
        .task-flow-section h2 {{
            margin: 0 0 20px 0;
            color: #1976d2;
        }}
        .task-step {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            position: relative;
            transition: transform 0.2s;
            margin-bottom: 20px;
        }}
        .task-step:hover {{
            transform: translateY(-2px);
            border-color: #2196f3;
        }}
        .task-step.active {{
            border-color: #28a745;
            background-color: #f8fff9;
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
        }}
        .task-number {{
            position: absolute;
            top: -10px;
            left: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2em;
        }}
        .task-content h4 {{
            margin: 10px 0 10px 0;
            color: #333;
            font-size: 1.1em;
        }}
        .task-content p {{
            margin: 0 0 15px 0;
            color: #666;
            line-height: 1.4;
        }}
        .task-stats {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .task-stats span {{
            background-color: #f8f9fa;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            color: #495057;
            border: 1px solid #dee2e6;
        }}
        .task-description {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            border-left: 4px solid #667eea;
        }}
        .task-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .summary-card {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }}
        .summary-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .video-item, .video-spec-item, .static-spec-item, .search-results-item, .generalization-item {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #28a745;
        }}
        .video-item h5, .video-spec-item h4, .static-spec-item h4, .search-results-item h4, .generalization-item h4 {{
            margin: 0 0 15px 0;
            color: #333;
        }}
        .video-content {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
        }}
        .static-specs-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .cleaning-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .status-badge.completed {{
            background-color: #d4edda;
            color: #155724;
        }}
        .cleaning-details {{
            grid-column: 1 / -1;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .cleaning-details ul {{
            list-style: none;
            padding: 0;
        }}
        .cleaning-details li {{
            margin: 8px 0;
            padding-left: 20px;
            position: relative;
        }}
        .search-stats {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }}
        .stat-item {{
            background-color: white;
            padding: 10px 15px;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }}
        .stat-label {{
            font-weight: bold;
            color: #666;
            margin-right: 10px;
        }}
        .stat-value {{
            color: #667eea;
            font-weight: bold;
        }}
        .generalization-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .generalization-metrics .metric-card {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .generalization-metrics h5 {{
            margin: 0 0 15px 0;
            color: #333;
            font-size: 1.1em;
        }}
        .next-button {{
            display: inline-block;
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.1em;
            margin: 20px 10px;
            transition: all 0.3s;
        }}
        .next-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        .next-button:disabled {{
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }}
        .error-message {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            display: none;
        }}
        .success-message {{
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            display: none;
        }}
        .task-panel {{
            display: none;
            animation: fadeIn 0.3s ease-in;
        }}
        .task-panel.active {{
            display: block;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>YouTube Comment Analysis</h1>
            <p>Interactive AI-powered insights from video comments</p>
        </div>
        
        <div class="content">
            <div class="upload-section" id="uploadSection">
                <h2>Upload Your Dataset</h2>
                <p>Select an Excel (.xlsx) or CSV file containing YouTube comments data</p>
                
                <input type="file" id="datasetFile" class="file-input" accept=".xlsx,.csv">
                <label for="datasetFile" class="file-label">Choose File</label>
                
                <div class="options-section">
                    <h3>Analysis Options</h3>
                    
                    <div class="option-group">
                        <label>
                            <input type="checkbox" id="useGptSearch" checked>
                            Use GPT for semantic search (more accurate, higher cost)
                        </label>
                </div>
                
                    <div class="option-group">
                        <label>
                            <input type="checkbox" id="useAllComments">
                            Analyze all comments (more accurate, higher cost)
                        </label>
                    </div>
                    
                    <div class="option-group">
                        <label>
                            <input type="checkbox" id="useSampling" checked>
                            Use sampling for faster analysis (recommended)
                        </label>
                    </div>
                </div>
                
                <button class="analyze-button" onclick="analyzeDataset()" id="analyzeBtn">Run Analysis</button>
            </div>
            
            <div class="progress" id="progress">
                <h3>Analyzing dataset...</h3>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <p id="progressText">Initializing...</p>
            </div>
            
            <div class="error-message" id="errorMessage"></div>
            <div class="success-message" id="successMessage"></div>
            
            <div class="task-flow-section" id="taskFlowSection">
                <h2>Step-by-Step Task Analysis</h2>
                <p>Follow each task to see detailed results for your videos</p>
                
                <div id="task1-step" class="task-step active">
                    <div class="task-number">1</div>
                    <div class="task-content">
                        <h4>📹 Task 1: Discover Videos in Dataset</h4>
                        <p><strong>Objective:</strong> Identify which posts are videos in the dataset</p>
                        <div class="task-stats">
                            <span>Videos: <span id="videos-count">0</span></span>
                            <span>Comments: <span id="comments-count">0</span></span>
                        </div>
                    <div id="task1-content" class="task-panel active">
                        <div class="task-description">
                            <p><strong>Objective:</strong> Identify which posts are videos in the dataset</p>
                            <div class="task-summary">
                                <div class="summary-card">
                                    <h4>Total Videos Found</h4>
                                        <div class="summary-number" id="task1-videos-count">{task_results['task1']['videos_count']}</div>
                                </div>
                                <div class="summary-card">
                                    <h4>Total Comments</h4>
                                        <div class="summary-number" id="task1-comments-count">{task_results['task1']['comments_count']}</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="videos-list">
                            <h4>Video Details:</h4>
                                <div id="task1-videos-list">
                            {generateTask1Content(task_results, results)}
                                </div>
                            </div>
                        </div>
                        <button class="next-button" onclick="showTask(2)" id="next-task1">Next: Dynamic Search Specs →</button>
                        </div>
                    </div>
                    
                <div id="task2-step" class="task-step">
                    <div class="task-number">2</div>
                    <div class="task-content">
                        <h4>🔍 Task 2: Dynamic Search Specifications</h4>
                        <p><strong>Objective:</strong> AI-generated search criteria for each video's author</p>
                    <div id="task2-content" class="task-panel">
                        <div class="task-description">
                            <p><strong>Objective:</strong> AI-generated search criteria for each video's author</p>
                        </div>
                            <div id="task2-specs-list">
                        {generateTask2Content(task_results, results)}
                            </div>
                        </div>
                        <button class="next-button" onclick="showTask(3)" id="next-task2">Next: Static Search Specs →</button>
                    </div>
                    </div>
                    
                <div id="task3-step" class="task-step">
                    <div class="task-number">3</div>
                    <div class="task-content">
                        <h4>🔍 Task 3: Static Search Specifications</h4>
                        <p><strong>Objective:</strong> Universal search criteria applied to all videos</p>
                    <div id="task3-content" class="task-panel">
                        <div class="task-description">
                            <p><strong>Objective:</strong> Universal search criteria applied to all videos</p>
                        </div>
                            <div id="task3-specs-list">
                        {generateTask3Content(task_results)}
                            </div>
                        </div>
                        <button class="next-button" onclick="showTask(4)" id="next-task3">Next: Data Cleaning →</button>
                    </div>
                    </div>
                    
                <div id="task4-step" class="task-step">
                    <div class="task-number">4</div>
                    <div class="task-content">
                        <h4>🧹 Task 4: Data Cleaning</h4>
                        <p><strong>Objective:</strong> Clean the dataset by removing URLs and empty content</p>
                    <div id="task4-content" class="task-panel">
                        <div class="task-description">
                            <p><strong>Objective:</strong> Clean the dataset by removing URLs and empty content</p>
                        </div>
                            <div id="task4-cleaning-summary">
                        {generateTask4Content(task_results)}
                            </div>
                        </div>
                        <button class="next-button" onclick="showTask(5)" id="next-task4">Next: Comment Search →</button>
                    </div>
                    </div>
                    
                <div id="task5-step" class="task-step">
                    <div class="task-number">5</div>
                    <div class="task-content">
                        <h4>🔍 Task 5: Comment Search Execution</h4>
                        <p><strong>Objective:</strong> Execute search algorithms using CommentSearchSpecs</p>
                    <div id="task5-content" class="task-panel">
                        <div class="task-description">
                            <p><strong>Objective:</strong> Execute search algorithms using CommentSearchSpecs</p>
                        </div>
                            <div id="task5-search-results">
                        {generateTask5Content(task_results, results)}
                            </div>
                        </div>
                        <button class="next-button" onclick="showTask(6)" id="next-task5">Next: Generalizations →</button>
                    </div>
                    </div>
                    
                <div id="task6-step" class="task-step">
                    <div class="task-number">6</div>
                    <div class="task-content">
                        <h4>📊 Task 6: Generalizations</h4>
                        <p><strong>Objective:</strong> Generate sentiment analysis, topics, and questions for each video</p>
                    <div id="task6-content" class="task-panel">
                        <div class="task-description">
                            <p><strong>Objective:</strong> Generate sentiment analysis, topics, and questions for each video</p>
                        </div>
                            <div id="task6-generalizations">
                        {generateTask6Content(task_results, results)}
                    </div>
                </div>

            </div>
                    </div>
            </div>
        </div>
    </div>

    <script>
        let currentFile = null;
        let currentResults = null;
        let currentTaskResults = null;
        
        document.getElementById('datasetFile').addEventListener('change', function(e) {{
            currentFile = e.target.files[0];
            if (currentFile) {{
                document.getElementById('successMessage').style.display = 'block';
                document.getElementById('successMessage').textContent = `File selected: ${{currentFile.name}}`;
            }}
        }});
        
        async function analyzeDataset() {{
            if (!currentFile) {{
                showError('Please select a file first');
                return;
            }}
            
            const useGptSearch = document.getElementById('useGptSearch').checked;
            const useAllComments = document.getElementById('useAllComments').checked;
            const useSampling = document.getElementById('useSampling').checked;
            
            // Show progress
            document.getElementById('progress').style.display = 'block';
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('errorMessage').style.display = 'none';
            document.getElementById('successMessage').style.display = 'none';
            
            try {{
                // Create FormData for file upload
                const formData = new FormData();
                formData.append('file', currentFile);
                formData.append('use_gpt_search', useGptSearch);
                formData.append('use_all_comments', useAllComments);
                formData.append('use_sampling', useSampling);
                
                // Update progress
                updateProgress(10, 'Uploading file...');
                
                // Send request to FastAPI backend
                const response = await fetch('/analyze', {{
                    method: 'POST',
                    body: formData
                }});
                
                if (!response.ok) {{
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Analysis failed');
                }}
                
                const result = await response.json();
                
                if (result.status === 'started') {{
                    // Start polling for progress
                    await pollProgress(result.analysis_id);
                }} else {{
                    throw new Error(result.message || 'Analysis failed to start');
                }}
                
            }} catch (error) {{
                showError('Error analyzing dataset: ' + error.message);
            }} finally {{
                document.getElementById('progress').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
            }}
        }}
        
        async function pollProgress(analysisId) {{
            const maxAttempts = 300; // 5 minutes max
            let attempts = 0;
            
            while (attempts < maxAttempts) {{
                try {{
                    // Get progress
                    const progressResponse = await fetch(`/progress/${{analysisId}}`);
                    if (!progressResponse.ok) {{
                        throw new Error('Failed to get progress');
                    }}
                    
                    const progress = await progressResponse.json();
                    
                    // Update progress display
                    updateProgress(progress.progress_percentage, progress.current_step);
                    
                    // Check if analysis is complete
                    if (progress.status === 'completed') {{
                        updateProgress(100, 'Analysis complete!');
                        
                        // Get final results
                        const resultsResponse = await fetch(`/results/${{analysisId}}`);
                        if (!resultsResponse.ok) {{
                            throw new Error('Failed to get results');
                        }}
                        
                        const results = await resultsResponse.json();
                        
                        // Update the page with results
                        updatePageWithResults(results);
                        return;
                    }} else if (progress.status === 'error') {{
                        throw new Error(progress.error || 'Analysis failed');
                    }}
                    
                    // Wait before next poll
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    attempts++;
                    
                }} catch (error) {{
                    throw new Error('Error polling progress: ' + error.message);
                }}
            }}
            
            throw new Error('Analysis timed out');
        }}
        
        function updatePageWithResults(results) {{
            // Store the results
            currentResults = results.results || {{}};
            currentTaskResults = results.task_results || {{}};
            
            // Show task flow section
            showTaskFlow();
            
            // Update task content
            updateTaskContent();
        }}
        
        function initializePage() {{
            // Re-attach event listeners after page update
            const fileInput = document.getElementById('datasetFile');
            if (fileInput) {{
                fileInput.addEventListener('change', function(e) {{
                    currentFile = e.target.files[0];
                    if (currentFile) {{
                        document.getElementById('successMessage').style.display = 'block';
                        document.getElementById('successMessage').textContent = `File selected: ${{currentFile.name}}`;
                    }}
                }});
            }}
        }}
        
        function updateProgress(percentage, text) {{
            document.getElementById('progressFill').style.width = percentage + '%';
            document.getElementById('progressText').textContent = text;
        }}
        
        function showError(message) {{
            document.getElementById('errorMessage').textContent = message;
            document.getElementById('errorMessage').style.display = 'block';
        }}
        
        function showTaskFlow() {{
            // Hide upload section and show task flow
            document.getElementById('uploadSection').style.display = 'none';
            document.getElementById('taskFlowSection').style.display = 'block';
            
            // Load real data if available
            const realResults = {json.dumps(results)};
            const realTaskResults = {json.dumps(task_results)};
            
            if (Object.keys(realResults).length > 0) {{
                currentResults = realResults;
                currentTaskResults = realTaskResults;
                updateTaskContent();
            }} else {{
                // Use sample data for demo
                currentResults = getSampleResults();
                currentTaskResults = getSampleTaskResults();
                updateTaskContent();
            }}
        }}
        
        function updateTaskContent() {{
            // Update task 1 content
            document.getElementById('videos-count').textContent = currentTaskResults.task1.videos_count;
            document.getElementById('comments-count').textContent = currentTaskResults.task1.comments_count;
            document.getElementById('task1-videos-count').textContent = currentTaskResults.task1.videos_count;
            document.getElementById('task1-comments-count').textContent = currentTaskResults.task1.comments_count;
            
            // Update task 1 videos list
            const videosList = document.getElementById('task1-videos-list');
            if (currentTaskResults.task1.videos && currentTaskResults.task1.videos.length > 0) {{
                videosList.innerHTML = generateVideosListHTML(currentTaskResults.task1.videos);
            }} else {{
                videosList.innerHTML = '<p>No video details available</p>';
            }}
            
            // Update other task content
            updateTask2Content();
            updateTask3Content();
            updateTask4Content();
            updateTask5Content();
            updateTask6Content();
        }}
        
        function generateVideosListHTML(videos) {{
            let html = '';
            for (const video of videos) {{
                const videoId = video.id || 'Unknown ID';
                const author = video.author_id || 'Unknown Author';
                const content = video.content || 'No content available';
                const url = video.url || '';
                
                html += '<div class="video-item">' +
                    '<div class="video-header">' +
                        '<h5>Video ID: ' + videoId + '</h5>' +
                        '<p><strong>Author:</strong> ' + author + '</p>' +
                        (url ? '<p><strong>URL:</strong> <a href="' + url + '" target="_blank">' + url + '</a></p>' : '') +
                    '</div>' +
                    '<div class="video-content">' +
                        '<p><strong>Content:</strong> ' + content.substring(0, 200) + (content.length > 200 ? '...' : '') + '</p>' +
                    '</div>' +
                '</div>';
            }}
            return html;
        }}
        
        function updateTask2Content() {{
            const specsList = document.getElementById('task2-specs-list');
            let html = '';
            
            for (const [videoId, spec] of Object.entries(currentTaskResults.task2)) {{
                const author = currentResults[videoId]?.video_author || 'Unknown Author';
                const keywords = spec.keywords || [];
                const description = spec.description || '';
                
                html += '<div class="video-spec-item">' +
                    '<h4>Video: ' + videoId + '</h4>' +
                    '<p><strong>Author:</strong> ' + author + '</p>' +
                    '<p><strong>Generated Keywords:</strong> ' + keywords.join(', ') + '</p>' +
                    '<p><strong>Description:</strong> ' + description + '</p>' +
                '</div>';
            }}
            
            specsList.innerHTML = html || '<p>No dynamic specs available</p>';
        }}
        
        function updateTask3Content() {{
            const specsList = document.getElementById('task3-specs-list');
            const specs = currentTaskResults.task3.static_specs || [];
            
            let html = '<div class="static-specs-list">';
            for (const spec of specs) {{
                html += '<div class="static-spec-item">' +
                    '<h4>' + spec.name + '</h4>' +
                    '<p><strong>Keywords:</strong> ' + spec.keywords.join(', ') + '</p>' +
                    '<p><strong>Description:</strong> ' + (spec.description || 'Universal search criteria') + '</p>' +
                '</div>';
            }}
            html += '</div>';
            
            specsList.innerHTML = html || '<p>No static specs available</p>';
        }}
        
        function updateTask4Content() {{
            const cleaningSummary = document.getElementById('task4-cleaning-summary');
            const cleanedRows = currentTaskResults.task4.cleaned_rows || 0;
            const status = currentTaskResults.task4.status || 'completed';
            
            cleaningSummary.innerHTML = '<div class="cleaning-summary">' +
                '<div class="summary-card">' +
                    '<h4>Data Cleaning Status</h4>' +
                    '<div class="status-badge ' + status + '">' + status.toUpperCase() + '</div>' +
                '</div>' +
                '<div class="summary-card">' +
                    '<h4>Cleaned Rows</h4>' +
                    '<div class="summary-number">' + cleanedRows + '</div>' +
                '</div>' +
                '<div class="cleaning-details">' +
                    '<h4>Cleaning Actions Performed:</h4>' +
                    '<ul>' +
                        '<li>✅ Removed URLs from content</li>' +
                        '<li>✅ Removed empty content entries</li>' +
                        '<li>✅ Stripped whitespace</li>' +
                        '<li>✅ Handled missing author IDs</li>' +
                    '</ul>' +
                '</div>' +
            '</div>';
        }}
        
        function updateTask5Content() {{
            const searchResults = document.getElementById('task5-search-results');
            let html = '';
            
            for (const [videoId, searchData] of Object.entries(currentTaskResults.task5)) {{
                const author = currentResults[videoId]?.video_author || 'Unknown Author';
                const dynamicMatches = searchData.dynamic?.length || 0;
                const staticMatches = Object.values(searchData.static || {{}}).reduce((sum, arr) => sum + arr.length, 0);
                
                html += '<div class="search-results-item">' +
                    '<h4>Video: ' + videoId + '</h4>' +
                    '<p><strong>Author:</strong> ' + author + '</p>' +
                    '<div class="search-stats">' +
                        '<div class="stat-item">' +
                            '<span class="stat-label">Dynamic Matches:</span>' +
                            '<span class="stat-value">' + dynamicMatches + '</span>' +
                        '</div>' +
                        '<div class="stat-item">' +
                            '<span class="stat-label">Static Matches:</span>' +
                            '<span class="stat-value">' + staticMatches + '</span>' +
                        '</div>' +
                    '</div>' +
                '</div>';
            }}
            
            searchResults.innerHTML = html || '<p>No search results available</p>';
        }}
        
        function updateTask6Content() {{
            const generalizations = document.getElementById('task6-generalizations');
            let html = '';
            
            for (const [videoId, task6Data] of Object.entries(currentTaskResults.task6)) {{
                const author = currentResults[videoId]?.video_author || 'Unknown Author';
                const sentiment = task6Data.sentiment || 0.5;
                const topics = task6Data.topics || [];
                const questions = task6Data.questions || [];
                const totalComments = task6Data.total_comments_analyzed || 0;
                
                const sentimentColor = getSentimentColor(sentiment);
                
                html += '<div class="generalization-item">' +
                    '<h4>Video: ' + videoId + '</h4>' +
                    '<p><strong>Author:</strong> ' + author + '</p>' +
                    '<p><strong>Comments Analyzed:</strong> ' + totalComments + '</p>' +
                    
                    '<div class="generalization-metrics">' +
                        '<div class="metric-card">' +
                            '<h5>Sentiment Analysis</h5>' +
                            '<div class="sentiment-score" style="color: ' + sentimentColor + '">' +
                                sentiment.toFixed(3) +
                            '</div>' +
                            '<p>Overall sentiment (0 = negative, 1 = positive)</p>' +
                        '</div>' +
                        
                        '<div class="metric-card">' +
                            '<h5>Top 5 Topics</h5>' +
                            '<ul class="topics-list">' +
                                topics.slice(0, 5).map(topic => '<li>' + topic + '</li>').join('') +
                            '</ul>' +
                        '</div>' +
                        
                        '<div class="metric-card">' +
                            '<h5>Top 5 Questions</h5>' +
                            '<ul class="questions-list">' +
                                questions.slice(0, 5).map(q => '<li>' + q.substring(0, 100) + (q.length > 100 ? '...' : '') + '</li>').join('') +
                            '</ul>' +
                        '</div>' +
                    '</div>' +
                '</div>';
            }}
            
            generalizations.innerHTML = html || '<p>No generalizations available</p>';
        }}
        
        function getSentimentColor(sentiment) {{
            if (sentiment >= 0.7) return '#28a745';
            if (sentiment >= 0.4) return '#ffc107';
            return '#dc3545';
        }}
        
        function getSampleResults() {{
            return {{
                "FqIMu4C87SM": {{
                    "video_author": "TechReviewer123",
                    "video_content": "Framework laptop review and analysis...",
                    "generalizations": {{
                        "avg_sentiment": 0.623,
                        "topics": ["Framework laptop", "Price analysis", "Modularity", "Build quality", "Linux support"],
                        "questions": ["What is the actual price?", "How does the modularity work?", "Is it worth the cost?", "Can you upgrade the RAM?", "Does it work with Linux?"]
                    }}
                }}
            }};
        }}
        
        function getSampleTaskResults() {{
            return {{
                "task1": {{"videos_count": 1, "comments_count": 100, "videos": [{{"id": "FqIMu4C87SM", "author_id": "TechReviewer123", "content": "Framework laptop review...", "url": ""}}]}},
                "task2": {{"FqIMu4C87SM": {{"keywords": ["Framework", "laptop", "review"], "description": "Sample dynamic spec"}}}},
                "task3": {{"static_specs": {json.dumps(STATIC_SPECS)}}},
                "task4": {{"cleaned_rows": 100, "status": "completed"}},
                "task5": {{"FqIMu4C87SM": {{"dynamic": [], "static": {{}}}}}},
                "task6": {{"FqIMu4C87SM": {{"sentiment": 0.623, "topics": ["Framework laptop"], "questions": ["What is the price?"], "total_comments_analyzed": 100}}}}
            }};
        }}
        
        function sleep(ms) {{
            return new Promise(resolve => setTimeout(resolve, ms));
        }}
        
        function showTask(taskNumber) {{
            // Hide all task panels
            const panels = document.querySelectorAll('.task-panel');
            panels.forEach(panel => panel.classList.remove('active'));
            
            // Remove active class from all task steps
            const taskSteps = document.querySelectorAll('.task-step');
            taskSteps.forEach(step => step.classList.remove('active'));
            
            // Show selected task panel
            const selectedPanel = document.getElementById(`task${{taskNumber}}-content`);
            if (selectedPanel) {{
                selectedPanel.classList.add('active');
            }}
            
            // Add active class to selected task step
            const selectedTaskStep = document.getElementById(`task${{taskNumber}}-step`);
            if (selectedTaskStep) {{
                selectedTaskStep.classList.add('active');
                selectedTaskStep.scrollIntoView({{ behavior: 'smooth' }});
            }}
        }}
        
        function showResults(results) {{
            // Hide progress and upload sections
            document.getElementById('liveProgress').style.display = 'none';
            document.getElementById('uploadSection').style.display = 'none';
            
            // Show the task flow section
            const taskFlowSection = document.getElementById('taskFlowSection');
            taskFlowSection.style.display = 'block';
            
            // Update the task results with actual data
            updateTaskResults(results.results, results.task_results);
            
            // Show Task 1 by default
            showTask(1);
        }}
        
        function updateTaskResults(results, task_results) {{
            // Update Task 1 content
            const task1Content = document.getElementById('task1-content');
            if (task1Content && task_results.task1) {{
                const videosCount = task_results.task1.videos_count || 0;
                const commentsCount = task_results.task1.comments_count || 0;
                
                task1Content.innerHTML = '<h3>Task 1: Data Cleaning & Separation</h3>' +
                    '<div class="task-summary">' +
                        '<div class="summary-card">' +
                            '<h4>Videos Found</h4>' +
                            '<div class="summary-number">' + videosCount + '</div>' +
                        '</div>' +
                        '<div class="summary-card">' +
                            '<h4>Comments Found</h4>' +
                            '<div class="summary-number">' + commentsCount + '</div>' +
                        '</div>' +
                    '</div>' +
                    '<div class="task-details">' +
                        '<h4>Analysis Summary:</h4>' +
                        '<p>✅ Successfully processed ' + videosCount + ' videos and ' + commentsCount + ' comments</p>' +
                        '<p>✅ Data cleaned and separated into videos and comments</p>' +
                        '<p>✅ Ready for analysis</p>' +
                    '</div>';
            }}
            
            // Update Task 2 content
            const task2Content = document.getElementById('task2-content');
            if (task2Content && task_results.task2) {{
                let dynamicSpecsHtml = '<h3>Task 2: Dynamic Search Specification Generation</h3>';
                dynamicSpecsHtml += '<p>AI-generated search criteria based on video content and author information.</p>';
                
                for (const [videoId, spec] of Object.entries(task_results.task2)) {{
                    const keywords = spec.keywords ? spec.keywords.join(', ') : 'No keywords';
                    const description = spec.description || 'No description';
                    dynamicSpecsHtml += `
                        <div class="video-spec-item">
                            <h4>Video: ${{videoId}}</h4>
                            <p><strong>Generated Keywords:</strong> ${{keywords}}</p>
                            <p><strong>Description:</strong> ${{description}}</p>
                        </div>
                    `;
                }}
                
                task2Content.innerHTML = dynamicSpecsHtml;
            }}
            
            // Update Task 3 content
            const task3Content = document.getElementById('task3-content');
            if (task3Content && task_results.task3) {{
                let staticSpecsHtml = '<h3>Task 3: Static Search Specifications</h3>';
                staticSpecsHtml += '<p>Pre-defined search criteria for consistent analysis across all videos.</p>';
                
                if (task_results.task3.static_specs) {{
                    staticSpecsHtml += '<div class="static-specs-list">';
                    for (const spec of task_results.task3.static_specs) {{
                        const keywords = spec.keywords ? spec.keywords.join(', ') : 'No keywords';
                        staticSpecsHtml += `
                            <div class="static-spec-item">
                                <h4>${{spec.name}}</h4>
                                <p><strong>Keywords:</strong> ${{keywords}}</p>
                                <p><strong>Description:</strong> ${{spec.description || 'Universal search criteria'}}</p>
                            </div>
                        `;
                    }}
                    staticSpecsHtml += '</div>';
                }}
                
                task3Content.innerHTML = staticSpecsHtml;
            }}
            
            // Update Task 4 content
            const task4Content = document.getElementById('task4-content');
            if (task4Content && task_results.task4) {{
                const cleanedRows = task_results.task4.cleaned_rows || 0;
                const status = task_results.task4.status || 'completed';
                
                task4Content.innerHTML = `
                    <h3>Task 4: Data Cleaning Summary</h3>
                    <div class="cleaning-summary">
                        <div class="summary-card">
                            <h4>Data Cleaning Status</h4>
                            <div class="status-badge ${{status}}">${{status.toUpperCase()}}</div>
                        </div>
                        <div class="summary-card">
                            <h4>Cleaned Rows</h4>
                            <div class="summary-number">${{cleanedRows}}</div>
                        </div>
                        <div class="cleaning-details">
                            <h4>Cleaning Actions Performed:</h4>
                            <ul>
                                <li>✅ Removed URLs from content</li>
                                <li>✅ Removed empty content entries</li>
                                <li>✅ Stripped whitespace</li>
                                <li>✅ Handled missing author IDs</li>
                            </ul>
                        </div>
                    </div>
                `;
            }}
            
            // Update Task 5 content
            const task5Content = document.getElementById('task5-content');
            if (task5Content && task_results.task5) {{
                let searchResultsHtml = '<h3>Task 5: Comment Search Results</h3>';
                searchResultsHtml += '<p>Comments that match the generated search specifications.</p>';
                
                for (const [videoId, searchResults] of Object.entries(task_results.task5)) {{
                    const dynamicMatches = searchResults.dynamic ? searchResults.dynamic.length : 0;
                    const staticMatches = searchResults.static ? 
                        Object.values(searchResults.static).reduce((sum, matches) => sum + (matches ? matches.length : 0), 0) : 0;
                    
                    searchResultsHtml += `
                        <div class="search-results-item">
                            <h4>Video: ${{videoId}}</h4>
                            <div class="search-stats">
                                <div class="stat-item">
                                    <span class="stat-label">Dynamic Matches:</span>
                                    <span class="stat-value">${{dynamicMatches}}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Static Matches:</span>
                                    <span class="stat-value">${{staticMatches}}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }}
                
                task5Content.innerHTML = searchResultsHtml;
            }}
            
            // Update Task 6 content
            const task6Content = document.getElementById('task6-content');
            if (task6Content && task_results.task6) {{
                let generalizationsHtml = '<h3>Task 6: Generalizations & Insights</h3>';
                generalizationsHtml += '<p>AI-generated insights including sentiment analysis, topic extraction, and question identification.</p>';
                
                for (const [videoId, task6Data] of Object.entries(task_results.task6)) {{
                    const sentiment = task6Data.sentiment || 0.5;
                    const topics = task6Data.topics || [];
                    const questions = task6Data.questions || [];
                    const totalComments = task6Data.total_comments_analyzed || 0;
                    
                    const sentimentColor = sentiment >= 0.7 ? '#28a745' : sentiment >= 0.4 ? '#ffc107' : '#dc3545';
                    
                    generalizationsHtml += `
                        <div class="generalization-item">
                            <h4>Video: ${{videoId}}</h4>
                            <p><strong>Comments Analyzed:</strong> ${{totalComments}}</p>
                            
                            <div class="generalization-metrics">
                                <div class="metric-card">
                                    <h5>Sentiment Analysis</h5>
                                    <div class="sentiment-score" style="color: ${{sentimentColor}}">
                                        ${{sentiment.toFixed(3)}}
                                    </div>
                                    <p>Overall sentiment (0 = negative, 1 = positive)</p>
                                </div>
                                
                                <div class="metric-card">
                                    <h5>Top 5 Topics</h5>
                                    <ul class="topics-list">
                                        ${{topics.slice(0, 5).map(topic => `<li>${{topic}}</li>`).join('')}}
                                    </ul>
                                </div>
                                
                                <div class="metric-card">
                                    <h5>Top 5 Questions</h5>
                                    <ul class="questions-list">
                                        ${{questions.slice(0, 5).map(q => `<li>${{q.length > 100 ? q.substring(0, 100) + '...' : q}}</li>`).join('')}}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    `;
                }}
                
                task6Content.innerHTML = generalizationsHtml;
            }}
            

        }}
    </script>
</body>
</html>
"""
    
    return html_content

# ==================================
# Enhanced Interactive Visualization with Real-time Progress
# ==================================
def create_interactive_visualization_enhanced(results=None, task_results=None):
    """
    Create an enhanced interactive web visualization with real-time progress tracking
    """
    # Use real data if provided, otherwise create demo structure
    if results is None or task_results is None:
        results = {}
        task_results = {
            "task1": {"videos_count": 0, "comments_count": 0, "videos": []},
            "task2": {},
            "task3": {"static_specs": STATIC_SPECS},
            "task4": {"cleaned_rows": 0, "status": "pending"},
            "task5": {},
            "task6": {}
        }
    
    # Helper functions for generating content
    def generateTask1Content(task_results, results):
        """Generate HTML content for Task 1"""
        html = ""
        if 'videos' in task_results['task1'] and task_results['task1']['videos']:
            for video in task_results['task1']['videos']:
                video_id = video.get('id', 'Unknown ID')
                author = video.get('author_id', 'Unknown Author') if video.get('author_id') else 'Unknown Author'
                content = video.get('content', 'No content available')
                url = video.get('url', '')
                
                html += f"""
                <div class="video-item">
                    <div class="video-header">
                        <h5>Video ID: {video_id}</h5>
                        <p><strong>Author:</strong> {author}</p>
                        {f'<p><strong>URL:</strong> <a href="{url}" target="_blank">{url}</a></p>' if url else ''}
                    </div>
                    <div class="video-content">
                        <p><strong>Content:</strong> {content[:200]}{'...' if len(content) > 200 else ''}</p>
                    </div>
                </div>
                """
        else:
            html = "<p>No video details available</p>"
        return html
    
    def generateTask2Content(task_results, results):
        """Generate HTML content for Task 2"""
        html = ""
        for video_id, video_data in results.items():
            if video_id in task_results['task2']:
                spec = task_results['task2'][video_id]
                author = video_data.get('video_author', 'Unknown Author')
                keywords = spec.get('keywords', [])
                description = spec.get('description', '')
                
                html += f"""
                <div class="video-spec-item">
                    <h4>Video: {video_id}</h4>
                    <p><strong>Author:</strong> {author}</p>
                    <p><strong>Generated Keywords:</strong> {', '.join(keywords)}</p>
                    <p><strong>Description:</strong> {description}</p>
                </div>
                """
        return html if html else "<p>No dynamic specs available</p>"
    
    def generateTask3Content(task_results):
        """Generate HTML content for Task 3"""
        if 'static_specs' in task_results['task3']:
            specs = task_results['task3']['static_specs']
            html = "<div class='static-specs-list'>"
            for spec in specs:
                html += f"""
                <div class="static-spec-item">
                    <h4>{spec['name']}</h4>
                    <p><strong>Keywords:</strong> {', '.join(spec['keywords'])}</p>
                    <p><strong>Description:</strong> {spec.get('description', 'Universal search criteria')}</p>
                </div>
                """
            html += "</div>"
            return html
        return "<p>No static specs available</p>"
    
    def generateTask4Content(task_results):
        """Generate HTML content for Task 4"""
        cleaned_rows = task_results['task4'].get('cleaned_rows', 0)
        status = task_results['task4'].get('status', 'completed')
        
        return f"""
        <div class="cleaning-summary">
            <div class="summary-card">
                <h4>Data Cleaning Status</h4>
                <div class="status-badge {status}">{status.upper()}</div>
            </div>
            <div class="summary-card">
                <h4>Cleaned Rows</h4>
                <div class="summary-number">{cleaned_rows}</div>
            </div>
            <div class="cleaning-details">
                <h4>Cleaning Actions Performed:</h4>
                <ul>
                    <li>✅ Removed URLs from content</li>
                    <li>✅ Removed empty content entries</li>
                    <li>✅ Stripped whitespace</li>
                    <li>✅ Handled missing author IDs</li>
                </ul>
            </div>
        </div>
        """
    
    def generateTask5Content(task_results, results):
        """Generate HTML content for Task 5"""
        html = ""
        for video_id, video_data in results.items():
            if video_id in task_results['task5']:
                search_results = task_results['task5'][video_id]
                author = video_data.get('video_author', 'Unknown Author')
                dynamic_matches = len(search_results.get('dynamic', []))
                static_matches = sum(len(matches) for matches in search_results.get('static', {}).values())
                
                html += f"""
                <div class="search-results-item">
                    <h4>Video: {video_id}</h4>
                    <p><strong>Author:</strong> {author}</p>
                    <div class="search-stats">
                        <div class="stat-item">
                            <span class="stat-label">Dynamic Matches:</span>
                            <span class="stat-value">{dynamic_matches}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Static Matches:</span>
                            <span class="stat-value">{static_matches}</span>
                        </div>
                    </div>
                </div>
                """
        return html if html else "<p>No search results available</p>"
    
    def generateTask6Content(task_results, results):
        """Generate HTML content for Task 6"""
        html = ""
        for video_id, video_data in results.items():
            if video_id in task_results['task6']:
                task6_data = task_results['task6'][video_id]
                author = video_data.get('video_author', 'Unknown Author')
                sentiment = task6_data.get('sentiment', 0.5)
                topics = task6_data.get('topics', [])
                questions = task6_data.get('questions', [])
                total_comments = task6_data.get('total_comments_analyzed', 0)
                
                sentiment_color = '#28a745' if sentiment >= 0.7 else '#ffc107' if sentiment >= 0.4 else '#dc3545'
                
                html += f"""
                <div class="generalization-item">
                    <h4>Video: {video_id}</h4>
                    <p><strong>Author:</strong> {author}</p>
                    <p><strong>Comments Analyzed:</strong> {total_comments}</p>
                    
                    <div class="generalization-metrics">
                        <div class="metric-card">
                            <h5>Sentiment Analysis</h5>
                            <div class="sentiment-score" style="color: {sentiment_color}">
                                {sentiment:.3f}
                            </div>
                            <p>Overall sentiment (0 = negative, 1 = positive)</p>
                        </div>
                        
                        <div class="metric-card">
                            <h5>Top 5 Topics</h5>
                            <ul class="topics-list">
                                {''.join([f'<li>{topic}</li>' for topic in topics[:5]])}
                            </ul>
                        </div>
                        
                        <div class="metric-card">
                            <h5>Top 5 Questions</h5>
                            <ul class="questions-list">
                                {''.join([f'<li>{q[:100]}{"..." if len(q) > 100 else ""}</li>' for q in questions[:5]])}
                            </ul>
                        </div>
                    </div>
                </div>
                """
        return html if html else "<p>No generalizations available</p>"
    
    def generateTask7Content(task_results, results):
        """Generate content for Task 7: Analysis Summary"""
        if not results:
            return "<p>No results available yet.</p>"
        
        # Calculate summary statistics
        total_videos = len(results)
        total_comments = task_results.get('task1', {}).get('comments_count', 0)
        
        # Calculate average sentiment
        sentiments = []
        total_topics = set()
        total_questions = []
        
        for video_data in results.values():
            generalizations = video_data.get('generalizations', {})
            sentiment = generalizations.get('avg_sentiment', 0.5)
            sentiments.append(sentiment)
            
            topics = generalizations.get('topics', [])
            total_topics.update(topics)
            
            questions = generalizations.get('questions', [])
            total_questions.extend(questions)
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5
        sentiment_color = '#28a745' if avg_sentiment >= 0.7 else '#ffc107' if avg_sentiment >= 0.4 else '#dc3545'
        
        html = f"""
        <div class="analysis-summary">
            <div class="summary-overview">
                <h4>📊 Analysis Overview</h4>
                <div class="summary-stats">
                    <div class="summary-stat">
                        <div class="stat-number">{total_videos}</div>
                        <div class="stat-label">Videos Analyzed</div>
                    </div>
                    <div class="summary-stat">
                        <div class="stat-number">{total_comments}</div>
                        <div class="stat-label">Comments Processed</div>
                    </div>
                    <div class="summary-stat">
                        <div class="stat-number" style="color: {sentiment_color}">{avg_sentiment:.3f}</div>
                        <div class="stat-label">Average Sentiment</div>
                    </div>
                    <div class="summary-stat">
                        <div class="stat-number">{len(total_topics)}</div>
                        <div class="stat-label">Unique Topics</div>
                    </div>
                </div>
            </div>
            
            <div class="key-findings">
                <h4>🔍 Key Findings</h4>
                <div class="findings-grid">
                    <div class="finding-card">
                        <h5>Sentiment Distribution</h5>
                        <p>Overall sentiment across all videos: <strong style="color: {sentiment_color}">{avg_sentiment:.3f}</strong></p>
                        <p>This indicates {'positive' if avg_sentiment >= 0.6 else 'neutral' if avg_sentiment >= 0.4 else 'negative'} community sentiment.</p>
                    </div>
                    
                    <div class="finding-card">
                        <h5>Top Topics</h5>
                        <ul>
                            {''.join([f'<li>{topic}</li>' for topic in list(total_topics)[:10]])}
                        </ul>
                    </div>
                    
                    <div class="finding-card">
                        <h5>Community Questions</h5>
                        <p>Total questions identified: <strong>{len(total_questions)}</strong></p>
                        <ul>
                            {''.join([f'<li>{q[:80] + "..." if len(q) > 80 else q}</li>' for q in total_questions[:5]])}
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="analysis-completion">
                <h4>✅ Analysis Complete</h4>
                <p>All tasks have been successfully completed:</p>
                <ul>
                    <li>✅ Data cleaning and separation</li>
                    <li>✅ Dynamic search specification generation</li>
                    <li>✅ Static search criteria application</li>
                    <li>✅ Comment search and matching</li>
                    <li>✅ Sentiment analysis and topic extraction</li>
                    <li>✅ Question identification and summarization</li>
                </ul>
            </div>
        </div>
        """
        return html
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Comment Analysis - Interactive</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .upload-section {{
            text-align: center;
            padding: 40px;
            border: 2px dashed #ddd;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .file-input {{
            display: none;
        }}
        .file-label {{
            display: inline-block;
            padding: 12px 24px;
            background-color: #007bff;
            color: white;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }}
        .file-label:hover {{
            background-color: #0056b3;
        }}
        .options-section {{
            margin: 30px 0;
            text-align: left;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }}
        .option-group {{
            margin: 15px 0;
            padding: 15px;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            background-color: #f8f9fa;
        }}
        .analyze-button {{
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 1.1em;
            border-radius: 5px;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .analyze-button:hover {{
            transform: translateY(-2px);
        }}
        .analyze-button:disabled {{
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }}
        .live-progress {{
            background-color: #e8f5e8;
            border: 2px solid #28a745;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            display: none;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 15px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            width: 0%;
            transition: width 0.3s ease;
        }}
        .progress-log {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            margin-top: 15px;
        }}
        .log-entry {{
            margin: 2px 0;
            padding: 2px 0;
        }}
        .log-entry.success {{
            color: #28a745;
        }}
        .log-entry.warning {{
            color: #ffc107;
        }}
        .log-entry.error {{
            color: #dc3545;
        }}
        .live-status {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .status-indicator {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #6c757d;
            animation: pulse 2s infinite;
        }}
        .status-indicator.running {{
            background-color: #28a745;
        }}
        .status-indicator.completed {{
            background-color: #007bff;
        }}
        .status-indicator.error {{
            background-color: #dc3545;
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        .error-message {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            display: none;
        }}
        .success-message {{
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            display: none;
        }}
        .task-flow-section {{
            display: none;
        }}
        .task-steps {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .task-step {{
            flex: 1;
            text-align: center;
            padding: 15px;
            margin: 0 5px;
            background-color: #f8f9fa;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            min-width: 120px;
        }}
        .task-step:hover {{
            background-color: #e9ecef;
            transform: translateY(-2px);
        }}
        .task-step.active {{
            background: linear-gradient(135deg, #007bff, #0056b3);
            color: white;
        }}
        .task-panel {{
            display: none;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .task-panel.active {{
            display: block;
        }}
        .video-item, .video-spec-item, .static-spec-item, .search-results-item, .generalization-item {{
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .summary-card {{
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 10px;
            text-align: center;
            display: inline-block;
            min-width: 200px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .status-badge.completed {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status-badge.pending {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .summary-number {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .search-stats {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }}
        .stat-item {{
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-radius: 5px;
            flex: 1;
        }}
        .stat-label {{
            font-weight: bold;
            color: #6c757d;
        }}
        .stat-value {{
            font-size: 1.2em;
            color: #007bff;
            font-weight: bold;
        }}
        .generalization-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .metric-card {{
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .sentiment-score {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .topics-list, .questions-list {{
            list-style: none;
            padding: 0;
            text-align: left;
        }}
        .topics-list li, .questions-list li {{
            padding: 5px 0;
            border-bottom: 1px solid #f1f1f1;
        }}
        .topics-list li:last-child, .questions-list li:last-child {{
            border-bottom: none;
        }}
        
        /* Analysis Summary Styles */
        .analysis-summary {{
            padding: 20px;
        }}
        .summary-overview {{
            margin-bottom: 30px;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .summary-stat {{
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 10px;
        }}
        .stat-label {{
            font-size: 1.1em;
            color: #6c757d;
        }}
        .key-findings {{
            margin-bottom: 30px;
        }}
        .findings-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .finding-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .finding-card h5 {{
            color: #007bff;
            margin-bottom: 15px;
        }}
        .analysis-completion {{
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .analysis-completion h4 {{
            margin-bottom: 15px;
        }}
        .analysis-completion ul {{
            text-align: left;
            max-width: 600px;
            margin: 0 auto;
        }}
        .analysis-completion li {{
            margin: 8px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>YouTube Comment Analysis</h1>
            <p>Real-time interactive AI-powered insights from video comments</p>
        </div>
        
        <div class="content">
            <div class="upload-section" id="uploadSection">
                <h2>Upload Your Dataset</h2>
                <p>Select an Excel (.xlsx) or CSV file containing YouTube comments data</p>
                
                <input type="file" id="datasetFile" class="file-input" accept=".xlsx,.csv">
                <label for="datasetFile" class="file-label">Choose File</label>
                
                <button class="analyze-button" onclick="startAnalysis()" id="analyzeBtn">Start Analysis</button>
            </div>
            
            <div class="live-progress" id="liveProgress">
                <h3>🔄 Live Analysis Progress</h3>
                <div class="live-status">
                    <div class="status-indicator" id="statusIndicator"></div>
                    <span id="statusText">Initializing...</span>
                </div>
                
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <p id="progressText">Starting analysis...</p>
                
                <div class="progress-log" id="progressLog">
                    <div class="log-entry">Waiting for analysis to start...</div>
                </div>
            </div>
            
            <div class="error-message" id="errorMessage"></div>
            <div class="success-message" id="successMessage"></div>
            
            <div class="task-flow-section" id="taskFlowSection">
                <div class="task-steps">
                    <div class="task-step active" onclick="showTask(1)" id="task1-step">
                        <h4>Task 1</h4>
                        <p>Data Cleaning</p>
                    </div>
                    <div class="task-step" onclick="showTask(2)" id="task2-step">
                        <h4>Task 2</h4>
                        <p>Dynamic Specs</p>
                    </div>
                    <div class="task-step" onclick="showTask(3)" id="task3-step">
                        <h4>Task 3</h4>
                        <p>Static Specs</p>
                    </div>
                    <div class="task-step" onclick="showTask(4)" id="task4-step">
                        <h4>Task 4</h4>
                        <p>Data Summary</p>
                    </div>
                    <div class="task-step" onclick="showTask(5)" id="task5-step">
                        <h4>Task 5</h4>
                        <p>Search Results</p>
                    </div>
                    <div class="task-step" onclick="showTask(6)" id="task6-step">
                        <h4>Task 6</h4>
                        <p>Generalizations</p>
                    </div>
                </div>
                
                <div class="task-panel active" id="task1-content">
                    <h3>Task 1: Data Cleaning & Separation</h3>
                    <div class="task-summary">
                        <div class="summary-card">
                            <h4>Videos Found</h4>
                            <div class="summary-number">{task_results['task1']['videos_count']}</div>
                        </div>
                        <div class="summary-card">
                            <h4>Comments Found</h4>
                            <div class="summary-number">{task_results['task1']['comments_count']}</div>
                        </div>
                    </div>
                    <div class="task-details">
                        <h4>Sample Videos:</h4>
                        {generateTask1Content(task_results, results)}
                    </div>
                </div>
                
                <div class="task-panel" id="task2-content">
                    <h3>Task 2: Dynamic Search Specification Generation</h3>
                    <p>AI-generated search criteria based on video content and author information.</p>
                    {generateTask2Content(task_results, results)}
                </div>
                
                <div class="task-panel" id="task3-content">
                    <h3>Task 3: Static Search Specifications</h3>
                    <p>Pre-defined search criteria for consistent analysis across all videos.</p>
                    {generateTask3Content(task_results)}
                </div>
                
                <div class="task-panel" id="task4-content">
                    <h3>Task 4: Data Cleaning Summary</h3>
                    {generateTask4Content(task_results)}
                </div>
                
                <div class="task-panel" id="task5-content">
                    <h3>Task 5: Comment Search Results</h3>
                    <p>Comments that match the generated search specifications.</p>
                    {generateTask5Content(task_results, results)}
                </div>
                
                <div class="task-panel" id="task6-content">
                    <h3>Task 6: Generalizations & Insights</h3>
                    <p>AI-generated insights including sentiment analysis, topic extraction, and question identification.</p>
                    {generateTask6Content(task_results, results)}
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFile = null;
        let currentAnalysisId = null;
        let progressInterval = null;
        
        document.getElementById('datasetFile').addEventListener('change', function(e) {{
            currentFile = e.target.files[0];
            if (currentFile) {{
                document.getElementById('successMessage').style.display = 'block';
                document.getElementById('successMessage').textContent = `File selected: ${{currentFile.name}}`;
            }}
        }});
        
        async function startAnalysis() {{
            if (!currentFile) {{
                showError('Please select a file first');
                return;
            }}
            
            // Use default optimized settings
            const useGptSearch = true;  // Use GPT for better accuracy
            const useAllComments = false;  // Use sampling for faster analysis
            const useSampling = true;  // Enable sampling for performance
            
            // Show live progress
            document.getElementById('liveProgress').style.display = 'block';
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('errorMessage').style.display = 'none';
            document.getElementById('successMessage').style.display = 'none';
            
            // Reset progress
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('progressText').textContent = 'Starting analysis...';
            document.getElementById('progressLog').innerHTML = '<div class="log-entry">Initializing analysis...</div>';
            
            try {{
                // Create FormData for file upload
                const formData = new FormData();
                formData.append('file', currentFile);
                formData.append('use_gpt_search', useGptSearch);
                formData.append('use_all_comments', useAllComments);
                formData.append('use_sampling', useSampling);
                
                // Start analysis
                const response = await fetch('/analyze', {{
                    method: 'POST',
                    body: formData
                }});
                
                if (!response.ok) {{
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Failed to start analysis');
                }}
                
                const result = await response.json();
                currentAnalysisId = result.analysis_id;
                
                // Start progress polling
                startProgressPolling();
                
            }} catch (error) {{
                showError('Error starting analysis: ' + error.message);
                document.getElementById('liveProgress').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
            }}
        }}
        
        function startProgressPolling() {{
            if (progressInterval) {{
                clearInterval(progressInterval);
            }}
            
            progressInterval = setInterval(async () => {{
                try {{
                    const response = await fetch(`/progress/${{currentAnalysisId}}`);
                    if (response.ok) {{
                        const progress = await response.json();
                        updateProgress(progress);
                        
                        if (progress.status === 'completed') {{
                            clearInterval(progressInterval);
                            showResults(progress.results);
                        }} else if (progress.status === 'error') {{
                            clearInterval(progressInterval);
                            showError('Analysis failed: ' + progress.error);
                        }}
                    }}
                }} catch (error) {{
                    console.error('Error fetching progress:', error);
                }}
            }}, 1000); // Poll every second
        }}
        
        function updateProgress(progress) {{
            // Update progress bar
            document.getElementById('progressFill').style.width = progress.progress_percentage + '%';
            document.getElementById('progressText').textContent = `${{progress.current_task}}: ${{progress.current_step}}`;
            
            // Update status
            const statusIndicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');
            
            statusIndicator.className = 'status-indicator ' + progress.status;
            
            if (progress.status === 'starting') {{
                statusText.textContent = 'Starting...';
            }} else if (progress.status === 'completed') {{
                statusText.textContent = 'Completed!';
            }} else if (progress.status === 'error') {{
                statusText.textContent = 'Error occurred';
            }} else {{
                statusText.textContent = 'Running...';
            }}
            
            // Update log
            const logContainer = document.getElementById('progressLog');
            logContainer.innerHTML = '';
            
            progress.logs.forEach(log => {{
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';
                
                if (log.includes('✅')) {{
                    logEntry.classList.add('success');
                }} else if (log.includes('⚠️')) {{
                    logEntry.classList.add('warning');
                }} else if (log.includes('❌')) {{
                    logEntry.classList.add('error');
                }}
                
                logEntry.textContent = log;
                logContainer.appendChild(logEntry);
            }});
            
            // Auto-scroll to bottom
            logContainer.scrollTop = logContainer.scrollHeight;
        }}
        
        function showResults(results) {{
            // Hide progress and upload sections
            document.getElementById('liveProgress').style.display = 'none';
            document.getElementById('uploadSection').style.display = 'none';
            
            // Show the task flow section
            const taskFlowSection = document.getElementById('taskFlowSection');
            taskFlowSection.style.display = 'block';
            
            // Update the task results with actual data
            updateTaskResults(results.results, results.task_results);
            
            // Show Task 1 by default
            showTask(1);
        }}
        
        function updateTaskResults(results, task_results) {{
            // Update Task 1 content
            const task1Content = document.getElementById('task1-content');
            if (task1Content && task_results.task1) {{
                const videosCount = task_results.task1.videos_count || 0;
                const commentsCount = task_results.task1.comments_count || 0;
                
                task1Content.innerHTML = `
                    <h3>Task 1: Data Cleaning & Separation</h3>
                    <div class="task-summary">
                        <div class="summary-card">
                            <h4>Videos Found</h4>
                            <div class="summary-number">${{videosCount}}</div>
                        </div>
                        <div class="summary-card">
                            <h4>Comments Found</h4>
                            <div class="summary-number">${{commentsCount}}</div>
                        </div>
                    </div>
                    <div class="task-details">
                        <h4>Analysis Summary:</h4>
                        <p>✅ Successfully processed ${{videosCount}} videos and ${{commentsCount}} comments</p>
                        <p>✅ Data cleaned and separated into videos and comments</p>
                        <p>✅ Ready for analysis</p>
                    </div>
                `;
            }}
            
            // Update Task 2 content
            const task2Content = document.getElementById('task2-content');
            if (task2Content && task_results.task2) {{
                let dynamicSpecsHtml = '<h3>Task 2: Dynamic Search Specification Generation</h3>';
                dynamicSpecsHtml += '<p>AI-generated search criteria based on video content and author information.</p>';
                
                for (const [videoId, spec] of Object.entries(task_results.task2)) {{
                    const keywords = spec.keywords ? spec.keywords.join(', ') : 'No keywords';
                    const description = spec.description || 'No description';
                    dynamicSpecsHtml += `
                        <div class="video-spec-item">
                            <h4>Video: ${{videoId}}</h4>
                            <p><strong>Generated Keywords:</strong> ${{keywords}}</p>
                            <p><strong>Description:</strong> ${{description}}</p>
                        </div>
                    `;
                }}
                
                task2Content.innerHTML = dynamicSpecsHtml;
            }}
            
            // Update Task 3 content
            const task3Content = document.getElementById('task3-content');
            if (task3Content && task_results.task3) {{
                let staticSpecsHtml = '<h3>Task 3: Static Search Specifications</h3>';
                staticSpecsHtml += '<p>Pre-defined search criteria for consistent analysis across all videos.</p>';
                
                if (task_results.task3.static_specs) {{
                    staticSpecsHtml += '<div class="static-specs-list">';
                    for (const spec of task_results.task3.static_specs) {{
                        const keywords = spec.keywords ? spec.keywords.join(', ') : 'No keywords';
                        staticSpecsHtml += `
                            <div class="static-spec-item">
                                <h4>${{spec.name}}</h4>
                                <p><strong>Keywords:</strong> ${{keywords}}</p>
                                <p><strong>Description:</strong> ${{spec.description || 'Universal search criteria'}}</p>
                            </div>
                        `;
                    }}
                    staticSpecsHtml += '</div>';
                }}
                
                task3Content.innerHTML = staticSpecsHtml;
            }}
            
            // Update Task 4 content
            const task4Content = document.getElementById('task4-content');
            if (task4Content && task_results.task4) {{
                const cleanedRows = task_results.task4.cleaned_rows || 0;
                const status = task_results.task4.status || 'completed';
                
                task4Content.innerHTML = `
                    <h3>Task 4: Data Cleaning Summary</h3>
                    <div class="cleaning-summary">
                        <div class="summary-card">
                            <h4>Data Cleaning Status</h4>
                            <div class="status-badge ${{status}}">${{status.toUpperCase()}}</div>
                        </div>
                        <div class="summary-card">
                            <h4>Cleaned Rows</h4>
                            <div class="summary-number">${{cleanedRows}}</div>
                        </div>
                        <div class="cleaning-details">
                            <h4>Cleaning Actions Performed:</h4>
                            <ul>
                                <li>✅ Removed URLs from content</li>
                                <li>✅ Removed empty content entries</li>
                                <li>✅ Stripped whitespace</li>
                                <li>✅ Handled missing author IDs</li>
                            </ul>
                        </div>
                    </div>
                `;
            }}
            
            // Update Task 5 content
            const task5Content = document.getElementById('task5-content');
            if (task5Content && task_results.task5) {{
                let searchResultsHtml = '<h3>Task 5: Comment Search Results</h3>';
                searchResultsHtml += '<p>Comments that match the generated search specifications.</p>';
                
                for (const [videoId, searchResults] of Object.entries(task_results.task5)) {{
                    const dynamicMatches = searchResults.dynamic ? searchResults.dynamic.length : 0;
                    const staticMatches = searchResults.static ? 
                        Object.values(searchResults.static).reduce((sum, matches) => sum + (matches ? matches.length : 0), 0) : 0;
                    
                    searchResultsHtml += `
                        <div class="search-results-item">
                            <h4>Video: ${{videoId}}</h4>
                            <div class="search-stats">
                                <div class="stat-item">
                                    <span class="stat-label">Dynamic Matches:</span>
                                    <span class="stat-value">${{dynamicMatches}}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Static Matches:</span>
                                    <span class="stat-value">${{staticMatches}}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }}
                
                task5Content.innerHTML = searchResultsHtml;
            }}
            
            // Update Task 6 content
            const task6Content = document.getElementById('task6-content');
            if (task6Content && task_results.task6) {{
                let generalizationsHtml = '<h3>Task 6: Generalizations & Insights</h3>';
                generalizationsHtml += '<p>AI-generated insights including sentiment analysis, topic extraction, and question identification.</p>';
                
                for (const [videoId, task6Data] of Object.entries(task_results.task6)) {{
                    const sentiment = task6Data.sentiment || 0.5;
                    const topics = task6Data.topics || [];
                    const questions = task6Data.questions || [];
                    const totalComments = task6Data.total_comments_analyzed || 0;
                    
                    const sentimentColor = sentiment >= 0.7 ? '#28a745' : sentiment >= 0.4 ? '#ffc107' : '#dc3545';
                    
                    generalizationsHtml += `
                        <div class="generalization-item">
                            <h4>Video: ${{videoId}}</h4>
                            <p><strong>Comments Analyzed:</strong> ${{totalComments}}</p>
                            
                            <div class="generalization-metrics">
                                <div class="metric-card">
                                    <h5>Sentiment Analysis</h5>
                                    <div class="sentiment-score" style="color: ${{sentimentColor}}">
                                        ${{sentiment.toFixed(3)}}
                                    </div>
                                    <p>Overall sentiment (0 = negative, 1 = positive)</p>
                                </div>
                                
                                <div class="metric-card">
                                    <h5>Top 5 Topics</h5>
                                    <ul class="topics-list">
                                        ${{topics.slice(0, 5).map(topic => `<li>${{topic}}</li>`).join('')}}
                                    </ul>
                                </div>
                                
                                <div class="metric-card">
                                    <h5>Top 5 Questions</h5>
                                    <ul class="questions-list">
                                        ${{questions.slice(0, 5).map(q => `<li>${{q.length > 100 ? q.substring(0, 100) + '...' : q}}</li>`).join('')}}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    `;
                }}
                
                task6Content.innerHTML = generalizationsHtml;
            }}
        }}
        
        function showError(message) {{
            document.getElementById('errorMessage').textContent = message;
            document.getElementById('errorMessage').style.display = 'block';
        }}
        
        function initializePage() {{
            // Re-attach event listeners after page update
            const fileInput = document.getElementById('datasetFile');
            if (fileInput) {{
                fileInput.addEventListener('change', function(e) {{
                    currentFile = e.target.files[0];
                    if (currentFile) {{
                        document.getElementById('successMessage').style.display = 'block';
                        document.getElementById('successMessage').textContent = `File selected: ${{currentFile.name}}`;
                    }}
                }});
            }}
        }}
        
        function showTask(taskNumber) {{
            // Hide all task panels
            const panels = document.querySelectorAll('.task-panel');
            panels.forEach(panel => panel.classList.remove('active'));
            
            // Remove active class from all task steps
            const taskSteps = document.querySelectorAll('.task-step');
            taskSteps.forEach(step => step.classList.remove('active'));
            
            // Show selected task panel
            const selectedPanel = document.getElementById(`task${{taskNumber}}-content`);
            if (selectedPanel) {{
                selectedPanel.classList.add('active');
            }}
            
            // Add active class to selected task step
            const selectedTaskStep = document.getElementById(`task${{taskNumber}}-step`);
            if (selectedTaskStep) {{
                selectedTaskStep.classList.add('active');
                selectedTaskStep.scrollIntoView({{ behavior: 'smooth' }});
            }}
        }}
    </script>
</body>
</html>
"""
    
    return html_content

# ==================================
# Enhanced Full Pipeline
# ==================================
def analyze_dataset_enhanced(file_path, use_gpt_search=True, use_all_comments=False, use_sampling=True):
    """
    Enhanced analysis pipeline with configurable options and task-by-task results
    """
    try:
        # Try to read as Excel first, then CSV if that fails
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"Failed to read as Excel: {e}")
            try:
                df = pd.read_csv(file_path, sep="\t")
            except Exception as e2:
                print(f"Failed to read as CSV: {e2}")
                raise ValueError(f"Cannot read file {file_path} as Excel or CSV")
        
        print("=" * 60)
        print("YOUTUBE COMMENT ANALYSIS - TASK BY TASK")
        print("=" * 60)
        
        # Task 1: Clean Data
        print("\n📋 TASK 1: Data Cleaning")
        print("-" * 30)
        df = clean_data(df)
        print(f"✅ Cleaned dataset: {len(df)} rows")
        
        # Task 1: Separate Videos & Comments
        print("\n📋 TASK 1: Separate Videos & Comments")
        print("-" * 30)
        videos, comments = separate_videos_and_comments(df)
        print(f"✅ Found {len(videos)} videos and {len(comments)} comments (including replies)")

        run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        output_dir = f"output/{run_id}"
        os.makedirs(output_dir, exist_ok=True)

        results = {}
        task_results = {
            "task1": {
                "videos_count": len(videos),
                "comments_count": len(comments),
                "videos": videos.to_dict(orient="records"),
                "comments_sample": comments.head(10).to_dict(orient="records")
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
        
        print(f"\n🚀 Starting analysis of {total_videos} videos...")
        print(f"Options: GPT Search={use_gpt_search}, All Comments={use_all_comments}, Sampling={use_sampling}")

        for idx, (_, video) in enumerate(videos.iterrows()):
            video_id = video['id']
            video_comments = comments[comments['parent_id'] == video_id]
            
            print(f"\n📹 Processing video {idx+1}/{total_videos}: {video_id}")
            print(f"   Comments: {len(video_comments)}")

            # Task 2: Dynamic Spec
            print("   🔍 TASK 2: Generating dynamic search spec...")
            dynamic_spec = generate_dynamic_spec(video)
            task_results["task2"][video_id] = dynamic_spec
            print(f"   ✅ Dynamic spec: {dynamic_spec['keywords']}")

            # Task 3: Static Specs
            print("   🔍 TASK 3: Applying static search specs...")
            static_results = {}
            for spec in STATIC_SPECS:
                static_results[spec["name"]] = search_comments(video_comments, spec, use_gpt_search, use_all_comments)
            task_results["task3"][video_id] = static_results
            print(f"   ✅ Static specs applied: {list(static_results.keys())}")

            # Task 5: Search Comments
            print("   🔍 TASK 5: Searching comments with dynamic spec...")
            matched_comments = {
                "dynamic": search_comments(video_comments, dynamic_spec, use_gpt_search, use_all_comments),
                "static": static_results
            }
            task_results["task5"][video_id] = matched_comments
            print(f"   ✅ Found {len(matched_comments['dynamic'])} dynamic matches")

            # Task 6: Generalizations
            print("   🔍 TASK 6: Generating generalizations...")
            comment_list = video_comments.to_dict(orient="records")
            
            # Use appropriate analysis method based on options
            if use_all_comments and not use_sampling:
                sentiment = sentiment_analysis(comment_list, use_all_comments=True)
                topics = extract_topics(comment_list, use_all_comments=True)
                questions = extract_questions(comment_list, use_all_comments=True)
            else:
                sentiment = sentiment_analysis(comment_list, use_all_comments=False)
                topics = extract_topics(comment_list, use_all_comments=False)
                questions = extract_questions(comment_list, use_all_comments=False)

            task_results["task6"][video_id] = {
                "sentiment": sentiment,
                "topics": topics,
                "questions": questions,
                "total_comments_analyzed": len(comment_list)
            }

            results[video_id] = {
                "video_author": video['author_id'] if video['author_id'] and str(video['author_id']).lower() != 'nan' else 'Unknown Author',
                "video_content": video['content'][:200] + "..." if len(video['content']) > 200 else video['content'],
                "video_url": video.get('url', ''),
                "dynamic_spec": dynamic_spec,
                "matched_comments": matched_comments,
                "generalizations": {
                    "avg_sentiment": sentiment,
                    "topics": topics,
                    "questions": questions
                },
                "analysis_options": {
                    "use_gpt_search": use_gpt_search,
                    "use_all_comments": use_all_comments,
                    "use_sampling": use_sampling,
                    "total_comments": len(comment_list)
                }
            }

        # Save detailed results
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save task-by-task results
        with open(f"{output_dir}/task_results.json", "w") as f:
            json.dump(task_results, f, indent=2)

        # Create enhanced web visualization with task flow
        try:
            html_content = create_interactive_visualization(results, task_results)
            html_file_path = os.path.join(output_dir, 'interactive_visualization.html')
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"\n✅ Interactive web visualization created: {html_file_path}")
        except Exception as e:
            print(f"Warning: Could not create interactive visualization: {e}")

        print(f"\n🎉 Enhanced analysis complete. Results saved to {output_dir}/")
        print("📁 Files created:")
        print(f"   - {output_dir}/results.json")
        print(f"   - {output_dir}/task_results.json")
        print(f"   - {output_dir}/interactive_visualization.html")
        
        return results, task_results
        
    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        raise

# ==================================
# Run Example
# ==================================
if __name__ == "__main__":
    # Example usage with enhanced settings
    analyze_dataset_enhanced(
        "dataset.xlsx",
        use_gpt_search=True,      # Use GPT for semantic search
        use_all_comments=True,    # Process ALL comments for better results
        use_sampling=False        # Disable sampling to process all comments
    ) 