import streamlit as st
import requests
import json
import pymongo
import hashlib
import datetime
import pandas as pd
from bson.objectid import ObjectId
import time
import re
import os
from dotenv import load_dotenv
import numpy as np
from scipy import stats

# Load environment variables - will look for a .env file
load_dotenv()

# ====================== APP CONFIGURATION ======================
st.set_page_config(page_title="AI-Powered Quiz App", layout="wide")

# Initialize session states
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'current_quiz' not in st.session_state:
    st.session_state.current_quiz = None
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = None
if 'selected_question_id' not in st.session_state:
    st.session_state.selected_question_id = None
if 'show_explanations' not in st.session_state:
    st.session_state.show_explanations = False

# ====================== DATABASE CONNECTION ======================
@st.cache_resource
def get_mongodb_client():
    try:
        # Get MongoDB connection string from environment variables, fall back to localhost if not set
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Verify connection works
        client.server_info()
        return client
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# Try to connect to the database
client = get_mongodb_client()
if client:
    db = client["quiz_app_db"]
    users_collection = db["users"]
    questions_collection = db["questions"]
    quiz_history_collection = db["quiz_history"]
    
    # Create indexes for better performance
    users_collection.create_index("username", unique=True)
    questions_collection.create_index("topic")
else:
    st.error("Failed to connect to database. Please check your connection.")

# ====================== GEMINI API INTEGRATION ======================
# Get API key from environment variable, fall back to the original key if not set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCYDTn-QZC5ZlvX9DGO10fWvk2jz-qyfhI")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def call_gemini_api(prompt):
    """Call the Gemini API with the given prompt and handle errors"""
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    # Try up to 3 times with backoff
    for attempt in range(3):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                data=json.dumps(data),
                timeout=10  # Add timeout to prevent hanging
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit
                time.sleep(2 ** attempt)  # Exponential backoff: 1, 2, 4 seconds
                continue
            else:
                st.error(f"API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error calling Gemini API: {e}")
            if attempt < 2:  # Don't sleep on the last attempt
                time.sleep(2 ** attempt)
    
    return None  # Return None if all attempts failed

def parse_mcq_response(response):
    """Parse the MCQ questions from the Gemini API response with improved error handling"""
    if not response or 'candidates' not in response:
        return []
    
    try:
        text = response['candidates'][0]['content']['parts'][0]['text']
        
        # Initialize list to store parsed questions
        questions = []
        
        # Split the text by question numbers
        question_blocks = re.split(r'\n\s*\d+[\.|\)]\s*', '\n' + text)
        
        if len(question_blocks) > 1:  # Skip the first split which is usually empty
            for block in question_blocks[1:]:
                if not block.strip():
                    continue
                    
                lines = block.strip().split('\n')
                if not lines:
                    continue
                    
                question_text = lines[0].strip()
                options = []
                correct_option = None
                
                # Parse options
                option_pattern = re.compile(r'^([a-d])[\.|\)]?\s+(.+)$', re.IGNORECASE)
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue
                        
                    match = option_pattern.match(line)
                    if match:
                        option_letter = match.group(1).lower()
                        option_text = match.group(2).strip()
                        
                        # Check if this option is marked as correct
                        if '*' in option_text or 'correct' in option_text.lower():
                            correct_option = option_letter
                            option_text = option_text.replace('*', '').strip()
                            
                        options.append({
                            "letter": option_letter,
                            "text": option_text
                        })
                
                if question_text and options and correct_option:
                    questions.append({
                        "question": question_text,
                        "options": options,
                        "correct_option": correct_option
                    })
        
        return questions
    except Exception as e:
        st.error(f"Error parsing MCQ response: {e}")
        return []

# ====================== GEMINI-POWERED ANALYTICS FUNCTIONS ======================

def get_gemini_quiz_analysis(stats_data):
    """
    Generate AI-powered insights about quiz statistics using Gemini
    
    Args:
        stats_data: Dictionary containing quiz statistics
        
    Returns:
        Dictionary with AI-generated insights
    """
    try:
        # Extract key statistics to send to Gemini
        if not stats_data or "topic_stats" not in stats_data or not stats_data["topic_stats"]:
            return {"error": "Insufficient data for analysis"}
        
        # Format topic statistics for analysis
        topic_stats_formatted = []
        for topic in stats_data["topic_stats"]:
            topic_stats_formatted.append({
                "topic": topic["_id"],
                "attempts": topic["attempts"],
                "avg_score": round(topic["avg_score"], 2)
            })
            
        # Format time-based statistics if available
        time_stats_formatted = []
        if "daily_stats" in stats_data and stats_data["daily_stats"]:
            time_stats_formatted = [
                {
                    "date": day["_id"],
                    "count": day["count"],
                    "avg_score": round(day["avg_score"], 2)
                }
                for day in stats_data["daily_stats"][-7:]  # Last 7 days
            ]
        
        # Format difficult topics if available
        difficult_topics = []
        if "difficult_topics" in stats_data and stats_data["difficult_topics"]:
            difficult_topics = [
                {
                    "topic": topic["_id"],
                    "avg_score": round(topic["avg_score"], 2),
                    "attempts": topic["attempts"]
                }
                for topic in stats_data["difficult_topics"][:3]  # Top 3 difficult topics
            ]
        
        # Create a prompt for Gemini to analyze the data
        prompt = f"""
        You are an educational analytics expert. Analyze the following quiz statistics and provide 
        insightful observations, trends, and actionable recommendations. Keep your analysis concise and focused.
        
        OVERALL STATISTICS:
        - Total quiz attempts: {stats_data.get('total_attempts', 'N/A')}
        - Number of unique users: {stats_data.get('unique_users', 'N/A')}
        - Overall average score: {round(stats_data.get('avg_score', 0), 2)}%
        - Number of active topics: {len(topic_stats_formatted)}
        
        TOPIC STATISTICS:
        {json.dumps(topic_stats_formatted, indent=2)}
        
        {f"RECENT ACTIVITY (LAST 7 DAYS):\\n{json.dumps(time_stats_formatted, indent=2)}" if time_stats_formatted else ""}
        
        {f"MOST CHALLENGING TOPICS:\\n{json.dumps(difficult_topics, indent=2)}" if difficult_topics else ""}
        
        Provide your analysis in the following format:
        1. Key Insights (3-4 bullet points about what the data reveals)
        2. Recommendations (3-4 concrete suggestions to improve quiz performance or engagement)
        3. Topics to Focus On (Identify which topics need attention and why)
        
        Be specific, data-driven, and actionable in your analysis.
        """
        
        # Call Gemini API for analysis
        response = call_gemini_api(prompt)
        
        if response and 'candidates' in response:
            analysis_text = response['candidates'][0]['content']['parts'][0]['text']
            
            # Split the analysis into sections
            sections = {}
            current_section = None
            section_content = []
            
            for line in analysis_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("Key Insights") or "key insights" in line.lower():
                    current_section = "key_insights"
                    section_content = []
                elif line.startswith("Recommendations") or "recommendations" in line.lower():
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "recommendations"
                    section_content = []
                elif line.startswith("Topics to Focus On") or "topics to focus" in line.lower():
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "focus_topics"
                    section_content = []
                elif line.startswith("1.") and not current_section:
                    current_section = "key_insights"
                    section_content = [line]
                elif line.startswith("2.") and current_section == "key_insights":
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "recommendations"
                    section_content = [line]
                elif line.startswith("3.") and current_section == "recommendations":
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "focus_topics"
                    section_content = [line]
                elif line.startswith("-") or line.startswith("*") or line[0].isdigit() and line[1] in ['.', ')']:
                    section_content.append(line)
                elif current_section:
                    section_content.append(line)
            
            # Add the last section
            if current_section and section_content:
                sections[current_section] = section_content
            
            return {
                "success": True,
                "insights": sections.get("key_insights", []),
                "recommendations": sections.get("recommendations", []),
                "focus_topics": sections.get("focus_topics", []),
                "raw_analysis": analysis_text
            }
        else:
            return {"success": False, "error": "Failed to generate analysis"}
    except Exception as e:
        return {"success": False, "error": f"Error generating analysis: {str(e)}"}

def get_gemini_user_analysis(user_stats):
    """
    Generate AI-powered personalized insights for a user's quiz performance
    
    Args:
        user_stats: Dictionary containing user's statistics
        
    Returns:
        Dictionary with AI-generated personalized insights
    """
    try:
        if not user_stats or not user_stats.get("has_data", False):
            return {"success": False, "error": "Insufficient data for analysis"}
        
        # Format topic statistics for analysis
        topic_stats_formatted = []
        if "topic_stats" in user_stats and not user_stats["topic_stats"].empty:
            topic_data = user_stats["topic_stats"].to_dict('records')
            topic_stats_formatted = [
                {
                    "topic": topic["Topic"],
                    "avg_score": topic["Average Score (%)"],
                    "attempts": topic["Attempts"],
                    "min_score": topic.get("Min Score (%)", "N/A"),
                    "max_score": topic.get("Max Score (%)", "N/A")
                }
                for topic in topic_data
            ]
        
        # Format performance trend data
        performance_trend = "No trend data available"
        if "recent_trend" in user_stats and user_stats["recent_trend"] is not None:
            trend_value = user_stats["recent_trend"]
            trend_direction = "improving" if trend_value > 0 else "declining"
            performance_trend = f"{trend_direction} by {abs(trend_value):.1f}%"
        
        # Format time and day performance data
        time_performance = []
        if "hour_performance" in user_stats and not user_stats["hour_performance"].empty:
            time_data = user_stats["hour_performance"].to_dict('records')
            time_performance = [
                {"hour": hour["Hour"], "avg_score": hour["Average Score (%)"]}
                for hour in time_data
            ]
        
        day_performance = []
        if "day_performance" in user_stats and not user_stats["day_performance"].empty:
            day_data = user_stats["day_performance"].to_dict('records')
            day_performance = [
                {"day": day["Day"], "avg_score": day["Average Score (%)"]}
                for day in day_data
            ]
        
        # Create a prompt for Gemini to analyze the data
        prompt = f"""
        You are a personalized learning coach. Analyze the following student quiz performance data 
        and provide personalized insights and recommendations. Be empathetic, motivational, 
        and provide actionable advice.
        
        OVERALL STATISTICS:
        - Total quizzes taken: {user_stats.get('total_quizzes', 'N/A')}
        - Total questions answered: {user_stats.get('total_questions', 'N/A')}
        - Correct answers: {user_stats.get('correct_answers', 'N/A')}
        - Overall average score: {round(user_stats.get('avg_score', 0), 2)}%
        - Performance trend: {performance_trend}
        
        TOPIC PERFORMANCE:
        {json.dumps(topic_stats_formatted, indent=2)}
        
        {f"TIME OF DAY PERFORMANCE:\\n{json.dumps(time_performance, indent=2)}" if time_performance else ""}
        
        {f"DAY OF WEEK PERFORMANCE:\\n{json.dumps(day_performance, indent=2)}" if day_performance else ""}
        
        {f"QUIZ SPEED: {user_stats.get('avg_questions_per_minute', 'N/A')} questions per minute" if 'avg_questions_per_minute' in user_stats else ""}
        
        Provide your analysis in the following format:
        1. Performance Summary (an overview of the student's performance)
        2. Strengths (identify 2-3 areas where the student is performing well)
        3. Areas for Improvement (identify 2-3 areas that need attention)
        4. Personalized Study Plan (provide 3-4 specific steps the student should take to improve)
        5. Motivational Note (end with an encouraging note tailored to their performance)
        
        Keep your analysis concise, encouraging, and actionable.
        """
        
        # Call Gemini API for analysis
        response = call_gemini_api(prompt)
        
        if response and 'candidates' in response:
            analysis_text = response['candidates'][0]['content']['parts'][0]['text']
            
            # Split the analysis into sections
            sections = {}
            current_section = None
            section_content = []
            
            for line in analysis_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if "Performance Summary" in line or "performance summary" in line.lower():
                    current_section = "summary"
                    section_content = []
                elif "Strengths" in line or "strengths" in line.lower():
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "strengths"
                    section_content = []
                elif "Areas for Improvement" in line or "areas for improvement" in line.lower() or "improvement" in line.lower():
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "improvements"
                    section_content = []
                elif "Personalized Study Plan" in line or "study plan" in line.lower():
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "study_plan"
                    section_content = []
                elif "Motivational Note" in line or "motivation" in line.lower():
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "motivation"
                    section_content = []
                elif line.startswith("1.") and not current_section:
                    current_section = "summary"
                    section_content = [line]
                elif line.startswith("2.") and current_section == "summary":
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "strengths"
                    section_content = [line]
                elif line.startswith("3.") and current_section == "strengths":
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "improvements"
                    section_content = [line]
                elif line.startswith("4.") and current_section == "improvements":
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "study_plan"
                    section_content = [line]
                elif line.startswith("5.") and current_section == "study_plan":
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "motivation"
                    section_content = [line]
                elif line.startswith("-") or line.startswith("*") or (line[0].isdigit() and line[1] in ['.', ')']):
                    section_content.append(line)
                elif current_section:
                    section_content.append(line)
            
            # Add the last section
            if current_section and section_content:
                sections[current_section] = section_content
            
            return {
                "success": True,
                "summary": sections.get("summary", []),
                "strengths": sections.get("strengths", []),
                "improvements": sections.get("improvements", []),
                "study_plan": sections.get("study_plan", []),
                "motivation": sections.get("motivation", []),
                "raw_analysis": analysis_text
            }
        else:
            return {"success": False, "error": "Failed to generate personalized analysis"}
    except Exception as e:
        return {"success": False, "error": f"Error generating analysis: {str(e)}"}

def get_gemini_topic_recommendations(topic, performance_data):
    """
    Generate topic-specific recommendations and improvement strategies
    
    Args:
        topic: Quiz topic to analyze
        performance_data: Dictionary containing performance data for this topic
        
    Returns:
        Dictionary with AI-generated topic recommendations
    """
    try:
        if not topic or not performance_data:
            return {"success": False, "error": "Insufficient data for analysis"}
        
        # Extract key performance metrics
        avg_score = performance_data.get("avg_score", 0)
        attempts = performance_data.get("attempts", 0)
        
        # Create a prompt for Gemini
        prompt = f"""
        You are an educational expert specializing in the topic of "{topic}". 
        Analyze the following performance data and provide targeted recommendations
        for improving understanding and performance in this subject.

        PERFORMANCE DATA:
        - Topic: {topic}
        - Average Score: {avg_score:.1f}%
        - Number of Attempts: {attempts}
        
        Based on this information, provide:
        1. Topic Overview (Briefly explain why this topic is important in its field)
        2. Common Challenges (What aspects of {topic} typically give students difficulty)
        3. Study Strategies (4-5 specific techniques to better learn this material)
        4. Recommended Resources (Suggest 3-4 types of resources that would help with this topic)
        5. Practice Approach (How to effectively practice and test knowledge in this area)
        
        Keep your recommendations specific to the topic, practical, and based on educational best practices.
        Focus on actionable advice that can directly improve quiz performance.
        """
        
        # Call Gemini API for analysis
        response = call_gemini_api(prompt)
        
        if response and 'candidates' in response:
            analysis_text = response['candidates'][0]['content']['parts'][0]['text']
            
            # Split the analysis into sections
            sections = {}
            current_section = None
            section_content = []
            
            for line in analysis_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if "Topic Overview" in line or "overview" in line.lower():
                    current_section = "overview"
                    section_content = []
                elif "Common Challenges" in line or "challenges" in line.lower():
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "challenges"
                    section_content = []
                elif "Study Strategies" in line or "strategies" in line.lower():
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "strategies"
                    section_content = []
                elif "Recommended Resources" in line or "resources" in line.lower():
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "resources"
                    section_content = []
                elif "Practice Approach" in line or "practice" in line.lower():
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "practice"
                    section_content = []
                elif line.startswith("1.") and not current_section:
                    current_section = "overview"
                    section_content = [line]
                elif line.startswith("2.") and current_section == "overview":
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "challenges"
                    section_content = [line]
                elif line.startswith("3.") and current_section == "challenges":
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "strategies"
                    section_content = [line]
                elif line.startswith("4.") and current_section == "strategies":
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "resources"
                    section_content = [line]
                elif line.startswith("5.") and current_section == "resources":
                    if current_section and section_content:
                        sections[current_section] = section_content
                    current_section = "practice"
                    section_content = [line]
                elif line.startswith("-") or line.startswith("*") or (line[0].isdigit() and line[1] in ['.', ')']):
                    section_content.append(line)
                elif current_section:
                    section_content.append(line)
            
            # Add the last section
            if current_section and section_content:
                sections[current_section] = section_content
            
            return {
                "success": True,
                "overview": sections.get("overview", []),
                "challenges": sections.get("challenges", []),
                "strategies": sections.get("strategies", []),
                "resources": sections.get("resources", []),
                "practice": sections.get("practice", []),
                "raw_analysis": analysis_text
            }
        else:
            return {"success": False, "error": "Failed to generate topic recommendations"}
    except Exception as e:
        return {"success": False, "error": f"Error generating topic recommendations: {str(e)}"}

# ====================== AUTHENTICATION FUNCTIONS ======================
def hash_password(password):
    """Create a SHA-256 hash of the password with salt"""
    salt = "quiz_app_salt"  # In production, use a unique salt per user
    return hashlib.sha256((password + salt).encode()).hexdigest()

def register_user(username, password, role):
    """Register a new user in the database"""
    try:
        # Check if username already exists
        if users_collection.find_one({"username": username}):
            return False, "Username already exists"
        
        # Basic validation
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        # Create new user
        user = {
            "username": username,
            "password": hash_password(password),
            "role": role,
            "created_at": datetime.datetime.now()
        }
        
        result = users_collection.insert_one(user)
        
        if result.inserted_id:
            return True, "Registration successful"
        else:
            return False, "Registration failed"
    except Exception as e:
        return False, f"Error: {str(e)}"

def login_user(username, password):
    """Verify user credentials and log them in"""
    try:
        user = users_collection.find_one({"username": username})
        
        if user and user["password"] == hash_password(password):
            return True, user
        
        return False, None
    except Exception as e:
        st.error(f"Login error: {e}")
        return False, None

def logout():
    """Log out the current user and reset session state"""
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.current_quiz = None
    st.session_state.user_answers = {}
    st.session_state.quiz_score = None
    st.session_state.selected_question_id = None
    st.session_state.show_explanations = False
    st.rerun()

# ====================== ADMIN FUNCTIONS ======================
def save_questions_to_db(topic, questions):
    """Save generated questions to the database"""
    try:
        if not topic or not questions:
            return False, "Topic and questions cannot be empty"
            
        for question in questions:
            question_data = {
                "topic": topic,
                "question": question["question"],
                "options": question["options"],
                "correct_option": question["correct_option"],
                "created_at": datetime.datetime.now(),
                "created_by": st.session_state.user_id
            }
            questions_collection.insert_one(question_data)
        
        return True, f"Successfully saved {len(questions)} questions"
    except Exception as e:
        return False, f"Error saving questions: {str(e)}"

def update_question(question_id, question_text, options, correct_option):
    """Update an existing question in the database"""
    try:
        questions_collection.update_one(
            {"_id": ObjectId(question_id)},
            {"$set": {
                "question": question_text,
                "options": options,
                "correct_option": correct_option,
                "updated_at": datetime.datetime.now()
            }}
        )
        return True, "Question updated successfully"
    except Exception as e:
        return False, f"Error updating question: {str(e)}"

def delete_question(question_id):
    """Delete a question from the database"""
    try:
        result = questions_collection.delete_one({"_id": ObjectId(question_id)})
        if result.deleted_count > 0:
            return True, "Question deleted successfully"
        else:
            return False, "Question not found"
    except Exception as e:
        return False, f"Error deleting question: {str(e)}"

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_topics():
    """Get all unique topics from the database"""
    try:
        return list(questions_collection.distinct("topic"))
    except Exception as e:
        st.error(f"Error fetching topics: {e}")
        return []

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_questions_by_topic(topic):
    """Get all questions for a specific topic"""
    try:
        return list(questions_collection.find({"topic": topic}))
    except Exception as e:
        st.error(f"Error fetching questions: {e}")
        return []

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_quiz_stats_by_topic():
    """Get statistics about quiz attempts by topic"""
    try:
        pipeline = [
            {"$group": {
                "_id": "$topic",
                "attempts": {"$sum": 1},
                "avg_score": {"$avg": "$score_percentage"}
            }}
        ]
        return list(quiz_history_collection.aggregate(pipeline))
    except Exception as e:
        st.error(f"Error getting quiz stats: {e}")
        return []

# ====================== USER FUNCTIONS ======================
def start_quiz(topic, num_questions=5):
    """Initialize a new quiz for the user"""
    try:
        questions = list(questions_collection.find({"topic": topic}).limit(num_questions))
        
        if not questions:
            return False, "No questions available for this topic"
        
        st.session_state.current_quiz = {
            "topic": topic,
            "questions": questions,
            "start_time": datetime.datetime.now()
        }
        
        st.session_state.user_answers = {}
        st.session_state.quiz_score = None
        st.session_state.show_explanations = False
        
        return True, "Quiz started"
    except Exception as e:
        return False, f"Error starting quiz: {str(e)}"

def submit_quiz():
    """Submit the current quiz and calculate score"""
    try:
        if not st.session_state.current_quiz:
            return False, "No active quiz"
        
        score = 0
        total_questions = len(st.session_state.current_quiz["questions"])
        
        for question in st.session_state.current_quiz["questions"]:
            question_id = str(question["_id"])
            if question_id in st.session_state.user_answers and st.session_state.user_answers[question_id] == question["correct_option"]:
                score += 1
        
        score_percentage = (score / total_questions) * 100 if total_questions > 0 else 0
        
        # Save quiz results to history
        quiz_history = {
            "user_id": st.session_state.user_id,
            "username": st.session_state.username,
            "topic": st.session_state.current_quiz["topic"],
            "score": score,
            "total_questions": total_questions,
            "score_percentage": score_percentage,
            "start_time": st.session_state.current_quiz["start_time"],
            "end_time": datetime.datetime.now(),
            "user_answers": st.session_state.user_answers
        }
        
        quiz_history_collection.insert_one(quiz_history)
        
        st.session_state.quiz_score = {
            "score": score,
            "total": total_questions,
            "percentage": score_percentage
        }
        
        return True, f"Quiz submitted. Your score: {score}/{total_questions} ({score_percentage:.1f}%)"
    except Exception as e:
        return False, f"Error submitting quiz: {str(e)}"

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_user_quiz_history():
    """Get quiz history for the current user"""
    try:
        return list(quiz_history_collection.find(
            {"user_id": st.session_state.user_id}
        ).sort("end_time", -1))
    except Exception as e:
        st.error(f"Error fetching quiz history: {e}")
        return []

def get_explanation(question, user_answer, correct_answer):
    """Get AI-generated explanation for quiz answers"""
    try:
        prompt = f"""
        Question: {question}
        User's answer: {user_answer}
        Correct answer: {correct_answer}
        
        Please explain why the correct answer is right and why the user's answer (if different) is incorrect.
        Keep the explanation clear, educational, and under 100 words.
        """
        
        response = call_gemini_api(prompt)
        
        if response and 'candidates' in response:
            return response['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Sorry, explanation could not be generated."
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# ====================== ENHANCED STATISTICS FUNCTIONS ======================

def get_quiz_detailed_stats():
    """
    Get comprehensive statistics about quiz attempts for admin dashboard.
    
    Returns:
        Dict containing various statistics metrics
    """
    try:
        # Total quiz attempts
        total_attempts = quiz_history_collection.count_documents({})
        
        # Total unique users who took quizzes
        unique_users = len(quiz_history_collection.distinct("user_id"))
        
        # Average score across all quizzes
        avg_score_pipeline = [
            {"$group": {
                "_id": None,
                "avg_score": {"$avg": "$score_percentage"}
            }}
        ]
        avg_score_result = list(quiz_history_collection.aggregate(avg_score_pipeline))
        avg_score = avg_score_result[0]["avg_score"] if avg_score_result else 0
        
        # Topic-wise statistics
        topic_stats = get_quiz_stats_by_topic()
        
        # Time-based statistics (quizzes per day for last 30 days)
        thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
        time_pipeline = [
            {"$match": {"end_time": {"$gte": thirty_days_ago}}},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$end_time"}},
                "count": {"$sum": 1},
                "avg_score": {"$avg": "$score_percentage"}
            }},
            {"$sort": {"_id": 1}}
        ]
        daily_stats = list(quiz_history_collection.aggregate(time_pipeline))
        
        # Most popular topics (by number of attempts)
        popular_topics_pipeline = [
            {"$group": {
                "_id": "$topic",
                "attempts": {"$sum": 1}
            }},
            {"$sort": {"attempts": -1}},
            {"$limit": 5}
        ]
        popular_topics = list(quiz_history_collection.aggregate(popular_topics_pipeline))
        
        # Difficulty analysis (topics with lowest avg scores might be hardest)
        difficulty_pipeline = [
            {"$group": {
                "_id": "$topic",
                "avg_score": {"$avg": "$score_percentage"},
                "attempts": {"$sum": 1}
            }},
            {"$match": {"attempts": {"$gte": 3}}},  # Only consider topics with at least 3 attempts
            {"$sort": {"avg_score": 1}},
            {"$limit": 5}
        ]
        difficult_topics = list(quiz_history_collection.aggregate(difficulty_pipeline))
        
        # Time spent analysis
        time_spent_pipeline = [
            {"$project": {
                "topic": 1,
                "duration_seconds": {"$divide": [{"$subtract": ["$end_time", "$start_time"]}, 1000]}
            }},
            {"$group": {
                "_id": "$topic",
                "avg_duration": {"$avg": "$duration_seconds"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"avg_duration": -1}}
        ]
        time_spent = list(quiz_history_collection.aggregate(time_spent_pipeline))
        
        # Recent activity
        recent_quizzes = list(quiz_history_collection.find().sort("end_time", -1).limit(10))
        
        return {
            "total_attempts": total_attempts,
            "unique_users": unique_users,
            "avg_score": avg_score,
            "topic_stats": topic_stats,
            "daily_stats": daily_stats,
            "popular_topics": popular_topics,
            "difficult_topics": difficult_topics,
            "time_spent": time_spent,
            "recent_quizzes": recent_quizzes
        }
    except Exception as e:
        st.error(f"Error getting detailed quiz stats: {e}")
        return {}

def get_user_detailed_stats(user_id):
    """
    Get comprehensive statistics for a specific user.
    
    Args:
        user_id: The ID of the user to get statistics for
        
    Returns:
        Dict containing various user statistics metrics
    """
    try:
        # User's quiz history
        user_history = list(quiz_history_collection.find({"user_id": user_id}).sort("end_time", -1))
        
        if not user_history:
            return {"has_data": False}
        
        # Convert to DataFrame for analysis
        history_df = pd.DataFrame(user_history)
        
        # Basic stats
        total_quizzes = len(history_df)
        total_questions = history_df["total_questions"].sum()
        correct_answers = history_df["score"].sum()
        avg_score = history_df["score_percentage"].mean()
        
        # Calculate improvement over time
        if len(history_df) >= 2:
            history_df = history_df.sort_values(by="end_time")
            first_quiz_score = history_df.iloc[0]["score_percentage"]
            last_quiz_score = history_df.iloc[-1]["score_percentage"]
            recent_trend = last_quiz_score - first_quiz_score
            
            # Calculate rolling average to smooth the trend
            if len(history_df) >= 5:
                history_df["rolling_avg"] = history_df["score_percentage"].rolling(window=3, min_periods=1).mean()
                rolling_trend = history_df["rolling_avg"].iloc[-1] - history_df["rolling_avg"].iloc[2]
            else:
                rolling_trend = None
        else:
            recent_trend = None
            rolling_trend = None
        
        # Topic analysis
        topic_stats = history_df.groupby("topic").agg({
            "score_percentage": ["mean", "min", "max", "std"],
            "_id": "count"
        }).reset_index()
        
        topic_stats.columns = ["Topic", "Average Score (%)", "Min Score (%)", "Max Score (%)", "Score Std Dev", "Attempts"]
        topic_stats = topic_stats.round(2)
        
        # Best and worst topics
        if len(topic_stats) >= 1:
            best_topic = topic_stats.loc[topic_stats["Average Score (%)"].idxmax()]
            worst_topic = topic_stats.loc[topic_stats["Average Score (%)"].idxmin()]
        else:
            best_topic = None
            worst_topic = None
        
        # Time analysis - when is the user most active?
        history_df["hour"] = history_df["start_time"].dt.hour
        history_df["day_of_week"] = history_df["start_time"].dt.dayofweek
        history_df["day_name"] = history_df["start_time"].dt.day_name()
        
        hour_activity = history_df.groupby("hour").size().reset_index(name="count")
        day_activity = history_df.groupby(["day_of_week", "day_name"]).size().reset_index(name="count")
        day_activity = day_activity.sort_values("day_of_week")
        
        # Performance by time of day
        hour_performance = history_df.groupby("hour").agg({
            "score_percentage": "mean"
        }).reset_index()
        
        hour_performance.columns = ["Hour", "Average Score (%)"]
        hour_performance = hour_performance.round(2)
        
        # Performance by day of week
        day_performance = history_df.groupby(["day_of_week", "day_name"]).agg({
            "score_percentage": "mean"
        }).reset_index()
        
        day_performance.columns = ["day_of_week", "Day", "Average Score (%)"]
        day_performance = day_performance.sort_values("day_of_week")
        day_performance = day_performance.round(2)
        
        # Quiz duration analysis
        history_df["duration"] = (history_df["end_time"] - history_df["start_time"]).dt.total_seconds()
        avg_duration = history_df["duration"].mean()
        
        # Questions per minute
        history_df["questions_per_minute"] = history_df["total_questions"] / (history_df["duration"] / 60)
        avg_questions_per_minute = history_df["questions_per_minute"].mean()
        
        return {
            "has_data": True,
            "total_quizzes": total_quizzes,
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "avg_score": avg_score,
            "recent_trend": recent_trend,
            "rolling_trend": rolling_trend,
            "topic_stats": topic_stats,
            "best_topic": best_topic,
            "worst_topic": worst_topic,
            "hour_activity": hour_activity,
            "day_activity": day_activity,
            "hour_performance": hour_performance,
            "day_performance": day_performance,
            "avg_duration": avg_duration,
            "avg_questions_per_minute": avg_questions_per_minute,
            "quizzes": user_history
        }
    except Exception as e:
        st.error(f"Error analyzing user statistics: {e}")
        return {"has_data": False, "error": str(e)}

# ====================== ADMIN QUIZ STATS UI WITH GEMINI ANALYSIS ======================
def render_admin_quiz_stats():
    """Render enhanced admin quiz statistics UI with Gemini-powered analysis"""
    st.header("Quiz Statistics Dashboard")
    
    # Get enhanced statistics
    with st.spinner("Loading statistics..."):
        stats = get_quiz_detailed_stats()
    
    if not stats:
        st.error("Could not retrieve statistics. Please try again later.")
        return
    
    # Top metrics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Quiz Attempts", stats["total_attempts"])
    
    with col2:
        st.metric("Unique Users", stats["unique_users"])
    
    with col3:
        st.metric("Average Score", f"{stats['avg_score']:.1f}%")
    
    # Calculate quiz completion rate
    if "topic_stats" in stats and stats["topic_stats"]:
        total_topics = len(stats["topic_stats"])
        with col4:
            st.metric("Active Topics", total_topics)
    
    # Get Gemini-powered insights if there's enough data
    if stats.get("total_attempts", 0) > 0:
        with st.spinner("Generating AI-powered insights..."):
            ai_analysis = get_gemini_quiz_analysis(stats)
            
        # Display AI insights in an expandable section
        if ai_analysis.get("success", False):
            st.markdown("### 🧠 AI-Powered Insights")
            
            # Create three columns for the different sections
            insight_col, recommend_col, focus_col = st.columns(3)
            
            with insight_col:
                st.markdown("#### Key Insights")
                for insight in ai_analysis.get("insights", []):
                    st.markdown(f"- {insight}")
            
            with recommend_col:
                st.markdown("#### Recommendations")
                for rec in ai_analysis.get("recommendations", []):
                    st.markdown(f"- {rec}")
            
            with focus_col:
                st.markdown("#### Topics to Focus On")
                for focus in ai_analysis.get("focus_topics", []):
                    st.markdown(f"- {focus}")
            
            # Show the full analysis in an expander
            with st.expander("Show detailed AI analysis"):
                st.markdown(ai_analysis.get("raw_analysis", "No detailed analysis available."))
    
    # Create tabs for different analytics views
    stat_tabs = st.tabs(["Topic Analysis", "Time Analysis", "User Insights", "Recent Activity", "Topic Explorer"])
    
    # Topic Analysis Tab
    with stat_tabs[0]:
        st.subheader("Topic Performance")
        
        if "topic_stats" in stats and stats["topic_stats"]:
            # Convert to DataFrame for display
            topic_df = pd.DataFrame(stats["topic_stats"])
            topic_df.columns = ["Topic", "Attempts", "Average Score (%)"]
            topic_df["Average Score (%)"] = topic_df["Average Score (%)"].round(2)
            
            # Allow sorting by different columns
            sort_by = st.radio("Sort by:", ["Attempts", "Average Score (%)", "Topic"], horizontal=True)
            ascending = sort_by == "Topic"  # Ascending for alphabetical, descending for metrics
            
            sorted_df = topic_df.sort_values(sort_by, ascending=ascending)
            st.dataframe(sorted_df, use_container_width=True)
            
            # Visualization
            st.subheader("Average Score by Topic")
            chart_df = sorted_df.sort_values("Average Score (%)", ascending=False)
            st.bar_chart(chart_df.set_index("Topic")["Average Score (%)"])
            
            st.subheader("Topic Popularity")
            st.bar_chart(chart_df.set_index("Topic")["Attempts"])
            
            # Most difficult topics (lowest avg scores)
            if "difficult_topics" in stats and stats["difficult_topics"]:
                st.subheader("Most Challenging Topics")
                difficult_df = pd.DataFrame(stats["difficult_topics"])
                difficult_df.columns = ["Topic", "Average Score (%)", "Attempts"]
                difficult_df["Average Score (%)"] = difficult_df["Average Score (%)"].round(2)
                st.dataframe(difficult_df, use_container_width=True)
        else:
            st.info("No topic data available yet.")
    
    # Time Analysis Tab
    with stat_tabs[1]:
        st.subheader("Quiz Activity Over Time")
        
        if "daily_stats" in stats and stats["daily_stats"]:
            # Convert to DataFrame
            daily_df = pd.DataFrame(stats["daily_stats"])
            daily_df.columns = ["Date", "Quizzes Taken", "Average Score (%)"]
            daily_df["Date"] = pd.to_datetime(daily_df["Date"])
            daily_df["Average Score (%)"] = daily_df["Average Score (%)"].round(2)
            
            # Line chart for quiz activity
            st.line_chart(daily_df.set_index("Date")["Quizzes Taken"])
            
            st.subheader("Average Score Trend")
            st.line_chart(daily_df.set_index("Date")["Average Score (%)"])
            
            # Time of day analysis if available
            if "time_spent" in stats and stats["time_spent"]:
                st.subheader("Average Time Spent by Topic")
                time_df = pd.DataFrame(stats["time_spent"])
                time_df.columns = ["Topic", "Average Duration (sec)", "Count"]
                time_df["Average Duration (min)"] = (time_df["Average Duration (sec)"] / 60).round(2)
                st.dataframe(time_df[["Topic", "Average Duration (min)", "Count"]], use_container_width=True)
        else:
            st.info("No time-based data available yet.")
    
    # User Insights Tab
    with stat_tabs[2]:
        st.subheader("User Performance Insights")
        
        # Top users by number of quizzes
        top_users_pipeline = [
            {"$group": {
                "_id": "$username",
                "user_id": {"$first": "$user_id"},
                "quizzes": {"$sum": 1},
                "avg_score": {"$avg": "$score_percentage"}
            }},
            {"$sort": {"quizzes": -1}},
            {"$limit": 10}
        ]
        
        try:
            top_users = list(quiz_history_collection.aggregate(top_users_pipeline))
            
            if top_users:
                top_users_df = pd.DataFrame(top_users)
                top_users_df.columns = ["Username", "User ID", "Quizzes Taken", "Average Score (%)"]
                top_users_df["Average Score (%)"] = top_users_df["Average Score (%)"].round(2)
                st.dataframe(top_users_df[["Username", "Quizzes Taken", "Average Score (%)"]], use_container_width=True)
                
                # Visualize top users
                st.subheader("Top Users by Quizzes Taken")
                st.bar_chart(top_users_df.set_index("Username")["Quizzes Taken"])
            else:
                st.info("No user data available yet.")
        except Exception as e:
            st.error(f"Error analyzing user data: {e}")
    
    # Recent Activity Tab
    with stat_tabs[3]:
        st.subheader("Recent Quiz Activity")
        
        if "recent_quizzes" in stats and stats["recent_quizzes"]:
            recent_df = pd.DataFrame([
                {
                    "Username": q["username"],
                    "Topic": q["topic"],
                    "Score": f"{q['score']}/{q['total_questions']} ({q['score_percentage']:.1f}%)",
                    "Date": q["end_time"].strftime("%Y-%m-%d %H:%M"),
                    "Duration": f"{(q['end_time'] - q['start_time']).seconds // 60}m {(q['end_time'] - q['start_time']).seconds % 60}s"
                }
                for q in stats["recent_quizzes"]
            ])
            
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No recent quiz activity available.")
    
    # Topic Explorer Tab (New Gemini-powered section)
    with stat_tabs[4]:
        st.subheader("Topic Explorer")
        st.markdown("Select a topic to get AI-powered analysis and recommendations")
        
        if "topic_stats" in stats and stats["topic_stats"]:
            # Convert to selectable format
            topics_data = {topic["_id"]: {"avg_score": topic["avg_score"], "attempts": topic["attempts"]} 
                         for topic in stats["topic_stats"]}
            
            # Topic selector
            selected_topic = st.selectbox("Choose a topic to analyze:", 
                                       list(topics_data.keys()),
                                       format_func=lambda x: f"{x} (Avg: {topics_data[x]['avg_score']:.1f}%, Attempts: {topics_data[x]['attempts']})")
            
            if selected_topic:
                with st.spinner(f"Analyzing topic: {selected_topic}..."):
                    topic_analysis = get_gemini_topic_recommendations(
                        selected_topic, 
                        topics_data[selected_topic]
                    )
                
                if topic_analysis.get("success", False):
                    # Display the topic analysis in a structured way
                    st.markdown(f"### Analysis for: {selected_topic}")
                    
                    # Topic overview
                    st.markdown("#### Topic Overview")
                    for line in topic_analysis.get("overview", []):
                        st.markdown(line)
                    
                    # Common challenges
                    st.markdown("#### Common Challenges")
                    for line in topic_analysis.get("challenges", []):
                        st.markdown(line)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Study strategies
                        st.markdown("#### Study Strategies")
                        for line in topic_analysis.get("strategies", []):
                            st.markdown(line)
                    
                    with col2:
                        # Recommended resources
                        st.markdown("#### Recommended Resources")
                        for line in topic_analysis.get("resources", []):
                            st.markdown(line)
                    
                    # Practice approach
                    st.markdown("#### Practice Approach")
                    for line in topic_analysis.get("practice", []):
                        st.markdown(line)
                else:
                    st.error(f"Failed to generate topic analysis: {topic_analysis.get('error', 'Unknown error')}")
        else:
            st.info("No topic data available yet.")

# ====================== USER PERFORMANCE STATS UI WITH GEMINI ANALYSIS ======================
def render_user_performance_stats():
    """Render enhanced user performance statistics UI with Gemini-powered analysis"""
    st.header("My Quiz Performance")
    
    # Get enhanced user statistics
    with st.spinner("Analyzing your quiz performance..."):
        user_stats = get_user_detailed_stats(st.session_state.user_id)
    
    if not user_stats.get("has_data", False):
        st.info("You haven't taken any quizzes yet. Start a quiz to see your performance statistics!")
        return
    
    # Top metrics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Quizzes Taken", user_stats["total_quizzes"])
    
    with col2:
        st.metric("Questions Answered", user_stats["total_questions"])
    
    with col3:
        st.metric("Correct Answers", user_stats["correct_answers"])
    
    with col4:
        st.metric("Average Score", f"{user_stats['avg_score']:.1f}%")
    
    # Get Gemini-powered personalized insights
    with st.spinner("Generating your personalized learning insights..."):
        ai_analysis = get_gemini_user_analysis(user_stats)
    
    # Display AI insights if available
    if ai_analysis.get("success", False):
        st.markdown("### 🧠 Your Personalized Learning Analysis")
        
        # Summary first
        st.markdown("#### Performance Summary")
        for line in ai_analysis.get("summary", []):
            st.markdown(line)
        
        # Use columns for strengths and improvements
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Your Strengths")
            for strength in ai_analysis.get("strengths", []):
                st.markdown(f"- {strength}")
        
        with col2:
            st.markdown("#### Areas for Improvement")
            for improvement in ai_analysis.get("improvements", []):
                st.markdown(f"- {improvement}")
        
        # Study plan in its own section
        st.markdown("#### Personalized Study Plan")
        for step in ai_analysis.get("study_plan", []):
            st.markdown(f"- {step}")
        
        # Motivational note in a special box
        if ai_analysis.get("motivation", []):
            motivation_text = " ".join(ai_analysis.get("motivation", []))
            st.info(motivation_text)
    
    # Show improvement trend if available
    if user_stats["recent_trend"] is not None:
        trend_text = "improving" if user_stats["recent_trend"] > 0 else "declining"
        trend_value = abs(user_stats["recent_trend"])
        st.info(f"Your performance is {trend_text} by {trend_value:.1f}% since your first quiz.")
    
    # Create tabs for different analytics views
    stat_tabs = st.tabs(["Progress Trends", "Topic Analysis", "Activity Patterns", "Quiz History"])
    
    # Progress Trends Tab
    with stat_tabs[0]:
        st.subheader("Your Progress Over Time")
        
        # Performance over time chart
        quiz_history = user_stats["quizzes"]
        
        if len(quiz_history) > 1:
            # Create a time series chart
            history_df = pd.DataFrame(quiz_history)
            history_df["date"] = history_df["end_time"]
            history_df = history_df.sort_values("date")
            
            chart_data = pd.DataFrame({
                "Date": history_df["date"],
                "Score (%)": history_df["score_percentage"]
            })
            
            st.line_chart(chart_data.set_index("Date")["Score (%)"])
            
            # Add linear trend line calculation
            # Convert dates to numbers for linear regression
            x = np.array(range(len(history_df)))
            y = history_df["score_percentage"].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            if abs(r_value) > 0.3:  # Check if there's a meaningful correlation
                if slope > 0:
                    st.success(f"Your scores show an upward trend! Keep up the good work!")
                else:
                    st.warning(f"Your scores show a downward trend. Try reviewing topics where you scored lower.")
            else:
                st.info("Your performance is relatively stable over time.")
            
            # Duration trend
            st.subheader("Quiz Duration Over Time")
            history_df["duration_minutes"] = history_df.apply(
                lambda row: (row["end_time"] - row["start_time"]).total_seconds() / 60, 
                axis=1
            )
            
            duration_data = pd.DataFrame({
                "Date": history_df["date"],
                "Duration (minutes)": history_df["duration_minutes"].round(2)
            })
            
            st.line_chart(duration_data.set_index("Date")["Duration (minutes)"])
        else:
            st.info("Take more quizzes to see your progress over time!")
    
    # Topic Analysis Tab
    with stat_tabs[1]:
        st.subheader("Performance by Topic")
        
        # Topic performance table
        if "topic_stats" in user_stats and not user_stats["topic_stats"].empty:
            topic_stats = user_stats["topic_stats"]
            
            # Allow sorting by different metrics
            sort_column = st.selectbox(
                "Sort topics by:", 
                ["Average Score (%)", "Attempts", "Min Score (%)", "Max Score (%)", "Topic"],
                index=0
            )
            
            ascending = sort_column == "Topic"  # Alphabetical for topics, descending for metrics
            sorted_topics = topic_stats.sort_values(sort_column, ascending=ascending)
            
            st.dataframe(sorted_topics, use_container_width=True)
            
            # Create a bar chart for topic performance
            st.bar_chart(sorted_topics.set_index("Topic")["Average Score (%)"])
            
            # Show strengths and weaknesses
            if len(topic_stats) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Your Strengths")
                    best_topics = topic_stats.nlargest(3, "Average Score (%)")
                    for _, row in best_topics.iterrows():
                        st.write(f"• {row['Topic']}: {row['Average Score (%)']:.1f}%")
                
                with col2:
                    st.subheader("Areas to Improve")
                    worst_topics = topic_stats.nsmallest(3, "Average Score (%)")
                    for _, row in worst_topics.iterrows():
                        st.write(f"• {row['Topic']}: {row['Average Score (%)']:.1f}%")
                
                # Add a section for topic-specific AI recommendations
                st.subheader("Topic-Specific Recommendations")
                
                # Let the user select a topic for detailed recommendations
                topic_for_analysis = st.selectbox(
                    "Select a topic for detailed recommendations:", 
                    topic_stats["Topic"].tolist(),
                    index=0 if not topic_stats["Topic"].empty else None
                )
                
                if topic_for_analysis:
                    # Get the performance data for this topic
                    topic_row = topic_stats[topic_stats["Topic"] == topic_for_analysis].iloc[0]
                    topic_data = {
                        "avg_score": topic_row["Average Score (%)"],
                        "attempts": topic_row["Attempts"]
                    }
                    
                    with st.spinner(f"Generating recommendations for {topic_for_analysis}..."):
                        topic_recommendations = get_gemini_topic_recommendations(
                            topic_for_analysis, 
                            topic_data
                        )
                    
                    if topic_recommendations.get("success", False):
                        with st.expander(f"Learning strategies for {topic_for_analysis}", expanded=True):
                            # Study strategies
                            st.markdown("#### Study Strategies")
                            for line in topic_recommendations.get("strategies", []):
                                st.markdown(line)
                            
                            # Recommended resources
                            st.markdown("#### Recommended Resources")
                            for line in topic_recommendations.get("resources", []):
                                st.markdown(line)
                            
                            # Practice approach
                            st.markdown("#### Practice Approach")
                            for line in topic_recommendations.get("practice", []):
                                st.markdown(line)
        else:
            st.info("Take quizzes on different topics to see your comparative performance.")
    
    # Activity Patterns Tab
    with stat_tabs[2]:
        st.subheader("Your Quiz Activity Patterns")
        
        if "hour_activity" in user_stats and not user_stats["hour_activity"].empty:
            # Time of day analysis
            hour_activity = user_stats["hour_activity"]
            hour_performance = user_stats["hour_performance"]
            
            # Merge activity and performance data
            hour_merged = pd.merge(hour_activity, hour_performance, left_on="hour", right_on="Hour")
            hour_merged["Hour"] = hour_merged["hour"].apply(lambda h: f"{h:02d}:00")
            
            # Format for display
            st.subheader("Performance by Time of Day")
            st.write("This shows when you take quizzes and how well you perform at different times:")
            
            # Use columns for better layout
            if not hour_merged.empty:
                # Find best time of day for quizzes
                best_hour_idx = hour_merged["Average Score (%)"].idxmax()
                best_hour = hour_merged.iloc[best_hour_idx]
                
                st.write(f"Your best performance is at **{best_hour['Hour']}** with an average score of **{best_hour['Average Score (%)']:.1f}%**")
                
                # Create a visualization
                hour_chart = pd.DataFrame({
                    "Hour": hour_merged["Hour"],
                    "Score (%)": hour_merged["Average Score (%)"],
                    "Quizzes Taken": hour_merged["count"]
                })
                
                st.bar_chart(hour_chart.set_index("Hour")["Score (%)"])
                st.bar_chart(hour_chart.set_index("Hour")["Quizzes Taken"])
            
            # Day of week analysis if available
            if "day_performance" in user_stats and not user_stats["day_performance"].empty:
                day_activity = user_stats["day_activity"]
                day_performance = user_stats["day_performance"]
                
                # Merge activity and performance
                day_merged = pd.merge(
                    day_activity, 
                    day_performance,
                    left_on=["day_of_week", "day_name"],
                    right_on=["day_of_week", "Day"]
                )
                
                st.subheader("Performance by Day of Week")
                
                if not day_merged.empty:
                    # Find best day of week
                    best_day_idx = day_merged["Average Score (%)"].idxmax()
                    best_day = day_merged.iloc[best_day_idx]
                    
                    st.write(f"Your best performance is on **{best_day['Day']}** with an average score of **{best_day['Average Score (%)']:.1f}%**")
                    
                    # Create visualization
                    day_chart = pd.DataFrame({
                        "Day": day_merged["Day"],
                        "Score (%)": day_merged["Average Score (%)"],
                        "Quizzes Taken": day_merged["count"]
                    })
                    
                    # Ensure proper day ordering
                    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    day_chart["Day"] = pd.Categorical(day_chart["Day"], categories=day_order, ordered=True)
                    day_chart = day_chart.sort_values("Day")
                    
                    st.bar_chart(day_chart.set_index("Day")["Score (%)"])
                    st.bar_chart(day_chart.set_index("Day")["Quizzes Taken"])
            
            # Quiz speed analysis
            if "avg_questions_per_minute" in user_stats:
                st.subheader("Quiz Speed Analysis")
                speed = user_stats["avg_questions_per_minute"]
                duration = user_stats["avg_duration"]
                
                st.write(f"On average, you spend **{duration/60:.1f} minutes** per quiz")
                st.write(f"You answer approximately **{speed:.2f} questions per minute**")
                
                # Correlation between speed and score
                history_df = pd.DataFrame(user_stats["quizzes"])
                if not history_df.empty and len(history_df) > 3:
                    history_df["duration"] = (history_df["end_time"] - history_df["start_time"]).dt.total_seconds()
                    history_df["questions_per_minute"] = history_df["total_questions"] / (history_df["duration"] / 60)
                    
                    # Calculate correlation
                    correlation = history_df["questions_per_minute"].corr(history_df["score_percentage"])
                    
                    if abs(correlation) > 0.3:  # Check if correlation is meaningful
                        if correlation > 0:
                            st.success(f"You tend to perform better when answering questions more quickly.")
                        else:
                            st.info(f"You tend to perform better when taking more time with questions.")
                    else:
                        st.write("There's no strong relationship between your quiz speed and performance.")
        else:
            st.info("Take more quizzes to see your activity patterns!")
    
    # Quiz History Tab
    with stat_tabs[3]:
        st.subheader("Quiz History")
        
        quiz_history = user_stats["quizzes"]
        
        if quiz_history:
            # Create a dataframe for better display
            history_df = pd.DataFrame([
                {
                    "Date": q["end_time"].strftime("%Y-%m-%d %H:%M"),
                    "Topic": q["topic"],
                    "Score": f"{q['score']}/{q['total_questions']} ({q['score_percentage']:.1f}%)",
                    "Duration": f"{(q['end_time'] - q['start_time']).seconds // 60}m {(q['end_time'] - q['start_time']).seconds % 60}s",
                    "ID": str(q["_id"])
                }
                for q in quiz_history
            ])
            
            st.dataframe(history_df[["Date", "Topic", "Score", "Duration"]], use_container_width=True)
            
            # Quiz details expander
            with st.expander("View Quiz Details"):
                quiz_id = st.selectbox("Select quiz to view:", 
                                       options=history_df["ID"].tolist(),
                                       format_func=lambda x: f"{history_df[history_df['ID']==x]['Topic'].values[0]} - {history_df[history_df['ID']==x]['Date'].values[0]}")
                
                if quiz_id:
                    # Find selected quiz
                    selected_quiz = next((q for q in quiz_history if str(q["_id"]) == quiz_id), None)
                    
                    if selected_quiz:
                        st.subheader(f"Quiz Details: {selected_quiz['topic']}")
                        st.write(f"Date: {selected_quiz['end_time'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"Duration: {(selected_quiz['end_time'] - selected_quiz['start_time']).seconds // 60} min {(selected_quiz['end_time'] - selected_quiz['start_time']).seconds % 60} sec")
                        st.write(f"Score: {selected_quiz['score']}/{selected_quiz['total_questions']} ({selected_quiz['score_percentage']:.1f}%)")
                        
                        try:
                            # Get the actual questions
                            questions = get_questions_by_topic(selected_quiz['topic'])
                            
                            # Create a mapping of question IDs to questions
                            question_map = {str(q["_id"]): q for q in questions}
                            
                            # Display the questions and user's answers
                            for i, (question_id, user_answer) in enumerate(selected_quiz['user_answers'].items()):
                                if question_id in question_map:
                                    question = question_map[question_id]
                                    
                                    with st.expander(f"Question {i+1}: {question['question']}"):
                                        for opt in question['options']:
                                            if opt['letter'] == question['correct_option'] and opt['letter'] == user_answer:
                                                st.markdown(f"- {opt['letter']}) {opt['text']} ✓ **(Your Answer - Correct)**")
                                            elif opt['letter'] == question['correct_option']:
                                                st.markdown(f"- {opt['letter']}) {opt['text']} ✓ **(Correct Answer)**")
                                            elif opt['letter'] == user_answer:
                                                st.markdown(f"- {opt['letter']}) {opt['text']} ❌ **(Your Answer)**")
                                            else:
                                                st.markdown(f"- {opt['letter']}) {opt['text']}")
                        except Exception as e:
                            st.error(f"Error displaying quiz details: {e}")
        else:
            st.info("You haven't taken any quizzes yet.")

# ====================== MAIN APPLICATION ======================
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .logo-img {
        width: 50px;
        height: 50px;
        margin-right: 10px;
        vertical-align: middle;
    }
    .app-header {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
    }
    .question-number {
        font-weight: bold;
        margin-right: 5px;
    }
    .correct-answer {
        color: green;
        font-weight: bold;
    }
    .user-answer-correct {
        color: green;
        font-weight: bold;
    }
    .user-answer-incorrect {
        color: red;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with logo
    st.markdown("""
    <div class="app-header">
        <img src="https://cdn-icons-png.flaticon.com/512/4616/4616271.png" class="logo-img"/>
        <h1>AI-Powered Quiz Application</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        if st.session_state.logged_in:
            st.write(f"Logged in as: **{st.session_state.username}** ({st.session_state.role})")
            
            if st.button("📤 Logout"):
                logout()
        else:
            st.subheader("Authentication")
            auth_option = st.radio("Select option:", ["Login", "Register"])
            
            if auth_option == "Login":
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    submit_button = st.form_submit_button("Login")
                    
                    if submit_button:
                        if username and password:
                            success, user = login_user(username, password)
                            if success:
                                st.session_state.logged_in = True
                                st.session_state.user_id = str(user["_id"])
                                st.session_state.username = user["username"]
                                st.session_state.role = user["role"]
                                st.rerun()
                            else:
                                st.error("Invalid username or password")
                        else:
                            st.error("Please enter both username and password")
            
            else:  # Register
                with st.form("register_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    role = st.selectbox("Role", ["User", "Admin"])
                    submit_button = st.form_submit_button("Register")
                    
                    if submit_button:
                        if username and password and confirm_password:
                            if password != confirm_password:
                                st.error("Passwords do not match")
                            else:
                                success, message = register_user(username, password, role)
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                        else:
                            st.error("Please fill all the fields")
    
    # Main Content
    if not st.session_state.logged_in:
        st.info("Please login or register to continue")
        
        # App description
        st.markdown("""
        ## Welcome to the AI-Powered Quiz Application!
        
        This application provides an interactive quiz experience, with questions generated and explained by AI.
        
        ### Features:
        - **For Admins**: Create custom quizzes on any topic using AI
        - **For Users**: Take quizzes and get AI-powered explanations
        - **Track your progress**: View your quiz history and performance
        - **Get AI Insights**: Receive personalized recommendations and analysis
        
        Get started by logging in or registering a new account.
        """)
        
    else:
        # Admin View
        if st.session_state.role == "Admin":
            admin_tabs = st.tabs(["Create Quiz", "Manage Questions", "Quiz Stats"])
            
            # Create Quiz Tab
            with admin_tabs[0]:
                st.header("Create New Quiz Questions")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    with st.form("generate_questions_form"):
                        topic = st.text_input("Quiz Topic (e.g., Photosynthesis, World War II)")
                        prompt = st.text_area("Prompt for Gemini", 
                                          value="Generate 5 MCQs on the topic with 4 options each and mark the correct one with an asterisk (*). Format each question as:\n\n1. Question\na) Option 1\nb) Option 2\nc) Option 3*\nd) Option 4")
                        num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=5)
                        
                        submit = st.form_submit_button("Generate Questions")
                        
                        if submit and topic and prompt:
                            with st.spinner("Generating questions with AI..."):
                                # Customize prompt with specific instructions
                                full_prompt = f"Topic: {topic}\n\n{prompt}\n\nCreate {num_questions} multiple choice questions. For each question, CLEARLY mark the correct answer with an asterisk (*) or explicitly state which option is correct."
                                response = call_gemini_api(full_prompt)
                                
                                if response:
                                    questions = parse_mcq_response(response)
                                    if questions:
                                        st.session_state.generated_questions = questions
                                        st.session_state.current_topic = topic
                                        st.success(f"Generated {len(questions)} questions successfully")
                                    else:
                                        st.error("Failed to parse questions from the response. Please try again with a clearer prompt.")
                                else:
                                    st.error("Failed to generate questions. Please try again later.")
                
                with col2:
                    if 'generated_questions' in st.session_state and 'current_topic' in st.session_state:
                        st.subheader(f"Preview: {st.session_state.current_topic} Quiz")
                        
                        # Display generated questions
                        for i, q in enumerate(st.session_state.generated_questions):
                            st.markdown(f"**Q{i+1}: {q['question']}**")
                            
                            for opt in q['options']:
                                if opt['letter'] == q['correct_option']:
                                    st.markdown(f"- {opt['letter']}) {opt['text']} ✓")
                                else:
                                    st.markdown(f"- {opt['letter']}) {opt['text']}")
                            
                            st.markdown("---")
                        
                        if st.button("Save Questions to Database"):
                            success, message = save_questions_to_db(
                                st.session_state.current_topic, 
                                st.session_state.generated_questions
                            )
                            if success:
                                st.success(message)
                                # Clear the session state
                                del st.session_state.generated_questions
                                del st.session_state.current_topic
                                # Clear cache to refresh topics
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.error(message)
            
            # Manage Questions Tab
            with admin_tabs[1]:
                st.header("Manage Existing Questions")
                
                topics = get_topics()
                if topics:
                    selected_topic = st.selectbox("Select Topic", topics)
                    
                    # Add a refresh button
                    if st.button("Refresh Questions"):
                        st.cache_data.clear()
                        st.rerun()
                    
                    questions = get_questions_by_topic(selected_topic)
                    
                    if questions:
                        for i, q in enumerate(questions):
                            st.markdown(f"**Q{i+1}: {q['question']}**")
                            
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                for opt in q['options']:
                                    if opt['letter'] == q['correct_option']:
                                        st.markdown(f"- {opt['letter']}) {opt['text']} ✓")
                                    else:
                                        st.markdown(f"- {opt['letter']}) {opt['text']}")
                            
                            with col2:
                                col_edit, col_delete = st.columns(2)
                                with col_edit:
                                    if st.button("Edit", key=f"edit_{q['_id']}"):
                                        st.session_state.selected_question_id = str(q['_id'])
                                        st.session_state.editing_question = True
                                
                                with col_delete:
                                    if st.button("Delete", key=f"delete_{q['_id']}"):
                                        success, message = delete_question(q['_id'])
                                        if success:
                                            st.success(message)
                                            # Clear cache to refresh questions
                                            st.cache_data.clear()
                                            st.rerun()
                                        else:
                                            st.error(message)
                            
                            st.markdown("---")
                        
                        # Edit question form
                        if 'selected_question_id' in st.session_state and st.session_state.selected_question_id and 'editing_question' in st.session_state and st.session_state.editing_question:
                            st.subheader("Edit Question")
                            
                            # Find the question
                            question_id = st.session_state.selected_question_id
                            question = questions_collection.find_one({"_id": ObjectId(question_id)})
                            
                            if question:
                                with st.form("edit_question_form"):
                                    q_text = st.text_area("Question", value=question['question'])
                                    
                                    options = []
                                    correct_option = question['correct_option']
                                    
                                    for opt in question['options']:
                                        col1, col2, col3 = st.columns([1, 3, 1])
                                        
                                        with col1:
                                            letter = st.text_input("Letter", value=opt['letter'], key=f"letter_{opt['letter']}")
                                        
                                        with col2:
                                            text = st.text_input("Text", value=opt['text'], key=f"text_{opt['letter']}")
                                        
                                        with col3:
                                            is_correct = st.checkbox("Correct", value=(opt['letter'] == correct_option), key=f"correct_{opt['letter']}")
                                            if is_correct:
                                                correct_option = letter
                                        
                                        options.append({"letter": letter, "text": text})
                                    
                                    if st.form_submit_button("Update Question"):
                                        success, message = update_question(question_id, q_text, options, correct_option)
                                        if success:
                                            st.success(message)
                                            st.session_state.editing_question = False
                                            st.session_state.selected_question_id = None
                                            # Clear cache to refresh questions
                                            st.cache_data.clear()
                                            st.rerun()
                                        else:
                                            st.error(message)
                            
                            if st.button("Cancel Editing"):
                                st.session_state.editing_question = False
                                st.session_state.selected_question_id = None
                                st.rerun()
                    else:
                        st.info(f"No questions found for topic: {selected_topic}")
                else:
                    st.info("No quiz topics found. Create questions first.")
            
            # Quiz Stats Tab - Using the new enhanced stats with Gemini analysis
            with admin_tabs[2]:
                render_admin_quiz_stats()
        
        # User View
        else:  # User role
            user_tabs = st.tabs(["Available Quizzes", "Take Quiz", "My Performance"])
            
            # Available Quizzes Tab
            with user_tabs[0]:
                st.header("Available Quiz Topics")
                
                topics = get_topics()
                
                if topics:
                    for topic in topics:
                        questions = get_questions_by_topic(topic)
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{topic}**")
                            st.write(f"Questions available: {len(questions)}")
                        
                        with col2:
                            if st.button("Start Quiz", key=f"start_{topic}"):
                                success, message = start_quiz(topic)
                                if success:
                                    st.session_state.active_tab = "Take Quiz"
                                    st.rerun()
                                else:
                                    st.error(message)
                        
                        st.markdown("---")
                else:
                    st.info("No quizzes available yet. Please check back later.")
            
            # Take Quiz Tab
            with user_tabs[1]:
                if st.session_state.current_quiz:
                    topic = st.session_state.current_quiz["topic"]
                    questions = st.session_state.current_quiz["questions"]
                    
                    st.header(f"Quiz: {topic}")
                    
                    # Display timer
                    start_time = st.session_state.current_quiz["start_time"]
                    elapsed = datetime.datetime.now() - start_time
                    st.write(f"Time elapsed: {elapsed.seconds // 60} min {elapsed.seconds % 60} sec")
                    
                    # Display quiz score if submitted
                    if st.session_state.quiz_score:
                        score = st.session_state.quiz_score
                        st.success(f"Quiz completed! Your score: {score['score']}/{score['total']} ({score['percentage']:.1f}%)")
                        
                        if not st.session_state.show_explanations:
                            if st.button("Show Explanations"):
                                st.session_state.show_explanations = True
                                st.rerun()
                    
                    # Display questions
                    for i, question in enumerate(questions):
                        question_id = str(question["_id"])
                        
                        st.markdown(f"**Question {i+1}: {question['question']}**")
                        
                        # If quiz not submitted, show options for selection
                        if not st.session_state.quiz_score:
                            options = {opt["letter"]: opt["text"] for opt in question["options"]}
                            selected_option = st.radio(
                                "Select your answer:",
                                options.keys(),
                                format_func=lambda x: f"{x}) {options[x]}",
                                key=f"q_{question_id}"
                            )
                            
                            # Save the answer to session state
                            st.session_state.user_answers[question_id] = selected_option
                        
                        # If quiz submitted and explanations requested, show results and explanations
                        elif st.session_state.show_explanations:
                            user_answer = st.session_state.user_answers.get(question_id, None)
                            correct_answer = question["correct_option"]
                            
                            for opt in question["options"]:
                                if opt["letter"] == correct_answer:
                                    st.markdown(f"- {opt['letter']}) {opt['text']} ✓ **(Correct Answer)**")
                                elif opt["letter"] == user_answer:
                                    if user_answer != correct_answer:
                                        st.markdown(f"- {opt['letter']}) {opt['text']} ❌ **(Your Answer)**")
                                    else:
                                        st.markdown(f"- {opt['letter']}) {opt['text']} ✓ **(Your Answer)**")
                                else:
                                    st.markdown(f"- {opt['letter']}) {opt['text']}")
                            
                            # Show explanation
                            with st.expander("Show Explanation"):
                                with st.spinner("Generating explanation..."):
                                    # Map option letters to full text
                                    options_dict = {opt["letter"]: opt["text"] for opt in question["options"]}
                                    user_answer_text = options_dict.get(user_answer, "No answer provided")
                                    correct_answer_text = options_dict.get(correct_answer, "")
                                    
                                    explanation = get_explanation(
                                        question["question"], 
                                        f"{user_answer}) {user_answer_text}", 
                                        f"{correct_answer}) {correct_answer_text}"
                                    )
                                    st.markdown(explanation)
                        
                        st.markdown("---")
                    
                    # Submit button (only show if quiz not submitted)
                    if not st.session_state.quiz_score:
                        if st.button("Submit Quiz"):
                            success, message = submit_quiz()
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
                    
                    # Start new quiz button (only show if quiz submitted)
                    if st.session_state.quiz_score:
                        if st.button("Start New Quiz"):
                            st.session_state.current_quiz = None
                            st.session_state.user_answers = {}
                            st.session_state.quiz_score = None
                            st.session_state.show_explanations = False
                            st.session_state.active_tab = "Available Quizzes"
                            st.rerun()
                else:
                    st.info("No active quiz. Please select a quiz from the Available Quizzes tab.")
            
            # My Performance Tab - Using the new enhanced stats with Gemini analysis
            with user_tabs[2]:
                render_user_performance_stats()

    # Display active tab if specified
    if 'active_tab' in st.session_state and st.session_state.active_tab:
        if st.session_state.active_tab == "Take Quiz":
            user_tabs[1].active = True
        elif st.session_state.active_tab == "Available Quizzes":
            user_tabs[0].active = True
        
        # Clear the active tab
        st.session_state.active_tab = None

# Run the application
if __name__ == "__main__":
    main()