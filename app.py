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
    # In production, replace with your actual MongoDB connection string
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    return client

client = get_mongodb_client()
db = client["quiz_app_db"]
users_collection = db["users"]
questions_collection = db["questions"]
quiz_history_collection = db["quiz_history"]

# ====================== GEMINI API INTEGRATION ======================
GEMINI_API_KEY = "AIzaSyCYDTn-QZC5ZlvX9DGO10fWvk2jz-qyfhI"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def call_gemini_api(prompt):
    """Call the Gemini API with the given prompt"""
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
    
    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        data=json.dumps(data)
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error calling Gemini API: {response.text}")
        return None

def parse_mcq_response(response):
    """Parse the MCQ questions from the Gemini API response"""
    if not response or 'candidates' not in response:
        return []
    
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

# ====================== AUTHENTICATION FUNCTIONS ======================
def hash_password(password):
    """Create a SHA-256 hash of the password"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, role):
    """Register a new user in the database"""
    # Check if username already exists
    if users_collection.find_one({"username": username}):
        return False, "Username already exists"
    
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

def login_user(username, password):
    """Verify user credentials and log them in"""
    user = users_collection.find_one({"username": username})
    
    if user and user["password"] == hash_password(password):
        return True, user
    
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

def update_question(question_id, question_text, options, correct_option):
    """Update an existing question in the database"""
    questions_collection.update_one(
        {"_id": ObjectId(question_id)},
        {"$set": {
            "question": question_text,
            "options": options,
            "correct_option": correct_option,
            "updated_at": datetime.datetime.now()
        }}
    )

def delete_question(question_id):
    """Delete a question from the database"""
    questions_collection.delete_one({"_id": ObjectId(question_id)})

def get_topics():
    """Get all unique topics from the database"""
    return list(questions_collection.distinct("topic"))

def get_questions_by_topic(topic):
    """Get all questions for a specific topic"""
    return list(questions_collection.find({"topic": topic}))

def get_quiz_stats_by_topic():
    """Get statistics about quiz attempts by topic"""
    pipeline = [
        {"$group": {
            "_id": "$topic",
            "attempts": {"$sum": 1},
            "avg_score": {"$avg": "$score_percentage"}
        }}
    ]
    return list(quiz_history_collection.aggregate(pipeline))

# ====================== USER FUNCTIONS ======================
def start_quiz(topic, num_questions=5):
    """Initialize a new quiz for the user"""
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

def submit_quiz():
    """Submit the current quiz and calculate score"""
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

def get_user_quiz_history():
    """Get quiz history for the current user"""
    return list(quiz_history_collection.find({"user_id": st.session_state.user_id}).sort("end_time", -1))

def get_explanation(question, user_answer, correct_answer):
    """Get AI-generated explanation for quiz answers"""
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
                            save_questions_to_db(st.session_state.current_topic, st.session_state.generated_questions)
                            st.success(f"Saved {len(st.session_state.generated_questions)} questions under topic '{st.session_state.current_topic}'")
                            
                            # Clear the session state
                            del st.session_state.generated_questions
                            del st.session_state.current_topic
                            st.rerun()
            
            # Manage Questions Tab
            with admin_tabs[1]:
                st.header("Manage Existing Questions")
                
                topics = get_topics()
                if topics:
                    selected_topic = st.selectbox("Select Topic", topics)
                    
                    questions = get_questions_by_topic(selected_topic)
                    
                    if questions:
                        for i, q in enumerate(questions):
                            st.markdown(f"**Q{i+1}: Which of the following is {q['question']}**")
                            
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
                                        delete_question(q['_id'])
                                        st.success("Question deleted successfully")
                                        st.rerun()
                            
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
                                        update_question(question_id, q_text, options, correct_option)
                                        st.success("Question updated successfully")
                                        st.session_state.editing_question = False
                                        st.session_state.selected_question_id = None
                                        st.rerun()
                            
                            if st.button("Cancel Editing"):
                                st.session_state.editing_question = False
                                st.session_state.selected_question_id = None
                                st.rerun()
                    else:
                        st.info(f"No questions found for topic: {selected_topic}")
                else:
                    st.info("No quiz topics found. Create questions first.")
            
            # Quiz Stats Tab
            with admin_tabs[2]:
                st.header("Quiz Statistics")
                
                stats = get_quiz_stats_by_topic()
                
                if stats:
                    stats_df = pd.DataFrame(stats)
                    stats_df.columns = ["Topic", "Attempts", "Average Score (%)"]
                    stats_df["Average Score (%)"] = stats_df["Average Score (%)"].round(2)
                    
                    st.dataframe(stats_df)
                    
                    # Create a bar chart
                    st.bar_chart(stats_df.set_index("Topic")["Average Score (%)"])
                else:
                    st.info("No quiz statistics available yet")
        
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
                            st.success(message)
                            st.rerun()
                    
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
            
            # My Performance Tab
            with user_tabs[2]:
                st.header("My Quiz History")
                
                quiz_history = get_user_quiz_history()
                
                if quiz_history:
                    for quiz in quiz_history:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Topic: {quiz['topic']}**")
                            st.write(f"Date: {quiz['end_time'].strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"Score: {quiz['score']}/{quiz['total_questions']} ({quiz['score_percentage']:.1f}%)")
                        
                        with col2:
                            if st.button("View Details", key=f"view_{quiz['_id']}"):
                                st.session_state.selected_quiz_history = quiz
                        
                        st.markdown("---")
                    
                    # Performance summary
                    st.subheader("Performance Summary")
                    
                    # Convert to DataFrame for analysis
                    history_df = pd.DataFrame(quiz_history)
                    
                    # Overall statistics
                    avg_score = history_df["score_percentage"].mean()
                    total_quizzes = len(history_df)
                    total_questions = history_df["total_questions"].sum()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Quizzes Taken", total_quizzes)
                    col2.metric("Total Questions Answered", total_questions)
                    col3.metric("Average Score", f"{avg_score:.1f}%")
                    
                    # Topic-wise performance
                    st.subheader("Topic-wise Performance")
                    topic_stats = history_df.groupby("topic").agg({
                        "score_percentage": "mean",
                        "_id": "count"
                    }).reset_index()
                    
                    topic_stats.columns = ["Topic", "Average Score (%)", "Attempts"]
                    topic_stats["Average Score (%)"] = topic_stats["Average Score (%)"].round(2)
                    
                    st.dataframe(topic_stats)
                    
                    # Create a bar chart for topic performance
                    st.bar_chart(topic_stats.set_index("Topic")["Average Score (%)"])
                    
                    # Performance over time
                    st.subheader("Performance Over Time")
                    history_df["date"] = history_df["end_time"].dt.date
                    time_stats = history_df.groupby("date").agg({
                        "score_percentage": "mean"
                    }).reset_index()
                    
                    time_stats.columns = ["Date", "Average Score (%)"]
                    time_stats["Average Score (%)"] = time_stats["Average Score (%)"].round(2)
                    
                    st.line_chart(time_stats.set_index("Date")["Average Score (%)"])
                    
                    # View quiz details if selected
                    if 'selected_quiz_history' in st.session_state and st.session_state.selected_quiz_history:
                        quiz = st.session_state.selected_quiz_history
                        
                        st.subheader(f"Quiz Details: {quiz['topic']}")
                        st.write(f"Date: {quiz['end_time'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"Duration: {(quiz['end_time'] - quiz['start_time']).seconds // 60} min {(quiz['end_time'] - quiz['start_time']).seconds % 60} sec")
                        st.write(f"Score: {quiz['score']}/{quiz['total_questions']} ({quiz['score_percentage']:.1f}%)")
                        
                        # Get the actual questions
                        questions = get_questions_by_topic(quiz['topic'])
                        
                        # Create a mapping of question IDs to questions
                        question_map = {str(q["_id"]): q for q in questions}
                        
                        # Display the questions and user's answers
                        for i, (question_id, user_answer) in enumerate(quiz['user_answers'].items()):
                            if question_id in question_map:
                                question = question_map[question_id]
                                
                                st.markdown(f"**Q{i+1}: {question['question']}**")
                                
                                for opt in question['options']:
                                    if opt['letter'] == question['correct_option'] and opt['letter'] == user_answer:
                                        st.markdown(f"- {opt['letter']}) {opt['text']} ✓ **(Your Answer - Correct)**")
                                    elif opt['letter'] == question['correct_option']:
                                        st.markdown(f"- {opt['letter']}) {opt['text']} ✓ **(Correct Answer)**")
                                    elif opt['letter'] == user_answer:
                                        st.markdown(f"- {opt['letter']}) {opt['text']} ❌ **(Your Answer)**")
                                    else:
                                        st.markdown(f"- {opt['letter']}) {opt['text']}")
                                
                                st.markdown("---")
                        
                        if st.button("Close Details"):
                            st.session_state.selected_quiz_history = None
                            st.rerun()
                else:
                    st.info("You haven't taken any quizzes yet.")

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