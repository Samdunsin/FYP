import sys
import os
import streamlit as st
import json


if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None


# ----------------------------
# Load CSS
# ----------------------------
def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Skipping custom styles.")

load_css(os.path.join(os.path.dirname(__file__), "style.css"))

# ----------------------------
# Import backend functions
# ----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))

from final_enhanced_backend import (
    setup_enhanced_database,
    generate_diet_recommendation,
    save_enhanced_user_profile,
    get_complete_food_info,
    get_multi_llm_recommendation,
    create_user, authenticate_user   # <--- add this
)


# ----------------------------
# Setup database
# ----------------------------
setup_enhanced_database()

# ----------------------------
# API Key Handling
# ----------------------------
try:
    api_key = st.secrets["HF_API_KEY"]
except KeyError:
    api_key = os.getenv("HF_API_KEY", "")
    if not api_key:
        st.error("❌ Hugging Face API key not found in secrets or environment variables.")
        st.stop()

# ----------------------------
# Session State
# ----------------------------
for key, default in {
    "chat_history": [],
    "recommendation": "",
    "token_usage": 0,
    "user_profile": {},
    "processing_status": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------
# Utility Functions
# ----------------------------
def clear_conversation():
    st.session_state.chat_history = []
    st.session_state.recommendation = ""
    st.session_state.token_usage = 0
    st.session_state.processing_status = ""
    st.rerun()
def display_analysis_results(result):
    if not result.get("success"):
        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        if "details" in result:
            st.error(f"Details: {result['details']}")
        return

    mode = result.get("mode", "unknown")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Analysis", "🍽️ Recommendations", "🛡️ Safety Check", "⚙️ Technical Info"]
    )

    # Nutritional Analysis (only if available)
        # Nutritional Analysis
    with tab1:
        st.subheader("Nutritional Analysis")
        try:
            analysis_text = result['analysis']

            # Case 1: General free-text mode
            if '"source": "free-text"' in analysis_text:
                st.info("ℹ️ No detailed nutritional breakdown available for this request.")
                st.write("This is a **general nutrition guidance prompt**. "
                         "Recommendations are tailored based on your profile but not tied to a specific food item.")

            # Case 2: Structured JSON analysis
            elif analysis_text.strip().startswith('{'):
                analysis_json = json.loads(analysis_text)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Heart Health Score", f"{analysis_json.get('heart_health_score', 'N/A')}")
                    st.metric("User Compatibility", analysis_json.get('user_compatibility', 'N/A'))
                with col2:
                    st.write("**Nutritional Breakdown:**")
                    nutrition = analysis_json.get('nutritional_analysis', {})
                    for key, value in nutrition.items():
                        st.write(f"• {key.replace('_', ' ').title()}: {value}")
                st.write("**Key Benefits:**")
                for benefit in analysis_json.get('key_benefits', []):
                    st.write(f"✅ {benefit}")
                st.write("**Concerns:**")
                for concern in analysis_json.get('concerns', []):
                    st.write(f"⚠️ {concern}")

            # Case 3: Fallback (unexpected format)
            else:
                st.write(analysis_text)

        except Exception:
            st.write(result.get('analysis', '❌ No analysis available'))

    # Recommendations
       # Recommendations
    with tab2:
        st.subheader("Meal Recommendations")

        if result.get("mode") == "general_prompt":
            st.markdown("💬 **Chatbot-style Nutrition Guidance**")
            st.markdown(result['recommendations'])
        else:
            st.write(result['recommendations'])


    # Safety Check
    with tab3:
        st.subheader("Safety Verification")
        if "safety_check" in result:
            try:
                safety_text = result['safety_check']
                if safety_text.strip().startswith('{'):
                    safety_json = json.loads(safety_text)
                    rating = safety_json.get('safety_rating', 'UNKNOWN')
                    if rating == 'APPROVED':
                        st.success(f"✅ Safety Rating: {rating}")
                    elif rating == 'LOW_RISK':
                        st.info(f"ℹ️ Safety Rating: {rating}")
                    elif rating == 'MEDIUM_RISK':
                        st.warning(f"⚠️ Safety Rating: {rating}")
                    else:
                        st.error(f"🚨 Safety Rating: {rating}")
                    st.metric("Confidence Score", f"{safety_json.get('confidence_score', 'N/A')}")
                    if safety_json.get('issues_found'):
                        st.write("**Issues Found:**")
                        for issue in safety_json['issues_found']:
                            st.write(f"• {issue}")
                    if safety_json.get('modifications_needed'):
                        st.write("**Modifications Needed:**")
                        for mod in safety_json['modifications_needed']:
                            st.write(f"• {mod}")
                    if safety_json.get('medical_disclaimers'):
                        st.write("**Medical Disclaimers:**")
                        for disclaimer in safety_json['medical_disclaimers']:
                            st.info(f"⚕️ {disclaimer}")
                    if safety_json.get('final_approved_recommendations'):
                        st.write("**Final Approved Recommendations:**")
                        st.write(safety_json['final_approved_recommendations'])
                else:
                    st.write(safety_text)
            except Exception:
                st.write(result['safety_check'])
        else:
            st.info("No safety verification for this request.")

    # Technical Info
    with tab4:
        st.subheader("Technical Information")
        st.write(f"**Mode:** {mode}")
        st.write(f"**Total Tokens Used:** {result.get('total_tokens', 0)}")
        st.write("**Models Used:**")
        for i, model in enumerate(result.get('models_used', []), 1):
            st.write(f"{i}. {model}")


# ----------------------------
# App Title & Intro
# ----------------------------
st.title("🍲 Heart Nutrition Recommender (Multi-LLM System)")
st.subheader("Advanced AI-powered cardiac nutrition recommendations 🫀")

with st.sidebar:
    if st.button("🚪 Sign Out"):
        st.session_state.is_authenticated = False
        st.session_state.user_info = None
        st.rerun()


# st.markdown("""
# This app uses **3 specialized AI models** to provide safe, accurate, and culturally-appropriate Nigerian meal recommendations for heart health:
# - 🔍 **Model A**: Analyzes nutritional data
# - 🍽️ **Model B**: Generates meal recommendations  
# - 🛡️ **Model C**: Verifies safety and medical accuracy
# """)

if not st.session_state.is_authenticated:
    st.title("🔐 Welcome to Heart Nutrition Recommender")
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        st.subheader("Login to Continue")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            user = authenticate_user(email, password)
            if user:
                st.session_state.is_authenticated = True
                st.session_state.user_info = user
                st.success(f"✅ Welcome back, {user['name']}!")
                st.rerun()
            else:
                st.error("❌ Invalid email or password. Please try again.")

    with tab_signup:
        st.subheader("Create a New Account")
        name = st.text_input("Name", key="signup_name")
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            if name and signup_email and signup_password:
                success = create_user(name, signup_email, signup_password)
                if success:
                    st.success("✅ Account created successfully! Please log in.")
                else:
                    st.error("❌ This email is already registered.")
            else:
                st.warning("⚠️ Please fill all fields.")

    st.stop()  # 🚨 Stops the rest of the app until login/signup succeeds

    st.write(f"👋 Hello, {st.session_state.user_info['name']}!")


# Sidebar UI # ------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
with st.sidebar:
    # User Info
    if st.session_state.user_info:
        st.markdown(f"### 👤 {st.session_state.user_info['name']}")
        st.caption(st.session_state.user_info['email'])

    # ----------------------
    # Account Management
    # ----------------------
    st.subheader("Account")
    if st.button("✏️ Edit Profile", key="edit_profile_btn"):
        st.session_state.show_edit_profile = True
    if st.button("🔑 Change Password", key="change_password_btn"):
        st.session_state.show_change_password = True
    if st.button("🗑️ Delete Account", key="delete_account_btn"):
        st.session_state.show_delete_account = True

    # ----------------------
    # Theme Switch
    # ----------------------
    st.subheader("Theme")
    dark_mode = st.toggle("🌙 Dark Mode", value=True, key="theme_toggle")
    if dark_mode:
        st.markdown("<style>body{background-color:#0f1116;color:#e1e1e1;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>body{background-color:white;color:black;}</style>", unsafe_allow_html=True)

    # ----------------------
    # App Settings
    # ----------------------
    st.subheader("Settings")
    show_tips = st.toggle("💡 Show Tips", value=True, key="show_tips_toggle")
    advanced_mode = st.toggle("⚡ Advanced Mode", value=False, key="advanced_mode_toggle")

    # ----------------------
    # Stats
    # ----------------------
    st.subheader("Stats")
    st.metric("Total Queries", len(st.session_state.chat_history))
    st.metric("Total Tokens", st.session_state.token_usage)

    # ----------------------
    # Sign Out
    # ----------------------
    if st.button("🚪 Sign Out", key="sidebar_signout_btn"):
        st.session_state.is_authenticated = False
        st.session_state.user_info = None
        st.rerun()



# ----------------------------
# User Profile Form
# ----------------------------
with st.expander("👤 Enter Your Profile & Preferences"):
   
    st.session_state.user_profile["age"] = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.user_profile.get("age", 30))
    st.session_state.user_profile["weight"] = st.number_input("Weight (kg)", min_value=20, max_value=300, value=st.session_state.user_profile.get("weight", 70))
    st.session_state.user_profile["gender"] = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.user_profile.get("gender", "Male")))
    st.session_state.user_profile["activity_level"] = st.selectbox("Activity Level", ["Low", "Moderate", "High"], index=["Low", "Moderate", "High"].index(st.session_state.user_profile.get("activity_level", "Moderate")))
    st.session_state.user_profile["restrictions"] = st.text_area("Dietary Restrictions (if any)", value=st.session_state.user_profile.get("restrictions", ""))
    st.session_state.user_profile["cardiac_condition"] = st.selectbox("Specific Heart Condition (if known)",
        ["General Heart Disease", "High Blood Pressure", "Heart Failure", "Post-Heart Attack", "Coronary Artery Disease", "Other"],
        index=0)
    st.session_state.user_profile["medications"] = st.text_area("Current Medications (optional - for interaction checking)",
        value=st.session_state.user_profile.get("medications", ""))

# ----------------------------
# ----------------------------
# ----------------------------
# Unified Input (Food or Prompt)
# ----------------------------
user_input = st.text_area(
    "🍲 Enter a food name OR ask any nutrition question:",
    placeholder="e.g. 'Jollof Rice' OR 'Give me a full Nigerian dinner plan for a heart patient'"
)

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Get Multi-LLM Recommendation"):
        if not user_input.strip():
            st.error("Please enter a food or a question!")
            st.stop()
        if not all([
            
            st.session_state.user_profile.get("age"),
            st.session_state.user_profile.get("weight")
        ]):
            st.error("Please fill in at least Age, and Weight in your profile!")
            st.stop()

        save_enhanced_user_profile(st.session_state.user_profile)
        st.success("✅ Profile saved to database!")

        with st.spinner("Processing with Multi-LLM system..."):
            try:
                # Try to match as food first
                db_food_info = get_complete_food_info(user_input)
                db_recs = generate_diet_recommendation("Heart Disease")

                if db_food_info:
                    # Food-specific pipeline
                    result = get_multi_llm_recommendation(
                        food_item=user_input,
                        user_profile=st.session_state.user_profile,
                        db_food_info=db_food_info,
                        db_recommendations=db_recs
                    )
                else:
                    # Free-text general pipeline
                    result = get_multi_llm_recommendation(
                        food_item=None,
                        user_profile=st.session_state.user_profile,
                        db_food_info=None,
                        db_recommendations=db_recs,
                        user_prompt=user_input
                    )

                st.session_state.token_usage += result.get('total_tokens', 0)
                st.session_state.chat_history.append({
                    "user": f"Query: {user_input}",
                    "bot": result
                })
                st.success("Multi-LLM analysis complete!")
            except Exception as e:
                st.exception(e)


with col2:
    if st.button("🧹 Clear Conversation"):
        clear_conversation()



# ----------------------------
# Display Results
# ----------------------------
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📊 Latest Analysis Results:")
    latest_result = st.session_state.chat_history[-1]["bot"]
    display_analysis_results(latest_result)

# ----------------------------
# Chat History
# ----------------------------
if len(st.session_state.chat_history) > 1:
    st.markdown("---")
    st.markdown("### 📝 Previous Recommendations:")
    for i, entry in enumerate(reversed(st.session_state.chat_history[:-1])):
        with st.expander(f"Analysis #{len(st.session_state.chat_history) - i - 1}: {entry['user']}"):
            display_analysis_results(entry['bot'])

# ----------------------------
# Stats
# ----------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Analyses", len(st.session_state.chat_history))
with col2:
    st.metric("Total Tokens Used", st.session_state.token_usage)
with col3:
    if st.session_state.chat_history:
        last_tokens = st.session_state.chat_history[-1]["bot"].get("total_tokens", 0)
        st.metric("Last Analysis Tokens", last_tokens)



import streamlit as st
from final_enhanced_backend import MultiLLMNutritionEngine  # import your backend

# ----------------------------
# Page Config (Dark Mode)
# ----------------------------
st.set_page_config(
    page_title="Heart Nutrition Assistant",
    page_icon="❤️",
    layout="wide"
)

# Custom Dark UI
st.markdown("""
    <style>
        body {
            background-color: #0f1116;
            color: #e1e1e1;
        }
        .stTextInput textarea, .stTextArea textarea {
            background-color: #1c1f26 !important;
            color: #e1e1e1 !important;
            border: 1px solid #333 !important;
            border-radius: 10px !important;
        }
        .stButton button {
            background-color: #3b82f6;
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }
        .user-msg {
            background-color: #1f2937;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 8px 0;
            text-align: right;
        }
        .bot-msg {
            background-color: #374151;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 8px 0;
            text-align: left;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Session State for Chat
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "step" not in st.session_state:
    st.session_state.step = 0
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# ----------------------------
