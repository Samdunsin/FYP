import os
import json
import sqlite3
import logging
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any




# =========================
#  OpenRouter API Key Loader
# =========================
try:
    import streamlit as st
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    APP_REFERER = st.secrets.get("APP_REFERER") or os.getenv("APP_REFERER", "http://localhost")
    APP_TITLE = st.secrets.get("APP_TITLE") or os.getenv("APP_TITLE", "NutritionApp")
except Exception:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    APP_REFERER = os.getenv("APP_REFERER", "http://localhost")
    APP_TITLE = os.getenv("APP_TITLE", "NutritionApp")

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "❌ OpenRouter API key not found. Please set OPENROUTER_API_KEY in .streamlit/secrets.toml "
        "or as an environment variable."
    )



# =========================
# ===== DB UTILITIES  =====
# =========================


def _db_path() -> str:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.abspath(os.path.join(backend_dir, '..', 'database'))
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, 'nutrition_recommendation.db')

def _connect():
    return sqlite3.connect(_db_path())

# ==========================================
# ===== DB SETUP, SEED & CORE QUERIES  =====
# ==========================================

def setup_enhanced_database() -> None:
    conn = _connect()
    cur = conn.cursor()

   
    

    cur.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    

# Call setup_user_table() inside setup_enhanced_database()


    # Food items
    cur.execute("""
        CREATE TABLE IF NOT EXISTS FoodItems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_name TEXT NOT NULL UNIQUE,
            local_name TEXT,
            calories REAL,
            fat REAL,
            protein REAL,
            carbohydrates REAL,
            sodium REAL,
            fiber REAL,
            cholesterol REAL,
            category TEXT
        )
    """)
  


    # Guidelines
    cur.execute("""
        CREATE TABLE IF NOT EXISTS DiseaseDietGuidelines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease_name TEXT,
            dietary_restriction TEXT,
            nutrient_limit REAL,
            description TEXT
        )
    """)

    # User profiles
    cur.execute("""
        CREATE TABLE IF NOT EXISTS UserProfiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            weight REAL,
            gender TEXT,
            activity_level TEXT,
            restrictions TEXT,
            cardiac_condition TEXT,
            medications TEXT
        )
    """)

    # Seed foods
    cur.execute("SELECT COUNT(*) FROM FoodItems")
    if cur.fetchone()[0] == 0:
        seed_foods = [
            ('Eba', 'Cassava paste', 210, 0.3, 2.0, 50.0, 5.0, 2.0, 0.0, 'Carbs'),
            ('Moi Moi', 'Bean pudding', 320, 10.0, 12.0, 20.0, 30.0, 4.0, 0.0, 'Protein'),
            ('Jollof Rice', 'Tomato rice', 380, 7.0, 6.0, 65.0, 420.0, 3.5, 0.0, 'Carbs'),
            ('Okra Soup', 'Okro Soup', 150, 5.0, 4.0, 6.0, 120.0, 3.0, 0.0, 'Vegetable'),
            ('Boiled Plantain', 'Dodo alata (boiled)', 180, 0.2, 1.6, 47.0, 3.0, 2.3, 0.0, 'Carbs'),
            ('Grilled Tilapia', 'Tilapia', 190, 4.0, 32.0, 0.0, 75.0, 0.0, 70.0, 'Protein'),
        ]
        cur.executemany("""
            INSERT INTO FoodItems
            (food_name, local_name, calories, fat, protein, carbohydrates, sodium, fiber, cholesterol, category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seed_foods)

    # Seed cardiac guidelines
    cur.execute("SELECT COUNT(*) FROM DiseaseDietGuidelines WHERE disease_name = 'Heart Disease'")
    if cur.fetchone()[0] == 0:
        seed_guidelines = [
            ('Heart Disease', 'Sodium (mg/day)', 2300, 'Keep total sodium under 2300 mg/day; ideal target closer to 1500 mg/day if possible.'),
            ('Heart Disease', 'Saturated fat (% of calories)', 6, 'Prefer <6% of daily calories from saturated fat; use oils like canola/olive.'),
            ('Heart Disease', 'Fiber (g/day)', 25, 'Aim 25–30 g fiber/day via beans, vegetables, whole grains.'),
            ('Heart Disease', 'Added sugar (g/day)', 50, 'Reduce added sugars; watch sweetened drinks and pastries.'),
            ('Heart Disease', 'Portion control', 0, 'Use measured portions; avoid deep-frying and heavy salts/spices.'),
        ]
        cur.executemany("""
            INSERT INTO DiseaseDietGuidelines
            (disease_name, dietary_restriction, nutrient_limit, description)
            VALUES (?, ?, ?, ?)
        """, seed_guidelines)

    conn.commit()
    conn.close()

    
import hashlib

def _hash_password(password: str) -> str:
    """Simple password hashing using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(name: str, email: str, password: str) -> bool:
    conn = _connect()
    cur = conn.cursor()
    try:
        # Reset Users table during development (delete and recreate table)
        cur.execute("DROP TABLE IF EXISTS Users")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
        """)
        cur.execute("""
            INSERT INTO Users (name, email, password_hash)
            VALUES (?, ?, ?)
        """, (name, email, _hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(email: str, password: str):
    """Check if email/password match. Returns user dict or None."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, email FROM Users WHERE email = ? AND password_hash = ?
    """, (email, _hash_password(password)))
    row = cur.fetchone()
    conn.close()
    if row:
        keys = ["id", "name", "email"]
        return dict(zip(keys, row))
    return None


# ==========DATABASE SECTION==============================

def save_enhanced_user_profile(profile: Dict) -> None:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO UserProfiles
        (name, age, weight, gender, activity_level, restrictions, cardiac_condition, medications)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        profile.get("name"),
        int(profile.get("age", 0)) if profile.get("age") is not None else None,
        float(profile.get("weight", 0)) if profile.get("weight") is not None else None,
        profile.get("gender"),
        profile.get("activity_level"),
        profile.get("restrictions"),
        profile.get("cardiac_condition"),
        profile.get("medications"),
    ))
    conn.commit()
    conn.close()

# Retrieve User Profile (new)
def get_user_profile(user_id: int) -> Optional[Dict]:
    """
    Fetch a saved user profile by ID.
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, age, weight, gender, activity_level, restrictions, cardiac_condition, medications
        FROM UserProfiles
        WHERE id = ?
    """, (user_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    keys = ["id", "name", "age", "weight", "gender", "activity_level",
            "restrictions", "cardiac_condition", "medications"]
    return dict(zip(keys, row))

# Update User Profile (new)
def update_user_profile(user_id: int, updated_profile: Dict) -> bool:
    """
    Update an existing user profile with new values.
    Returns True if update was successful, False otherwise.
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        UPDATE UserProfiles
        SET name = ?, age = ?, weight = ?, gender = ?, activity_level = ?, 
            restrictions = ?, cardiac_condition = ?, medications = ?
        WHERE id = ?
    """, (
        updated_profile.get("name"),
        updated_profile.get("age"),
        updated_profile.get("weight"),
        updated_profile.get("gender"),
        updated_profile.get("activity_level"),
        updated_profile.get("restrictions"),
        updated_profile.get("cardiac_condition"),
        updated_profile.get("medications"),
        user_id
    ))
    conn.commit()
    success = cur.rowcount > 0
    conn.close()
    return success

# ===============================================================


def get_complete_food_info(food_name: str) -> Optional[Dict]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT food_name, local_name, calories, fat, protein, carbohydrates,
               sodium, fiber, cholesterol, category
        FROM FoodItems WHERE food_name = ?
    """, (food_name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    keys = ['food_name','local_name','calories','fat','protein','carbohydrates',
            'sodium','fiber','cholesterol','category']
    return dict(zip(keys, row))

def generate_diet_recommendation(disease_name: str) -> List[tuple]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT dietary_restriction, description
        FROM DiseaseDietGuidelines
        WHERE disease_name = ?
        ORDER BY id ASC
    """, (disease_name,))
    rows = cur.fetchall()
    conn.close()
    return rows

    # ======================================
# =====  OpenRouter HTTP Helpers   =====
# ======================================

def _openrouter_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_REFERER,
        "X-Title": APP_TITLE,
    }

def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 90) -> Tuple[Optional[Dict], Optional[str], int]:
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        status = resp.status_code
        if 200 <= status < 300:
            try:
                return resp.json(), None, status
            except Exception as je:
                return None, f"JSON parse error: {je}", status
        else:
            return None, f"HTTP {status}: {resp.text}", status
    except Exception as e:
        logging.exception("POST request failed")
        return None, f"Request error: {e}", -1


# ==========================================
# ===== OPENROUTER-ONLY LLM PIPELINE   =====
# ==========================================

@dataclass
class LLMResponse:
    content: str
    tokens_used: int
    model: str
    confidence_score: float = 0.0


# Multi LLM Engine
# =========================
import os
from typing import Optional

class MultiLLMNutritionEngine:
    """
    Calls OpenRouter API models to analyze food items,
    generate recommendations, and verify safety.
    """

    def __init__(self, openrouter_api_key: Optional[str] = None):
        # Use the provided key OR fallback to environment variable
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError(
                "No OpenRouter API key provided. "
                "Pass it when creating the engine OR set OPENROUTER_API_KEY as an environment variable."
            )

        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.total_tokens = 0

        # All models are served via OpenRouter
        self.models = {
            "analysis_engine": "anthropic/claude-3.5-sonnet",
            "recommendation_generator": "openai/gpt-4o-mini",
            "safety_verifier": "anthropic/claude-3-haiku"
        }


    def _call_openrouter_chat(self, prompt: str, model: str,
                              temperature: float = 0.2,
                              max_tokens: int = 800) -> LLMResponse:
        headers = _openrouter_headers()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        data, err, status = _post_json(self.openrouter_url, headers, payload, timeout=120)
        if err:
            return LLMResponse(content=f"HTTP Error {status}: {err}", tokens_used=0, model=model)

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = json.dumps(data)

        approx_tokens = int(len(prompt.split()) * 1.2 + len(content.split()) * 1.2)
        self.total_tokens += approx_tokens
        return LLMResponse(content=content.strip(), tokens_used=approx_tokens, model=model)


    # ---------- Prompts ----------
    def analyze_nutritional_data(
        self,
        food_item: str,
        user_profile: Dict,
        db_food_info: Optional[Dict],
        db_recommendations: List
    ) -> LLMResponse:
        """
        Analyze nutritional data for a given food item.
        Works with both database foods and foods not in the database.
        """

        if db_food_info:
            db_nutrition = f"""
DATABASE NUTRITION FACTS for {food_item}:
- Local Name: {db_food_info.get('local_name', 'N/A')}
- Calories: {db_food_info.get('calories', 'N/A')} per 100g
- Fat: {db_food_info.get('fat', 'N/A')} g
- Protein: {db_food_info.get('protein', 'N/A')} g
- Carbohydrates: {db_food_info.get('carbohydrates', 'N/A')} g
- Sodium: {db_food_info.get('sodium', 'N/A')} mg
- Fiber: {db_food_info.get('fiber', 'N/A')} g
- Cholesterol: {db_food_info.get('cholesterol', 'N/A')} mg
- Category: {db_food_info.get('category', 'N/A')}
"""
            nutrition_source_note = "database"
        else:
            db_nutrition = "⚠️ No database entry found. Estimate values from nutritional science knowledge."
            nutrition_source_note = "estimate"

        # Expanded + flexible prompt
        prompt = f"""
### Instruction
You are a Nutritional Data Analyzer for Cardiac Health. 
- If database facts are provided, rely on them strictly.  
- If no database facts exist, generate reasonable estimates using nutritional science.  
- Always tailor to HEART DISEASE considerations.  
- Use Nigerian/local food context when possible.  
- Explicitly state the SOURCE of data.

FOOD ITEM: {food_item}

{db_nutrition}

USER PROFILE JSON:
{json.dumps(user_profile, indent=2)}

DIETARY GUIDELINES (tuples of restriction, description):
{db_recommendations}

### Requirements
- Give a numeric heart-health score (/10).  
- Provide portion recommendations based on age, weight, gender, and activity.  
- Highlight benefits AND concerns.  
- Indicate compatibility (high/medium/low) with reasoning.  
- Estimate sodium contribution vs. 2300 mg/day.  
- Include a `"source"` field in JSON output to mark "database" or "estimate".

### Response Format
Return JSON ONLY, exactly like this:
{{
  "nutritional_analysis": {{
    "calories_per_100g": {db_food_info.get('calories', 0) if db_food_info else 0},
    "sodium_mg": {db_food_info.get('sodium', 0) if db_food_info else 0},
    "fat_g": {db_food_info.get('fat', 0) if db_food_info else 0},
    "fiber_g": {db_food_info.get('fiber', 0) if db_food_info else 0},
    "cholesterol_mg": {db_food_info.get('cholesterol', 0) if db_food_info else 0}
  }},
  "heart_health_score": "X/10",
  "portion_recommendation": "specific grams for this user",
  "key_benefits": ["benefit1", "benefit2"],
  "concerns": ["concern1", "concern2"],
  "user_compatibility": "high/medium/low with explanation",
  "daily_sodium_impact": "percentage of 2300mg daily limit",
  "source": "{nutrition_source_note}"
}}
### Response
"""
        return self._call_openrouter_chat(prompt, self.models["analysis_engine"],
                                          temperature=0.1, max_tokens=700)


    def generate_recommendations(self, analysis_data: str,
                                 user_profile: Dict, food_item: str) -> LLMResponse:
        prompt = f"""
### Instruction
You are a Nigerian Heart-Healthy Meal Planner.

CONTEXT (JSON analysis):
{analysis_data}

USER PROFILE JSON:
{json.dumps(user_profile, indent=2)}

FOCUS FOOD: {food_item}

Create:
1) A Nigerian meal plan (breakfast, lunch, dinner) featuring this food
2) Cooking methods that preserve heart benefits
3) Complementary foods to balance nutrients
4) Practical prep tips common in Nigerian households
5) Local ingredients & substitutes

Requirements:
- Keep sodium under ~2300 mg/day total
- Be culturally authentic
- Include measurements, times, portions
- Consider activity level & restrictions
- Plain text output

### Response
"""
        return self._call_openrouter_chat(prompt, self.models["recommendation_generator"],
                             temperature=0.7, max_tokens=800)

    def verify_safety(self, recommendations: str, analysis_data: str,
                      user_profile: Dict) -> LLMResponse:
        prompt = f"""
### Instruction
You are a Medical Safety Verifier for Cardiac Nutrition.

Task: Review these recommendations for a heart disease patient and produce a strict JSON verdict.

RECOMMENDATIONS:
{recommendations}

NUTRITIONAL ANALYSIS (JSON):
{analysis_data}

USER PROFILE JSON:
{json.dumps(user_profile, indent=2)}

Checklist:
1) Sodium vs 2300 mg/day guideline
2) Contraindicated foods for cardiac patients
3) Calories suitable for profile
4) Drug-nutrient interactions (if any meds given)
5) Portion sanity
6) Avoid extreme/restrictive advice

Output a compact JSON ONLY (no markdown fences):
{{
  "safety_rating": "HIGH_RISK" | "MEDIUM_RISK" | "LOW_RISK" | "APPROVED",
  "issues_found": ["issue1", "issue2"],
  "modifications_needed": ["modification1", "modification2"],
  "medical_disclaimers": ["disclaimer1", "disclaimer2"],
  "final_approved_recommendations": "modified recommendations if needed",
  "confidence_score": "0-100%"
}}
### Response
"""
        return self._call_openrouter_chat(prompt, self.models["safety_verifier"],
                                          temperature=0.1, max_tokens=600)


        # ---------- Orchestration ----------
    def process_complete_recommendation(
        self,
        food_item: Optional[str],
        user_profile: Dict,
        db_food_info: Optional[Dict],
        db_recommendations: List,
        user_prompt: Optional[str] = None   # <--- supports both
    ) -> Dict:
        """
        Orchestrates analysis → recommendations → safety.
        Supports both specific food items and general user prompts.
        """

        # --- Case 1: General free-form prompt ---
        if not food_item and user_prompt:
            # Step 1: Run a true LLM-based nutritional analysis
            analysis = self._call_openrouter_chat(
                f"""You are a Nigerian Heart-Healthy Nutrition Analyst.

USER PROMPT: {user_prompt}

USER PROFILE JSON:
{json.dumps(user_profile, indent=2)}

Task:
1. Provide a structured nutritional analysis tailored to the profile.
2. Highlight potential benefits and risks for heart health.
3. Suggest portion considerations if relevant.
4. Keep explanations clear and culturally appropriate.
""",
                self.models["analysis_engine"],
                temperature=0.5,
                max_tokens=600,
            )

            # Step 2: Generate recommendations
            recs = self._call_openrouter_chat(
                f"""You are a Nigerian Heart-Healthy Nutrition Advisor.

INPUT ANALYSIS:
{analysis.content}

USER PROFILE JSON:
{json.dumps(user_profile, indent=2)}

Based on the above, provide specific, culturally relevant dietary recommendations with
measurements, precautions, and practical examples.""",
                self.models["recommendation_generator"],
                temperature=0.7,
                max_tokens=800,
            )

            # Step 3: Safety Verification
            safety = self.verify_safety(recs.content, analysis.content, user_profile)

            return {
                "success": True,
                "mode": "general_prompt",
                "user_prompt": user_prompt,
                "analysis": analysis.content,   # <--- now a real LLM-driven analysis
                "recommendations": recs.content,
                "safety_check": safety.content,
                "models_used": [
                    self.models["analysis_engine"],
                    self.models["recommendation_generator"],
                    self.models["safety_verifier"]
                ],
                "total_tokens": self.total_tokens,
            }

        # --- Case 2: Food-specific pipeline ---
        analysis = self.analyze_nutritional_data(
            food_item, user_profile, db_food_info, db_recommendations
        )
        if any(err in analysis.content for err in ("Error", "HTTP Error", "Model error")):
            return {"success": False, "error": "Failed at analysis stage", "details": analysis.content}

        recs = self.generate_recommendations(analysis.content, user_profile, food_item)
        if any(err in recs.content for err in ("Error", "HTTP Error", "Model error")):
            return {"success": False, "error": "Failed at recommendation stage", "details": recs.content}

        safety = self.verify_safety(recs.content, analysis.content, user_profile)
        if any(err in safety.content for err in ("Error", "HTTP Error", "Model error")):
            return {"success": False, "error": "Failed at safety verification", "details": safety.content}

        return {
            "success": True,
            "mode": "food_item",
            "food_item": food_item,
            "database_info": db_food_info,
            "analysis": analysis.content,
            "recommendations": recs.content,
            "safety_check": safety.content,
            "total_tokens": self.total_tokens,
            "models_used": list(self.models.values()),
        }


# Public entry for frontend:
def get_multi_llm_recommendation(
    food_item: Optional[str],
    user_profile: Dict,
    db_food_info: Optional[Dict],
    db_recommendations: List,
    user_prompt: Optional[str] = None  # New keyword parameter added here
) -> Dict:
    """
    Entry point for frontend.
    Decides whether to run food-specific pipeline or free-text pipeline.
    """
    engine = MultiLLMNutritionEngine(OPENROUTER_API_KEY)
    return engine.process_complete_recommendation(
        food_item=food_item,
        user_profile=user_profile,
        db_food_info=db_food_info,
        db_recommendations=db_recommendations,
        user_prompt=user_prompt  # Now passed through properly
    )


