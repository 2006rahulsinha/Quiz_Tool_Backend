import os
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import traceback

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

# --- Initialize FastAPI ---
app = FastAPI()

# --- Configure CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_ORIGIN,  # From environment variable
        "https://quiz-tool-b7z67zp00-rahul-sinhas-projects-5b5f067f.vercel.app",  # Your Vercel preview
        "https://quiz-tool.vercel.app",  # Your production domain (if different)
        "https://*.vercel.app",  # All Vercel preview deployments
        "http://localhost:3000",  # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configure Gemini ---
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in environment variables!")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully")

# --- Pydantic Models ---
class QuizRequest(BaseModel):
    topic: str
    num_questions: int
    difficulty: str

class Answer(BaseModel):
    question_number: int
    user_answer: str

class EvaluateRequest(BaseModel):
    answers: list[Answer]

# --- Global storage for questions ---
current_quiz = {"questions": []}

# --- Helper function to validate question structure ---
def validate_question(q):
    """Ensure question has the required structure"""
    try:
        if not isinstance(q, dict):
            print(f"Invalid question type: {type(q)}")
            return None
        
        # Ensure options exist and have the right structure
        if "options" not in q or not isinstance(q["options"], dict):
            print(f"Invalid options in question: {q.get('question', 'unknown')}")
            return None
        
        # Ensure all required keys exist
        required_keys = ["question", "options", "correct_answer", "explanation"]
        missing_keys = [key for key in required_keys if key not in q]
        if missing_keys:
            print(f"Missing keys {missing_keys} in question: {q.get('question', 'unknown')}")
            return None
        
        # Ensure options has A, B, C, D
        option_keys = ["A", "B", "C", "D"]
        missing_options = [key for key in option_keys if key not in q["options"]]
        if missing_options:
            print(f"Missing options {missing_options} in question: {q['question']}")
            return None
        
        # Ensure correct_answer is valid
        if q["correct_answer"] not in option_keys:
            print(f"Invalid correct_answer '{q['correct_answer']}' in question: {q['question']}")
            return None
        
        return q
    except Exception as e:
        print(f"Error validating question: {e}")
        return None

# --- Endpoints ---
@app.post("/generate_quiz")
async def generate_quiz(req: QuizRequest):
    print(f"\n=== Generating Quiz ===")
    print(f"Topic: {req.topic}")
    print(f"Questions: {req.num_questions}")
    print(f"Difficulty: {req.difficulty}")
    
    prompt = f"""
Generate exactly {req.num_questions} {req.difficulty.lower()} multiple choice questions
about "{req.topic}". 

You MUST return ONLY a valid JSON array with NO additional text, following this structure:
[
  {{
    "question": "Question text here?",
    "options": {{
      "A": "First option",
      "B": "Second option",
      "C": "Third option",
      "D": "Fourth option"
    }},
    "correct_answer": "A",
    "explanation": "Brief explanation of the correct answer"
  }}
]

CRITICAL RULES:
- Return ONLY the JSON array, nothing else
- Each question must have exactly 4 options (A, B, C, D)
- correct_answer must be one of: A, B, C, or D
- All fields are required and must be strings
"""

    try:
        print("Calling Gemini API...")
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt)
        
        print(f"Received response from Gemini")
        print(f"Response text length: {len(response.text)}")
        print(f"Response preview: {response.text[:500]}...")
        
        # Try to extract JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        
        # Try to find JSON array
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        
        if not match:
            print("ERROR: No JSON array found in response")
            print(f"Full response: {response_text}")
            raise ValueError("No JSON array found in response")
        
        json_text = match.group(0)
        print(f"Extracted JSON length: {len(json_text)}")
        
        # Parse JSON
        print("Parsing JSON...")
        questions = json.loads(json_text)
        print(f"Parsed {len(questions)} questions")
        
        # Validate each question
        valid_questions = []
        for i, q in enumerate(questions):
            print(f"Validating question {i+1}...")
            validated = validate_question(q)
            if validated:
                valid_questions.append(validated)
                print(f"  ✓ Question {i+1} valid")
            else:
                print(f"  ✗ Question {i+1} invalid")
        
        if not valid_questions:
            print("ERROR: No valid questions were generated")
            raise ValueError("No valid questions were generated")
        
        print(f"Total valid questions: {len(valid_questions)}")
        
        # Store questions globally for evaluation
        current_quiz["questions"] = valid_questions
        
        return {"questions": valid_questions}

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Failed JSON text: {json_text if 'json_text' in locals() else 'N/A'}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse JSON from AI response: {str(e)}"
        )
    except Exception as e:
        print(f"Unexpected Error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error generating quiz: {str(e)}"
        )

@app.post("/evaluate")
async def evaluate_quiz(req: EvaluateRequest):
    print(f"\n=== Evaluating Quiz ===")
    print(f"Answers received: {len(req.answers)}")
    
    try:
        if not current_quiz["questions"]:
            raise HTTPException(
                status_code=400,
                detail="No quiz found. Please generate a quiz first."
            )
        
        questions = current_quiz["questions"]
        score = 0
        results = []
        
        for answer in req.answers:
            q_idx = answer.question_number - 1
            
            if q_idx < 0 or q_idx >= len(questions):
                print(f"Skipping invalid question number: {answer.question_number}")
                continue
            
            question = questions[q_idx]
            is_correct = answer.user_answer == question["correct_answer"]
            
            if is_correct:
                score += 1
            
            results.append({
                "question_number": answer.question_number,
                "is_correct": is_correct,
                "user_answer": answer.user_answer,
                "correct_answer": question["correct_answer"],
                "explanation": question["explanation"]
            })
        
        print(f"Score: {score}/{len(questions)}")
        
        return {
            "score": score,
            "total": len(questions),
            "results": results
        }
    
    except Exception as e:
        print(f"Evaluation Error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error evaluating quiz: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Quiz API is running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "questions_loaded": len(current_quiz["questions"])
    }