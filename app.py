# app.py — Flask backend for Personal Finance Chatbot demo (improved version)

import os, json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# -----------------------------
# Try to load a small HF model
# -----------------------------
HF_MODEL_NAME = os.environ.get("HF_MODEL", "gpt2")  # change if needed
use_hf = True
hf_pipeline = None
try:
    from transformers import pipeline
    hf_pipeline = pipeline("text-generation", model=HF_MODEL_NAME, device=-1)  # CPU mode
    print("HuggingFace pipeline loaded:", HF_MODEL_NAME)
except Exception as e:
    print("HuggingFace pipeline not available:", str(e))
    use_hf = False

# -----------------------------
# Profile contexts for demographic-aware prompts
# -----------------------------
PROFILE_CONTEXTS = {
    "student": "User is a student with limited income. Explain things in simple language with practical, low-cost steps.",
    "professional": "User is a working professional. Provide formal, actionable advice including tax and investment options.",
    "retired": "User is retired or near-retired. Focus on safety, fixed income, healthcare, and low-risk options.",
    "default": "Provide clear and practical personal finance advice."
}

# -----------------------------
# Utility: clean AI output
# -----------------------------
def clean_text(text):
    # Remove repeated consecutive lines
    lines, final = [], []
    for line in text.split("\n"):
        if not final or line.strip() != final[-1].strip():
            final.append(line)
    return "\n".join(final).strip()

# -----------------------------
# Utility: generate text
# -----------------------------
def generate_text(prompt, max_len=150):
    if use_hf and hf_pipeline:
        try:
            out = hf_pipeline(
                prompt,
                max_length=max_len,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=hf_pipeline.tokenizer.eos_token_id,
            )
            generated = out[0]['generated_text']
            result = generated[len(prompt):].strip()

            # Stop at unwanted tokens
            for stop_token in ["User:", "Assistant:", "Instruction:", "Answer:"]:
                if stop_token in result:
                    result = result.split(stop_token)[0].strip()

            return clean_text(result)
        except Exception as e:
            print("HF generation failed:", e)
    return "Sorry — AI backend not configured. Install HuggingFace/Torch or connect IBM/Granite."

# -----------------------------
# Serve frontend
# -----------------------------
@app.route("/", methods=["GET"])
def serve_index():
    if os.path.exists("index.html"):
        return send_file("index.html")
    return "<h2>index.html not found — place your frontend file in the same folder.</h2>", 404

# -----------------------------
# Chat endpoint
# -----------------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    message = data.get("message", "")
    demographic = data.get("demographic", "default")
    complexity = data.get("complexity", "moderate")
    demographic = demographic if demographic in PROFILE_CONTEXTS else "default"

    context = PROFILE_CONTEXTS.get(demographic, "")
    # Use Instruction/Answer style instead of User/Assistant
    prompt = f"{context}\nComplexity: {complexity}\nInstruction: {message}\nAnswer:"
    reply = generate_text(prompt, max_len=200)
    return jsonify({"reply": reply})

# -----------------------------
# Budget summarizer
# -----------------------------
@app.route("/api/budget", methods=["POST"])
def api_budget():
    data = request.get_json(force=True)
    income = float(data.get("incomeMonthly", 0) or 0)
    txns = data.get("transactions", [])
    demographic = data.get("demographic", "default")

    total_spent = sum(float(t.get("amount", 0) or 0) for t in txns)
    savings = income - total_spent

    summary = f"Monthly Income: ₹{income}\nTotal Spent: ₹{total_spent}\nEstimated Savings: ₹{savings}\n"

    prompt = f"{PROFILE_CONTEXTS.get(demographic,'')}\nGiven the following transactions: {json.dumps(txns)}\nIncome: {income}\nInstruction: Write an easy-to-read budget summary and 3 suggestions to improve savings.\nAnswer:"
    ai_paragraph = generate_text(prompt, max_len=180)
    summary += "\nAI Summary:\n" + ai_paragraph
    return jsonify({"summary": summary})

# -----------------------------
# Insights endpoint
# -----------------------------
@app.route("/api/insights", methods=["POST"])
def api_insights():
    data = request.get_json(force=True)
    txns = data.get("transactions", [])
    demographic = data.get("demographic", "default")

    categories = {}
    for t in txns:
        desc = (t.get("desc", "") or "").lower()
        amt = float(t.get("amount", 0) or 0)
        if any(k in desc for k in ["rent", "home"]):
            cat = "Rent"
        elif any(k in desc for k in ["food", "dine", "restaurant", "cafe"]):
            cat = "Dining"
        elif any(k in desc for k in ["grocery", "supermarket"]):
            cat = "Grocery"
        elif any(k in desc for k in ["tax", "income tax"]):
            cat = "Taxes"
        else:
            cat = "Other"
        categories[cat] = categories.get(cat, 0) + amt

    if categories:
        top_cat = max(categories.items(), key=lambda x: x[1])
        insights = f"Top spending category: {top_cat[0]} (₹{top_cat[1]:.2f}). Consider reducing it by 10%."
    else:
        insights = "No transactions supplied — provide some transactions for insights."

    prompt = f"{PROFILE_CONTEXTS.get(demographic,'')}\nTransactions summary: {json.dumps(categories)}\nInstruction: Give 3 actionable spending insights in short sentences.\nAnswer:"
    ai_advice = generate_text(prompt, max_len=140)
    insights += "\n\nAI Suggestions:\n" + ai_advice
    return jsonify({"insights": insights})

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    print("Starting backend on http://127.0.0.1:5000 — serving index.html if present")
    app.run(host="0.0.0.0", port=5000, debug=True)