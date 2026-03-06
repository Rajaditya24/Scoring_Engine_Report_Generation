import json
import os
import pickle
from sentence_transformers import util


# -----------------------------------
# ABSOLUTE PATHS
# -----------------------------------

MODEL_PATH = r"C:\Users\AdityaRaj\OneDrive\Desktop\InternProject\Interview Room\Scoring_Engine_Report_Generation\app\ml_models\nlp_model.pkl"

INPUT_FILE = r"C:\Users\AdityaRaj\OneDrive\Desktop\InternProject\Interview Room\Scoring_Engine_Report_Generation\data\users\input\chat\chat_AA0001.json"

OUTPUT_DIR = r"C:\Users\AdityaRaj\OneDrive\Desktop\InternProject\Interview Room\Scoring_Engine_Report_Generation\data\users\output\text"

# -----------------------------------
# Load Model
# -----------------------------------

print("Loading NLP model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# -----------------------------------
# Semantic Relevance Score
# -----------------------------------

def compute_text_score(question, answer):

    q_emb = model.encode(question, convert_to_tensor=True)
    a_emb = model.encode(answer, convert_to_tensor=True)

    similarity = util.cos_sim(q_emb, a_emb).item()

    # normalize (-1,1) → (0,1)
    score = (similarity + 1) / 2

    return round(float(score), 2)


# -----------------------------------
# Process Interview Chat
# -----------------------------------

def process_chat():

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    scores = []

    for item in data["chat"]:
        q = item["question"]
        a = item["answer"]

        score = compute_text_score(q, a)
        scores.append(score)

    final_score = sum(scores) / len(scores)

    output = {
        "text_score": round(final_score, 2)
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        "text_AA0001.json"
    )

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"✅ Output Generated:\n{output_path}")


# -----------------------------------
# Run
# -----------------------------------

if __name__ == "__main__":
    process_chat()