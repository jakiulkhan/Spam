import os
import joblib
from flask import Flask, render_template, request, jsonify
from preprocess import clean_text

MODEL_PATH = os.path.join("model", "spam_svm_pipeline.joblib")
app = Flask(__name__)

model = joblib.load(MODEL_PATH)


def predict(text: str):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return {"label": "Ham", "confidence": None, "cleaned": ""}

    pred = int(model.predict([cleaned])[0])
    score = float(model.decision_function([cleaned])[0])  # +ve => spam, -ve => ham
    # Map raw margin to a rough 0-1 confidence via sigmoid
    import math
    conf = 1 / (1 + math.exp(-score))
    return {
        "label": "Spam" if pred == 1 else "Ham",
        "confidence": round(conf, 4),
        "cleaned": cleaned,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = ""
    if request.method == "POST":
        text = request.form.get("message", "")
        if text.strip():
            result = predict(text)
    return render_template("index.html", result=result, text=text)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("message", "")
    if not text.strip():
        return jsonify({"error": "message field required"}), 400
    return jsonify(predict(text))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)