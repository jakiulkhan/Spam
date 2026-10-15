# SMS Spam Classifier

Classifies SMS messages as **Spam** or **Ham** using TF-IDF + Linear SVM.

## Dataset
[SMS Spam Collection (UCI/Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) → place as `data/spam.csv`.

## Results
| Metric | Score |
|---|---|
| Accuracy | ~98.5% |
| Precision | ~98.3% |
| Recall | ~89.3% |
| F1 | ~93.6% |

## Setup
```bash
git clone https://github.com/<you>/sms-spam-classifier.git
cd sms-spam-classifier
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train
```bash
python train.py
```
Produces `model/spam_svm_pipeline.joblib`.

## Run Web App
```bash
python app.py
```
Open http://localhost:5000

## API
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You won a free iPhone"}'
```