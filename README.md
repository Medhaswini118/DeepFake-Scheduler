# DeepFake Scheduler Project

## 📦 Setup

1. Create virtual environment:
```bash
python -m venv venv
# Windows CMD:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

Place your dataset (CSV with columns `text,label`) in the project folder, e.g. `deepfake_dataset.csv`.

Run:
```bash
python train_model.py
```
This will train a Logistic Regression classifier and save `deepfake_model.pkl`.

---

## 🚀 Run the Scheduler

Start the FastAPI app:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

- Enter text into the input field and submit.  
- Workers will process it with the DeepFake detection model.  
- Results (prediction + confidence) appear in the table.

---

## 🔍 API Examples

Submit task:
```bash
curl -X POST "http://127.0.0.1:8000/submit" -H "Content-Type: application/json" -d "{\"payload\":{\"text\":\"Breaking news: celebrity scandal\"}}"
```

Check jobs:
```bash
curl "http://127.0.0.1:8000/jobs"
```

---

✅ You now have a full pipeline: Dataset → Train Model → Serve with Scheduler → Web Dashboard.
