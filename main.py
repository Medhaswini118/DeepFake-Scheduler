# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import threading, queue, time, uuid, random
import joblib

app = FastAPI(title="Mini Distributed Task Scheduler with DeepFake Detection")

# ---------- Configurable parameters ----------
NUM_WORKERS = 3
RETRY_TIMEOUT = 10
# --------------------------------------------

# Load DeepFake model (trained with train_model.py)
try:
    deepfake_model = joblib.load("deepfake_model.pkl")
    print("✅ DeepFake model loaded successfully")
except Exception as e:
    deepfake_model = None
    print(f"⚠️ Could not load DeepFake model: {e}")

def run_deepfake_inference(text: str):
    if deepfake_model is None:
        return {"error": "Model not loaded"}
    try:
        vectorizer = deepfake_model["vectorizer"]
        clf = deepfake_model["model"]
        features = vectorizer.transform([text])
        prediction = clf.predict(features)[0]
        proba = float(clf.predict_proba(features)[0].max())
        return {"prediction": str(prediction), "confidence": proba}
    except Exception as e:
        return {"error": str(e)}

class SubmitRequest(BaseModel):
    payload: dict = {}

class Scheduler:
    def __init__(self, num_workers=NUM_WORKERS, retry_timeout=RETRY_TIMEOUT):
        self.q = queue.Queue()
        self.jobs = {}
        self.lock = threading.Lock()
        self.num_workers = num_workers
        self.retry_timeout = retry_timeout
        self.workers = []
        self.monitor_thread = None
        self._stop = threading.Event()

    def start(self):
        for wid in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, args=(wid,), daemon=True)
            t.start()
            self.workers.append(t)
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Scheduler started with {self.num_workers} workers")

    def stop(self):
        self._stop.set()

    def submit(self, payload=None):
        task_id = str(uuid.uuid4())[:8]
        job = {
            "id": task_id,
            "status": "queued",
            "payload": payload or {},
            "worker_id": None,
            "start_time": None,
            "end_time": None,
            "attempts": 0,
            "result": None,
        }
        with self.lock:
            self.jobs[task_id] = job
        self.q.put(task_id)
        return task_id

    def _worker_loop(self, worker_id):
        while not self._stop.is_set():
            try:
                task_id = self.q.get(timeout=1)
            except queue.Empty:
                continue

            with self.lock:
                job = self.jobs.get(task_id)
                if not job or job["status"] != "queued":
                    self.q.task_done()
                    continue
                job["status"] = "processing"
                job["worker_id"] = worker_id
                job["start_time"] = time.time()
                job["attempts"] += 1

            try:
                if "text" in job["payload"]:
                    text = job["payload"]["text"]
                    result = run_deepfake_inference(text)
                else:
                    result = {"message": "No text provided"}

                with self.lock:
                    job["status"] = "done"
                    job["result"] = result
                    job["end_time"] = time.time()

            except Exception as e:
                print(f"[worker {worker_id}] crashed on task {task_id}: {e}")

            finally:
                try:
                    self.q.task_done()
                except Exception:
                    pass

    def _monitor_loop(self):
        while not self._stop.is_set():
            time.sleep(1)
            now = time.time()
            requeued = []
            with self.lock:
                for tid, job in list(self.jobs.items()):
                    if job["status"] == "processing" and job["start_time"]:
                        if now - job["start_time"] > self.retry_timeout:
                            job["status"] = "queued"
                            job["worker_id"] = None
                            job["start_time"] = None
                            requeued.append(tid)
            for tid in requeued:
                print(f"[monitor] re-queuing stuck task {tid}")
                self.q.put(tid)

    def get_job(self, task_id):
        with self.lock:
            return self.jobs.get(task_id)

    def list_jobs(self):
        with self.lock:
            return list(self.jobs.values())

    def stats(self):
        with self.lock:
            total = len(self.jobs)
            queued = sum(1 for j in self.jobs.values() if j["status"] == "queued")
            processing = sum(1 for j in self.jobs.values() if j["status"] == "processing")
            done = sum(1 for j in self.jobs.values() if j["status"] == "done")
        return {"total": total, "queued": queued, "processing": processing, "done": done, "workers": self.num_workers}

scheduler = Scheduler()
scheduler.start()

@app.post("/submit")
async def submit_task(req: SubmitRequest):
    tid = scheduler.submit(payload=req.payload)
    return {"task_id": tid}

@app.get("/status/{task_id}")
async def task_status(task_id: str):
    job = scheduler.get_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="task not found")
    return job

@app.get("/jobs")
async def all_jobs():
    return scheduler.list_jobs()

@app.get("/stats")
async def get_stats():
    return scheduler.stats()

DASHBOARD = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>DeepFake Scheduler Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%; margin-top: 10px; }
    th, td { border: 1px solid #ddd; padding: 6px; text-align: left; }
    th { background: #f2f2f2; }
  </style>
</head>
<body>
  <h2>Mini Scheduler with DeepFake Detection</h2>
  <label>Text: <input id="text" type="text" size="40" value="Breaking news: celebrity scandal"/></label>
  <button onclick="submitTask()">Submit Task</button>
  <button onclick="refresh()">Refresh</button>
  <div id="stats" style="margin-top:10px"></div>

  <table id="jobs-table">
    <thead><tr><th>ID</th><th>Status</th><th>Attempts</th><th>Worker</th><th>Result</th></tr></thead>
    <tbody></tbody>
  </table>

<script>
async function refresh(){
  const jobs = await (await fetch('/jobs')).json();
  const stats = await (await fetch('/stats')).json();
  document.getElementById('stats').innerText = `Total: ${stats.total} | Queued: ${stats.queued} | Processing: ${stats.processing} | Done: ${stats.done}`;
  const tbody = document.querySelector('#jobs-table tbody');
  tbody.innerHTML = '';
  jobs.sort((a,b)=> (a.id>b.id?1:-1));
  for(const j of jobs){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${j.id}</td><td>${j.status}</td><td>${j.attempts}</td><td>${j.worker_id ?? ''}</td><td>${j.result ? JSON.stringify(j.result) : ''}</td>`;
    tbody.appendChild(tr);
  }
}

async function submitTask(){
  const text = document.getElementById('text').value || 'Test tweet';
  const r = await fetch('/submit', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({payload:{text}}) });
  const j = await r.json();
  console.log('submitted', j);
  refresh();
}

setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=DASHBOARD, status_code=200)
