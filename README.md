# 📚 Readora — Book Recommender System

A minimal, production-ready **book recommender** built with **Python + Streamlit**.  
Under the hood it uses **item–item KNN with cosine similarity** on a **user×book pivot** to surface fast, relevant suggestions. The app is containerized with **Docker** and ready to run locally or in the cloud.

**Live:** https://readora-v3kv.onrender.com/  
**Docker Image:** https://hub.docker.com/repository/docker/vatsalm1611/readora  
**Contact:** [vatsalm16@gmail.com](mailto:vatsalm16@gmail.com)

---

## ✨ Features

- **Instant recommendations:** type/select a book → get top-N similar titles
- **Item–item KNN (cosine):** scikit-learn `NearestNeighbors` on a sparse pivot
- **Config-driven pipeline:** ingestion → cleaning → transformation (pivot) → model training
- **Streamlit UI:** clean, responsive, distraction-free
- **Images with graceful fallback:** shows covers when available
- **Artifacts first:** pivot/model/book names are serialized & cached at startup
- **Works offline after startup:** no external APIs on the request path
- **Dockerized:** easy to run anywhere; also helped me get hands-on with image build/tag/publish & deployment

---

## 🧠 How it works (ML)

- Build a **user × book interaction pivot** (implicit feedback) as a **CSR sparse matrix**
- **L2-normalize** rows so **cosine similarity ≡ dot product** (robust across popularity)
- Query path: seed title → vector lookup → `kneighbors` → top-N (excluding seed)  
- **Artifacts** (`book_pivot.pkl`, `final_rating.pkl`, `model.pkl`, `book_names.pkl`) are **pickled** and **cached** for fast serving
- Scales simply; for larger catalogs swap the neighbor search to **ANN** (FAISS/Annoy/ScaNN) without changing the interface

---

## 🗂️ Project Structure

```
├── app.py                      # Streamlit app
├── main.py                     # Pipeline runner (optional retraining)
├── requirements.txt            # Pinned deps (Py 3.7 compatible)
├── Dockerfile                  # Container image
├── .streamlit/
│   └── config.toml             # Streamlit server config (WS compression off)
├── books_recommender/          # Core package (components, pipeline, utils, etc.)
├── artifacts/
│   └── serialized_objects/     # *.pkl artifacts (pivot, model, names, etc.)
├── notebook/                   # Data exploration / raw CSVs (if any)
└── README.md
```

---

## 🚀 Quickstart (local, without Docker)

> **Python 3.7** is required for these pinned dependencies.  
> If you use pip, keep tooling compatible (e.g., `pip<24`, `setuptools<70`, `wheel<0.44`).

### 1) Clone
```bash
git clone https://github.com/vatsalm1611/book-recommender-system.git
cd book-recommender-system
```

### 2) Create environment
```bash
# Conda (recommended)
conda create -n books python=3.7.10 -y
conda activate books
```

### 3) Install deps
```bash
pip install -r requirements.txt
```

### 4) (Optional) Retrain the pipeline
```bash
python main.py
```

### 5) Run the app
```bash
streamlit run app.py
# open http://localhost:8501
```

If port 8501 is busy:
```bash
streamlit run app.py --server.port 8888
# open http://localhost:8888
```

---

## 🐳 Docker

### A) Run the prebuilt public image (easiest)
```bash
docker run --rm -e PORT=8501 -p 8888:8501 vatsalm1611/readora:latest
# then open http://localhost:8888
```

### B) Build from source (this repo)
```bash
# from repo root
docker build -t readora:local .
docker run --rm -e PORT=8501 -p 8888:8501 readora:local
```

> The image sets Streamlit server flags compatible with the app and disables WebSocket compression for better proxy compatibility.

---

## ⚙️ Configuration

- Primary runtime settings live in `.streamlit/config.toml`:
  ```toml
  [server]
  headless = true
  address = "0.0.0.0"
  port = 8501
  enableCORS = false
  enableXsrfProtection = false
  enableWebsocketCompression = false

  [browser]
  gatherUsageStats = false
  ```
- Code/config paths (artifacts, etc.) are read from the project’s configuration module and/or `config.yaml` inside the package.

---

## 🧪 Retraining from the UI

- Click **“Train Recommender System”** in the app to run the pipeline end-to-end (ingestion → cleaning → pivot → training).  
- Artifacts are written into `artifacts/serialized_objects/` and subsequently cached by the app.

---

## 🛠️ Troubleshooting

- **Port already in use:** use another host port (`-p 9999:8501`) or `--server.port 9999`.
- **WebSocket errors on some networks:** try a different network / private window (some corporate networks block WSS).
- **Pip fails on Python 3.7:** ensure tooling is compatible:
  ```bash
  python -m pip install --upgrade "pip<24" "setuptools<70" "wheel<0.44"
  ```
- **Large image build times:** add a `.dockerignore` to exclude data/notebooks/venvs; prefer `--prefer-binary` wheels.

---

## 🙌 Credits

- [scikit-learn](https://scikit-learn.org/)  
- [Streamlit](https://streamlit.io/)

---

## 📧 Contact

Questions, feedback, or ideas?  
**Email:** [vatsalm16@gmail.com](mailto:vatsalm16@gmail.com)
