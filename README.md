
# 📚 Book Recommender System

An end-to-end, production-ready book recommender system using collaborative filtering and Streamlit. This project features a modular data pipeline, robust configuration, and a beautiful interactive UI for real-time book recommendations.

---

##  Features

- **Collaborative Filtering** with cosine similarity (scikit-learn NearestNeighbors)
- **Modular Pipeline**: Data ingestion, validation, transformation, and model training
- **Config-Driven**: All paths and settings in `config.yaml`
- **Streamlit UI**: Clean, responsive, and user-friendly web app
- **Book Cover Integration**: Fetches book covers via OpenLibrary API
- **Robust Logging & Exception Handling**
- **Docker Support**: Easy deployment anywhere

---

##  Project Structure

```
├── app.py                  # Streamlit app
├── main.py                 # Pipeline runner
├── config/                 # YAML config
├── books_recommender/      # Core package (components, pipeline, utils, etc.)
├── artifacts/              # Models, processed data, serialized objects
├── notebook/               # Raw CSVs and research
├── requirements.txt        # Python dependencies
├── Dockerfile              # For containerization
└── README.md
```

---

##  Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/vatsalm1611/book-recommender-system.git
cd book-recommender-system
```

### 2. Create and activate environment
```bash
conda create -n books python=3.7.10 -y
conda activate books
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline (optional, only if you want to retrain)
```bash
python main.py
```

### 5. Launch the Streamlit app
```bash
streamlit run app.py
```

---

##  Docker (Optional)

Build and run the app in a container:
```bash
docker build -t book-recommender .
docker run -p 8501:8501 book-recommender
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Credits

- [OpenLibrary Covers API](https://openlibrary.org/dev/docs/api/covers)
- [scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)

---

## 📧 Contact

For any queries or collaboration, reach out at [your-email@example.com](mailto:your-email@example.com)
