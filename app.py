from pathlib import Path
import sys
import pickle
import numpy as np
import streamlit as st

from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException

# Page config (1.10.0 compatible)
st.set_page_config(page_title="Readora", page_icon="📚", layout="wide")

# ---------- Cache shim: st.cache_resource if available, else st.cache ----------
try:
    _cache_resource = st.cache_resource  # Streamlit >=1.18
    def cache_resource(**kwargs):
        return _cache_resource(**kwargs)
except AttributeError:
    def cache_resource(**kwargs):        # Streamlit 1.10.0 fallback
        kwargs.setdefault("allow_output_mutation", True)
        return st.cache(**kwargs)

def _sanitize_url(url):
    """Return a safe http(s) url or None."""
    if url is None:
        return None
    s = str(url).strip()
    if s in {"", "0", "None", "nan", "NaN"}:
        return None
    if s.startswith("//"):
        s = "https:" + s
    if s.startswith("http://"):
        # try https first; if that host doesn't support https you can revert
        s = "https://" + s[len("http://"):]
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return None

def show_img(url: str):
    """Render image with backward compatible arg name."""
    safe = _sanitize_url(url)
    if not safe:
        st.caption("No image available")
        return
    try:
        st.image(safe, use_container_width=True)
    except TypeError:
        st.image(safe, use_column_width=True)
    except Exception:
        st.caption("No image available")

@cache_resource(show_spinner=False)
def _load_pickled(path: Path):
    """Load a pickle once (cached)."""
    with path.open("rb") as f:
        return pickle.load(f)

class Recommendation:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.cfg = app_config.get_recommendation_config()

            self.book_pivot_path   = Path(self.cfg.book_pivot_serialized_objects).resolve()
            self.final_rating_path = Path(self.cfg.final_rating_serialized_objects).resolve()
            self.model_path        = Path(self.cfg.trained_model_path).resolve()

            # Heavy objects (cached by _load_pickled)
            self.book_pivot   = _load_pickled(self.book_pivot_path)
            self.final_rating = _load_pickled(self.final_rating_path)
            self.model        = _load_pickled(self.model_path)
        except Exception as e:
            raise AppException(e, sys) from e

    def fetch_posters(self, neighbor_ids):
        try:
            names = [self.book_pivot.index[i] for i in neighbor_ids]
            idxs = []
            for title in names:
                match = np.where(self.final_rating["title"] == title)[0]
                idxs.append(int(match[0]) if len(match) else None)

            posters = []
            for ix in idxs:
                posters.append(None if ix is None else self.final_rating.iloc[ix].get("image_url", None))
            return posters
        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name: str, n_to_show: int):
        """Return exactly n_to_show neighbors (excluding the query itself)."""
        try:
            where = np.where(self.book_pivot.index == book_name)[0]
            if len(where) == 0:
                raise AppException(f"Book not found in pivot: {book_name}", sys)

            book_id = int(where[0])

            # request one extra neighbor to account for self
            max_possible = self.book_pivot.shape[0]
            n_neighbors = min(n_to_show + 1, max_possible)

            _, suggestion = self.model.kneighbors(
                self.book_pivot.iloc[book_id, :].values.reshape(1, -1),
                n_neighbors=n_neighbors
            )
            neighbor_ids = suggestion[0].tolist()

            # drop self if present
            if neighbor_ids and neighbor_ids[0] == book_id:
                neighbor_ids = neighbor_ids[1:]

            # clip to requested count
            neighbor_ids = neighbor_ids[:n_to_show]

            neighbor_names = [self.book_pivot.index[i] for i in neighbor_ids]
            posters = self.fetch_posters(neighbor_ids)
            return neighbor_names, posters
        except Exception as e:
            raise AppException(e, sys) from e

    def train_engine(self):
        try:
            obj = TrainingPipeline()
            with st.spinner("Training recommender…"):
                obj.start_training_pipeline()
            st.success("Training Completed!")
            logging.info("Training pipeline finished.")
        except Exception as e:
            raise AppException(e, sys) from e

    def recommendations_engine(self, selected_book: str, n_to_show: int = 5):
        try:
            names, posters = self.recommend_book(selected_book, n_to_show=n_to_show)
            n = len(names)
            if n == 0:
                st.info("No recommendations found.")
                return

            cols = st.columns(n)
            for i in range(n):
                with cols[i]:
                    full_name = names[i]
                    display_name = full_name if len(full_name) <= 25 else (full_name[:22] + "…")
                    st.markdown(
                        (
                            '<div style="text-align:center; font-size:14px; white-space:nowrap; '
                            'overflow:hidden; text-overflow:ellipsis;" title="{0}">{1}</div>'
                        ).format(full_name, display_name),
                        unsafe_allow_html=True,
                    )
                    url = posters[i] if i < len(posters) else None
                    show_img(url)
        except Exception as e:
            raise AppException(e, sys) from e


# ------------------------------ Main UI ------------------------------
if __name__ == "__main__":
    st.header("📚 Readora")
    st.caption("Minimal book recommendations using collaborative filtering (KNN with cosine similarity on a user–item pivot table).")

    # Load only the names early (cheap)
    try:
        app_cfg = AppConfiguration()
        rec_cfg = app_cfg.get_recommendation_config()
        book_names_attr = getattr(rec_cfg, "book_name_serialized_objects", None)
        if book_names_attr is None:
            book_names_path = Path(__file__).resolve().parent / "artifacts" / "serialized_objects" / "book_names.pkl"
        else:
            book_names_path = Path(book_names_attr).resolve()
        book_names = _load_pickled(book_names_path)
    except Exception as e:
        st.error("Failed to load book_names.pkl: {}".format(e))
        st.stop()

    # Lazy instantiate recommender so cold-start is fast
    if "recommender" not in st.session_state:
        st.session_state["recommender"] = None

    def get_recommender():
        if st.session_state["recommender"] is None:
            st.session_state["recommender"] = Recommendation(app_cfg)  # heavy (cached), only once
        return st.session_state["recommender"]

    # --- Layout (as requested) ---
    train_clicked = st.button("Train Recommender System")
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

    selected = st.selectbox("Type or select a book from the dropdown", options=book_names)
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

    n_to_show = st.slider("Number of recommendations", min_value=3, max_value=10, value=5, step=1)
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

    show_clicked = st.button("Show Recommendation")

    if train_clicked:
        try:
            get_recommender().train_engine()
        except Exception as e:
            st.error("Training failed: {}".format(e))

    if show_clicked:
        try:
            with st.spinner("Finding recommendations…"):
                get_recommender().recommendations_engine(selected, n_to_show=n_to_show)
        except Exception as e:
            st.error("Recommendation failed: {}".format(e))
