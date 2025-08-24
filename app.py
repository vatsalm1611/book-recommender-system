from pathlib import Path
import sys
import pickle
import numpy as np
import streamlit as st

from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException

try:
    _cache_resource = st.cache_resource
    def cache_resource(**kwargs):
        return _cache_resource(**kwargs)
except AttributeError:
    def cache_resource(**kwargs):
        kwargs.setdefault("allow_output_mutation", True)
        return st.cache(**kwargs)

def show_img(url):
    try:
        st.image(url, use_container_width=True)
    except TypeError:
        st.image(url, use_column_width=True)

@cache_resource(show_spinner=False)
def _load_pickled(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

class Recommendation:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.cfg = app_config.get_recommendation_config()
            self.book_pivot_path   = Path(self.cfg.book_pivot_serialized_objects).resolve()
            self.final_rating_path = Path(self.cfg.final_rating_serialized_objects).resolve()
            self.model_path        = Path(self.cfg.trained_model_path).resolve()
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
                match = np.where(self.final_rating['title'] == title)[0]
                idxs.append(int(match[0]) if len(match) else None)
            posters = []
            for ix in idxs:
                posters.append(None if ix is None else self.final_rating.iloc[ix].get('image_url', None))
            return posters
        except Exception as e:
            raise AppException(e, sys) from e

    def recommend_book(self, book_name: str):
        try:
            where = np.where(self.book_pivot.index == book_name)[0]
            if len(where) == 0:
                raise AppException(f"Book not found in pivot: {book_name}", sys)
            book_id = int(where[0])
            _, suggestion = self.model.kneighbors(
                self.book_pivot.iloc[book_id, :].values.reshape(1, -1),
                n_neighbors=6
            )
            neighbor_ids = suggestion[0].tolist()
            if neighbor_ids and neighbor_ids[0] == book_id:
                neighbor_ids = neighbor_ids[1:]
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

    def recommendations_engine(self, selected_book: str):
        try:
            names, posters = self.recommend_book(selected_book)
            n = min(5, len(names))
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
                    if url:
                        show_img(url)
                    else:
                        st.caption("No image available")
        except Exception as e:
            raise AppException(e, sys) from e

if __name__ == "__main__":
    st.header("📚 Readora")
    st.caption("Minimal book recommendations using collaborative filtering (KNN with cosine similarity on a user–item pivot table).")

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
        st.error(f"Failed to load book_names.pkl: {e}")
        st.stop()

    recommender = Recommendation(app_cfg)

    if st.button("Train Recommender System"):
        recommender.train_engine()

    selected = st.selectbox("Type or select a book from the dropdown", options=book_names)
    if st.button("Show Recommendation"):
        recommender.recommendations_engine(selected)
