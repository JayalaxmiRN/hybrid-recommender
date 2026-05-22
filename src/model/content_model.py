"""
Content-Based Recommender
Uses SentenceTransformers to generate semantic embeddings of item metadata
and cosine similarity to find similar items.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, item_df, model_name='all-MiniLM-L6-v2'):
        """
        item_df: DataFrame with at least 'title' and 'combined' columns.
        'combined' = title + description + category (created by data_adapter).
        """
        self.df = item_df.reset_index(drop=True)
<<<<<<< HEAD:content_model.py
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
        )
        if "combined" not in self.df.columns:
            self.df["combined"] = (
                self.df.get("title", "").astype(str) + " " +
                self.df.get("author", "").astype(str) + " " +
                self.df.get("category", "").astype(str)
            )
        self.matrix = self.vectorizer.fit_transform(self.df['combined'].fillna(''))
        # Do not compute full similarity matrix here to avoid OOM
        self._title_to_idx = {
            t.lower(): i for i, t in enumerate(self.df['title'].astype(str))
=======
        self.model = SentenceTransformer(model_name)
        
        # Generate embeddings for all items
        texts = self.df['combined'].fillna('').tolist()
        self.matrix = self.model.encode(texts, show_progress_bar=False)
        
        self._title_to_idx = {
            t.lower(): i for i, t in enumerate(self.df['title'])
>>>>>>> main:src/model/content_model.py
        }

    def recommend(self, title, top_n=10):
        """
        Get content-based recommendations for a given item title.
        Returns list of dicts: [{ 'title', 'content_score' }, ...]
        """
        if title.lower() not in self._title_to_idx:
            return []

<<<<<<< HEAD:content_model.py
        title_key = title.lower()

        if title_key not in self._title_to_idx:
            return []

        idx = self._title_to_idx[title_key]

    
        query_vec = self.matrix[idx]
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        
        scores[idx] = float("-inf")

        top_indices = np.argsort(-scores)

        results = []
        seen = set() 
        for i in top_indices+1:
            t = str(self.df.iloc[i]["title"]).strip()
            key = t.lower()
            if key in seen:
=======
        idx = self._title_to_idx[title.lower()]
        query_vec = self.matrix[idx].reshape(1, -1)
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        
        sim_scores = list(enumerate(scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        results = []
        seen = set()
        for i, score in sim_scores:
            t = self.df.iloc[i]['title']
            if t.lower() == title.lower() or t in seen:
>>>>>>> main:src/model/content_model.py
                continue
            seen.add(key)
            
            results.append({
                "title": t,
                "content_score": float(scores[i])
            })
            
            if len(results) >= top_n:
                break

        return results

    def explain_similarity(self, source_title, candidate_title, top_n=5):
        """
        Return a placeholder or basic explanation since dense vectors 
        don't have interpretable individual features like TF-IDF terms.
        """
        if source_title.lower() not in self._title_to_idx or candidate_title.lower() not in self._title_to_idx:
            return []

        source_idx = self._title_to_idx[source_title.lower()]
        candidate_idx = self._title_to_idx[candidate_title.lower()]
        
        score = cosine_similarity(
            self.matrix[source_idx].reshape(1, -1), 
            self.matrix[candidate_idx].reshape(1, -1)
        )[0][0]
        
        return [{'term': 'semantic_similarity', 'score': round(float(score), 4)}]

    def search(self, query, top_n=20):
        """
        Search items by query text using semantic similarity.
        Returns list of matching item titles with scores.
        """
        query_vec = self.model.encode([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argpartition(-scores, top_n)[:top_n]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        seen = set()
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            t = self.df.iloc[idx]['title']
            if t in seen:
                continue
            seen.add(t)
            
            tp = self.df.at[idx, 'top_reviews'] if 'top_reviews' in self.df.columns else []
            top_reviews = tp if isinstance(tp, list) else []

            results.append({
                'title': t,
                'score': float(scores[idx]),
                'item_id': str(self.df.iloc[idx].get('item_id', idx)),
                'category': self.df.iloc[idx].get('category', ''),
                'description': str(self.df.iloc[idx].get('description', ''))[:200],
                'top_reviews': top_reviews,
            })
        return results
