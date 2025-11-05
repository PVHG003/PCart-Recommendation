# recommendation_engine.py
# Complete recommendation system for Next.js ecommerce integration
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import scipy.sparse as sps
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# ============================================
# 1. DATABASE CONNECTION
# ============================================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://username:password@ep-withered-lake-adw512mi-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"
)

def get_engine(url: str = DATABASE_URL) -> create_engine:
    """
    Creates a SQLAlchemy engine for database connection.

    Args:
        url (str): The database connection URL. Defaults to DATABASE_URL environment variable.

    Returns:
        create_engine: A SQLAlchemy engine object for database operations.

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    return create_engine(url, pool_pre_ping=True)

def load_products(engine: create_engine) -> pd.DataFrame:
    """
    Loads product data from the database where products are in stock.

    Args:
        engine (create_engine): The SQLAlchemy engine for database connection.

    Returns:
        pd.DataFrame: A DataFrame containing product details (id, name, description, category, price, etc.).

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    query = """
            SELECT id, 
                   name, 
                   description, 
                   category, 
                   price, 
                   mrp, 
                   images, 
                   "storeId", 
                   "inStock", 
                   "createdAt"
            FROM "Product"
            WHERE "inStock" = true 
            """
    return pd.read_sql_query(text(query), engine)

def load_ratings(engine: create_engine) -> pd.DataFrame:
    """
    Loads rating data from the database.

    Args:
        engine (create_engine): The SQLAlchemy engine for database connection.

    Returns:
        pd.DataFrame: A DataFrame containing rating details (id, rating, review, userId, productId, etc.).

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    query = """
            SELECT id, 
                   rating, 
                   review, 
                   "userId", 
                   "productId", 
                   "orderId", 
                   "createdAt"
            FROM "Rating" 
            """
    return pd.read_sql_query(text(query), engine)

def load_user_interactions(engine: create_engine) -> pd.DataFrame:
    """
    Loads user-product interaction data by joining Order and OrderItem tables.

    Args:
        engine (create_engine): The SQLAlchemy engine for database connection.

    Returns:
        pd.DataFrame: A DataFrame containing user interaction details (userId, productId, quantity, etc.).

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    query = """
            SELECT o."userId", 
                   oi."productId", 
                   oi.quantity, 
                   oi.price, 
                   o."createdAt", 
                   o.status
            FROM "OrderItem" oi
                     JOIN "Order" o ON oi."orderId" = o.id
            WHERE o.status IN ('DELIVERED', 'SHIPPED') 
            """
    return pd.read_sql_query(text(query), engine)

def load_user_ratings_interactions(engine: create_engine) -> pd.DataFrame:
    """
    Combines purchase data and ratings for collaborative filtering.

    Args:
        engine (create_engine): The SQLAlchemy engine for database connection.

    Returns:
        pd.DataFrame: A DataFrame with combined user interaction and rating data (userId, productId, implicit_rating, etc.).

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    query = """
            SELECT o."userId", 
                   oi."productId", 
                   COALESCE(r.rating, 3) as implicit_rating, 
                   oi.quantity, 
                   o."createdAt"
            FROM "OrderItem" oi
                     JOIN "Order" o ON oi."orderId" = o.id
                     LEFT JOIN "Rating" r ON r."userId" = o."userId"
                AND r."productId" = oi."productId"
            WHERE o.status IN ('DELIVERED', 'SHIPPED') 
            """
    return pd.read_sql_query(text(query), engine)

# ============================================
# 2. CONTENT-BASED FILTERING
# ============================================
class ContentBasedRecommender:
    def __init__(self):
        self.products_df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.product_indices = None

    def fit(self, products_df: pd.DataFrame) -> None:
        """
        Trains the content-based recommendation model using product data.

        Args:
            products_df (pd.DataFrame): DataFrame containing product details (id, name, category, description).

        Returns:
            None: Updates the instance attributes with trained model components.

        Author: Pham Viet Hung
        Date: November 05, 2025
        """
        self.products_df = products_df.copy()
        # Create combined text field
        self.products_df['combined_text'] = (
                self.products_df['name'].fillna('') + ' ' +
                self.products_df['category'].fillna('') + ' ' +
                self.products_df['description'].fillna('')
        )
        # Build TF-IDF matrix
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1
        )
        self.tfidf_matrix = self.tfidf.fit_transform(
            self.products_df['combined_text']
        )
        # Create product ID to index mapping
        self.product_indices = pd.Series(
            self.products_df.index,
            index=self.products_df['id']
        ).to_dict()
        print(f"✓ Content-based model trained on {len(self.products_df)} products")

    def recommend(self, product_id: str, top_n: int = 10) -> List[dict]:
        """
        Generates content-based recommendations for a given product.

        Args:
            product_id (str): The ID of the product to find similar products for.
            top_n (int): Number of recommendations to return. Defaults to 10.

        Returns:
            List[dict]: A list of dictionaries containing recommended product details and similarity scores.

        Author: Pham Viet Hung
        Date: November 05, 2025
        """
        try:
            # Get product index
            idx = self.product_indices[product_id]
            # Calculate cosine similarities
            cosine_sim = linear_kernel(
                self.tfidf_matrix[idx:idx + 1],
                self.tfidf_matrix
            ).flatten()
            # Get top similar products (excluding itself)
            similar_indices = cosine_sim.argsort()[::-1][1:top_n + 1]
            # Build results
            recommendations = []
            for i in similar_indices:
                product = self.products_df.iloc[i]
                recommendations.append({
                    'id': product['id'],
                    'name': product['name'],
                    'category': product['category'],
                    'price': float(product['price']) if pd.notna(product['price']) else None,
                    'mrp': float(product['mrp']) if pd.notna(product['mrp']) else None,
                    'images': product['images'] if isinstance(product['images'], list) else [],
                    'storeId': product['storeId'],
                    'score': float(cosine_sim[i])
                })
            return recommendations
        except KeyError:
            raise ValueError(f"Product {product_id} not found")

# ============================================
# 3. COLLABORATIVE FILTERING
# ============================================
class CollaborativeRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.users = None
        self.products = None
        self.user_idx_map = None
        self.product_idx_map = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, interactions_df: pd.DataFrame, n_factors: int = 50, n_iterations: int = 15, reg: float = 0.01) -> None:
        """
        Trains the collaborative filtering model using Alternating Least Squares (ALS).

        Args:
            interactions_df (pd.DataFrame): DataFrame containing user-product interactions (userId, productId, quantity, implicit_rating).
            n_factors (int): Number of latent factors for matrix factorization. Defaults to 50.
            n_iterations (int): Number of ALS iterations. Defaults to 15.
            reg (float): Regularization parameter for ALS. Defaults to 0.01.

        Returns:
            None: Updates the instance attributes with trained model components.

        Author: Pham Viet Hung
        Date: November 05, 2025
        """
        # Aggregate interactions per user-product pair
        agg_interactions = interactions_df.groupby(['userId', 'productId']).agg({
            'quantity': 'sum',
            'implicit_rating': 'mean'
        }).reset_index()
        # Create confidence score (combination of quantity and rating)
        agg_interactions['confidence'] = (
                agg_interactions['quantity'] *
                agg_interactions['implicit_rating']
        )
        # Create user-item matrix
        users = agg_interactions['userId'].unique()
        products = agg_interactions['productId'].unique()
        self.user_idx_map = {u: i for i, u in enumerate(users)}
        self.product_idx_map = {p: i for i, p in enumerate(products)}
        # Build sparse matrix (users x products)
        rows = agg_interactions['userId'].map(self.user_idx_map)
        cols = agg_interactions['productId'].map(self.product_idx_map)
        data = agg_interactions['confidence'].astype(float)
        self.user_item_matrix = sps.coo_matrix(
            (data, (rows, cols)),
            shape=(len(users), len(products))
        ).tocsr()
        self.users = users
        self.products = products
        # Train ALS
        self._train_als(n_factors, n_iterations, reg)
        print(f"Collaborative model trained on {len(users)} users and {len(products)} products")

    def _train_als(self, n_factors: int, n_iterations: int, reg: float) -> None:
        """
        Implements Alternating Least Squares (ALS) for collaborative filtering.

        Args:
            n_factors (int): Number of latent factors for matrix factorization.
            n_iterations (int): Number of ALS iterations.
            reg (float): Regularization parameter for ALS.

        Returns:
            None: Updates user_factors and item_factors attributes.

        Author: Pham Viet Hung
        Date: November 05, 2025
        """
        n_users, n_items = self.user_item_matrix.shape
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
        # ALS iterations
        for iteration in range(n_iterations):
            # Fix item factors, optimize user factors
            for u in range(n_users):
                items_u = self.user_item_matrix[u].indices
                if len(items_u) > 0:
                    A = self.item_factors[items_u].T @ self.item_factors[items_u] + reg * np.eye(n_factors)
                    b = self.user_item_matrix[u, items_u].toarray().flatten() @ self.item_factors[items_u]
                    self.user_factors[u] = np.linalg.solve(A, b)
            # Fix user factors, optimize item factors
            user_item_matrix_csc = self.user_item_matrix.tocsc()
            for i in range(n_items):
                users_i = user_item_matrix_csc[:, i].indices
                if len(users_i) > 0:
                    A = self.user_factors[users_i].T @ self.user_factors[users_i] + reg * np.eye(n_factors)
                    b = user_item_matrix_csc[users_i, i].toarray().flatten() @ self.user_factors[users_i]
                    self.item_factors[i] = np.linalg.solve(A, b)
            if (iteration + 1) % 5 == 0:
                print(f"  ALS iteration {iteration + 1}/{n_iterations}")

    def recommend(self, user_id: str, top_n: int = 10, exclude_purchased: bool = True) -> List[dict]:
        """
        Generates personalized recommendations for a user using collaborative filtering.

        Args:
            user_id (str): The ID of the user to generate recommendations for.
            top_n (int): Number of recommendations to return. Defaults to 10.
            exclude_purchased (bool): Whether to exclude already purchased items. Defaults to True.

        Returns:
            List[dict]: A list of dictionaries containing recommended product IDs and scores.

        Author: Pham Viet Hung
        Date: November 05, 2025
        """
        try:
            user_idx = self.user_idx_map[user_id]
            # Calculate scores for all products
            scores = self.user_factors[user_idx] @ self.item_factors.T
            # Exclude already purchased items
            if exclude_purchased:
                purchased = self.user_item_matrix[user_idx].indices
                scores[purchased] = -np.inf
            # Get top N
            top_indices = np.argsort(scores)[::-1][:top_n]
            recommendations = []
            for idx in top_indices:
                if scores[idx] > -np.inf:
                    recommendations.append({
                        'productId': self.products[idx],
                        'score': float(scores[idx])
                    })
            return recommendations
        except KeyError:
            raise ValueError(f"User {user_id} not found")

# ============================================
# 4. HYBRID RECOMMENDER
# ============================================
class HybridRecommender:
    def __init__(self, content_model: ContentBasedRecommender, collab_model: CollaborativeRecommender, products_df: pd.DataFrame):
        self.content_model = content_model
        self.collab_model = collab_model
        self.products_df = products_df

    def recommend(self, user_id: Optional[str] = None, product_id: Optional[str] = None, top_n: int = 10) -> List[dict]:
        """
        Generates hybrid recommendations combining content-based and collaborative filtering.

        Args:
            user_id (Optional[str]): The ID of the user for collaborative filtering. Defaults to None.
            product_id (Optional[str]): The ID of the product for content-based filtering. Defaults to None.
            top_n (int): Number of recommendations to return. Defaults to 10.

        Returns:
            List[dict]: A list of dictionaries containing recommended product details and scores.

        Author: Pham Viet Hung
        Date: November 05, 2025
        """
        if user_id and product_id:
            # Combine both approaches
            try:
                collab_recs = self.collab_model.recommend(user_id, top_n * 2)
                content_recs = self.content_model.recommend(product_id, top_n * 2)
                combined = self._merge_recommendations(collab_recs, content_recs, top_n)
                return combined
            except ValueError:
                # Fallback to content-based only
                return self.content_model.recommend(product_id, top_n)
        elif user_id:
            # Collaborative only
            try:
                recs = self.collab_model.recommend(user_id, top_n)
                return self._enrich_with_product_data(recs)
            except ValueError:
                # User not found, return popular products
                return self._get_popular_products(top_n)
        elif product_id:
            # Content-based only
            return self.content_model.recommend(product_id, top_n)
        else:
            # Return popular products
            return self._get_popular_products(top_n)

    def _merge_recommendations(self, collab_recs: List[dict], content_recs: List[dict], top_n: int) -> List[dict]:
        """
        Merges collaborative and content-based recommendations with weighted scores.

        Args:
            collab_recs (List[dict]): List of collaborative filtering recommendations.
            content_recs (List[dict]): List of content-based recommendations.
            top_n (int): Number of recommendations to return.

        Returns:
            List[dict]: A list of merged recommendations with product details and combined scores.

        Author: Pham Viet Hung
        Date: November 05, 2025
        """
        # Weight: 60% collaborative, 40% content-based
        scores = {}
        for rec in collab_recs:
            scores[rec['productId']] = rec['score'] * 0.6
        for rec in content_recs:
            pid = rec['id']
            scores[pid] = scores.get(pid, 0) + rec['score'] * 0.4
        # Sort and get top N
        sorted_products = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        # Enrich with product data
        recommendations = []
        for pid, score in sorted_products:
            product_match = self.products_df[self.products_df['id'] == pid]
            if len(product_match) > 0:
                product = product_match.iloc[0]
                recommendations.append({
                    'id': pid,
                    'name': product['name'],
                    'category': product['category'],
                    'price': float(product['price']) if pd.notna(product['price']) else None,
                    'mrp': float(product['mrp']) if pd.notna(product['mrp']) else None,
                    'images': product['images'] if isinstance(product['images'], list) else [],
                    'storeId': product['storeId'],
                    'score': float(score)
                })
        return recommendations

    def _enrich_with_product_data(self, recs: List[dict]) -> List[dict]:
        """
        Adds product details to collaborative filtering recommendations.

        Args:
            recs (List[dict]): List of recommendations with product IDs and scores.

        Returns:
            List[dict]: A list of recommendations enriched with product details (name, category, price, etc.).

        Author: Pham Viet Hung
        Date: November 05, 2025
        """
        enriched = []
        for rec in recs:
            product_match = self.products_df[
                self.products_df['id'] == rec['productId']
                ]
            if len(product_match) > 0:
                product = product_match.iloc[0]
                enriched.append({
                    'id': rec['productId'],
                    'name': product['name'],
                    'category': product['category'],
                    'price': float(product['price']) if pd.notna(product['price']) else None,
                    'mrp': float(product['mrp']) if pd.notna(product['mrp']) else None,
                    'images': product['images'] if isinstance(product['images'], list) else [],
                    'storeId': product['storeId'],
                    'score': rec['score']
                })
        return enriched

    def _get_popular_products(self, top_n: int) -> List[dict]:
        """
        Returns a list of popular or recent products as a fallback.

        Args:
            top_n (int): Number of popular products to return.

        Returns:
            List[dict]: A list of dictionaries containing popular product details and a default score.

        Author: Pham Viet Hung
        Date: November 05, 2025
        """
        # Sort by createdAt (most recent) or you could add view/purchase counts
        popular = self.products_df.sort_values('createdAt', ascending=False).head(top_n)
        return [{
            'id': row['id'],
            'name': row['name'],
            'category': row['category'],
            'price': float(row['price']) if pd.notna(row['price']) else None,
            'mrp': float(row['mrp']) if pd.notna(row['mrp']) else None,
            'images': row['images'] if isinstance(row['images'], list) else [],
            'storeId': row['storeId'],
            'score': 1.0
        } for _, row in popular.iterrows()]

# ============================================
# 5. FASTAPI APPLICATION
# ============================================
app = FastAPI(title="Ecommerce Recommendation API", version="1.0")
# Add CORS middleware for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global models (loaded at startup)
content_recommender = None
collab_recommender = None
hybrid_recommender = None
products_df = None

@app.on_event("startup")
async def load_models() -> None:
    """
    Loads or trains recommendation models at application startup.

    Args:
        None

    Returns:
        None: Updates global model variables (content_recommender, collab_recommender, hybrid_recommender, products_df).

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    global content_recommender, collab_recommender, hybrid_recommender, products_df
    print("=" * 50)
    print("Starting Recommendation Engine...")
    print("=" * 50)
    # Check if pre-trained models exist
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)
    try:
        # Try loading pre-trained models
        with open(models_path / "content_model.pkl", "rb") as f:
            content_recommender = pickle.load(f)
        with open(models_path / "collab_model.pkl", "rb") as f:
            collab_recommender = pickle.load(f)
        with open(models_path / "products_df.pkl", "rb") as f:
            products_df = pickle.load(f)
        print("✓ Loaded pre-trained models from disk")
    except FileNotFoundError:
        print("No pre-trained models found. Training new models...")
        # Train new models
        engine = get_engine()
        # Load data
        print("\nLoading data from database...")
        products_df = load_products(engine)
        print(f"  - Loaded {len(products_df)} products")
        interactions_df = load_user_ratings_interactions(engine)
        print(f"  - Loaded {len(interactions_df)} user interactions")
        # Train content-based
        print("\nTraining content-based recommender...")
        content_recommender = ContentBasedRecommender()
        content_recommender.fit(products_df)
        # Train collaborative (if enough data)
        if len(interactions_df) >= 50:
            print("\nTraining collaborative recommender...")
            collab_recommender = CollaborativeRecommender()
            collab_recommender.fit(interactions_df, n_factors=30, n_iterations=10)
        else:
            print(f"\n⚠ Not enough interaction data ({len(interactions_df)} records)")
            print("  Need at least 50 interactions for collaborative filtering")
            collab_recommender = None
        # Save models
        print("\nSaving models to disk...")
        with open(models_path / "content_model.pkl", "wb") as f:
            pickle.dump(content_recommender, f)
        if collab_recommender:
            with open(models_path / "collab_model.pkl", "wb") as f:
                pickle.dump(collab_recommender, f)
        with open(models_path / "products_df.pkl", "wb") as f:
            pickle.dump(products_df, f)
        print("✓ Models trained and saved")
    # Initialize hybrid recommender
    if collab_recommender:
        hybrid_recommender = HybridRecommender(
            content_recommender,
            collab_recommender,
            products_df
        )
        print("✓ Hybrid recommender initialized")
    else:
        print("⚠ Hybrid recommender not available (using content-based only)")
    print("=" * 50)
    print("Recommendation Engine Ready!")
    print("=" * 50)

# Pydantic models
class RecommendationResponse(BaseModel):
    recommendations: List[dict]
    algorithm: str
    count: int

@app.get("/")
async def root() -> dict:
    """
    Provides a health check endpoint for the recommendation API.

    Args:
        None

    Returns:
        dict: A dictionary with API status, version, and loaded model information.

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    return {
        "message": "Recommendation API is running",
        "version": "1.0",
        "status": "healthy",
        "models": {
            "content_based": content_recommender is not None,
            "collaborative": collab_recommender is not None,
            "hybrid": hybrid_recommender is not None
        }
    }

@app.get("/health")
async def health() -> dict:
    """
    Provides a detailed health check of the recommendation system.

    Args:
        None

    Returns:
        dict: A dictionary with system status, product count, and model loading status.

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    return {
        "status": "healthy",
        "products_count": len(products_df) if products_df is not None else 0,
        "models_loaded": {
            "content": content_recommender is not None,
            "collaborative": collab_recommender is not None,
            "hybrid": hybrid_recommender is not None
        }
    }

@app.get("/api/recommendations/similar/{product_id}", response_model=RecommendationResponse)
async def get_similar_products(product_id: str, limit: int = 10) -> dict:
    """
    Returns products similar to the specified product using content-based filtering.

    Args:
        product_id (str): The ID of the product to find similar products for.
        limit (int): Number of recommendations to return. Defaults to 10.

    Returns:
        dict: A dictionary containing recommendations, algorithm type, and count.

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    if not content_recommender:
        raise HTTPException(status_code=503, detail="Content model not loaded")
    try:
        recommendations = content_recommender.recommend(product_id, top_n=limit)
        return {
            "recommendations": recommendations,
            "algorithm": "content-based",
            "count": len(recommendations)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/api/recommendations/personalized/{user_id}", response_model=RecommendationResponse)
async def get_personalized_recommendations(user_id: str, limit: int = 10) -> dict:
    """
    Returns personalized recommendations for a user using collaborative or hybrid filtering.

    Args:
        user_id (str): The ID of the user to generate recommendations for.
        limit (int): Number of recommendations to return. Defaults to 10.

    Returns:
        dict: A dictionary containing recommendations, algorithm type, and count.

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    try:
        if hybrid_recommender:
            recommendations = hybrid_recommender.recommend(user_id=user_id, top_n=limit)
            algorithm = "hybrid" if collab_recommender else "popular"
        elif collab_recommender:
            recs = collab_recommender.recommend(user_id, top_n=limit)
            # Enrich with product data
            enriched = []
            for rec in recs:
                product_match = products_df[products_df['id'] == rec['productId']]
                if len(product_match) > 0:
                    product = product_match.iloc[0]
                    enriched.append({
                        'id': rec['productId'],
                        'name': product['name'],
                        'category': product['category'],
                        'price': float(product['price']) if pd.notna(product['price']) else None,
                        'mrp': float(product['mrp']) if pd.notna(product['mrp']) else None,
                        'images': product['images'] if isinstance(product['images'], list) else [],
                        'storeId': product['storeId'],
                        'score': rec['score']
                    })
            recommendations = enriched
            algorithm = "collaborative"
        else:
            # Fallback to popular products
            popular = products_df.sort_values('createdAt', ascending=False).head(limit)
            recommendations = [{
                'id': row['id'],
                'name': row['name'],
                'category': row['category'],
                'price': float(row['price']) if pd.notna(row['price']) else None,
                'mrp': float(row['mrp']) if pd.notna(row['mrp']) else None,
                'images': row['images'] if isinstance(row['images'], list) else [],
                'storeId': row['storeId'],
                'score': 1.0
            } for _, row in popular.iterrows()]
            algorithm = "popular"
        return {
            "recommendations": recommendations,
            "algorithm": algorithm,
            "count": len(recommendations)
        }
    except ValueError:
        # User not found, return popular products
        popular = products_df.sort_values('createdAt', ascending=False).head(limit)
        recommendations = [{
            'id': row['id'],
            'name': row['name'],
            'category': row['category'],
            'price': float(row['price']) if pd.notna(row['price']) else None,
            'mrp': float(row['mrp']) if pd.notna(row['mrp']) else None,
            'images': row['images'] if isinstance(row['images'], list) else [],
            'storeId': row['storeId'],
            'score': 1.0
        } for _, row in popular.iterrows()]
        return {
            "recommendations": recommendations,
            "algorithm": "popular",
            "count": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/api/recommendations/hybrid", response_model=RecommendationResponse)
async def get_hybrid_recommendations(
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        limit: int = 10
) -> dict:
    """
    Returns hybrid recommendations combining collaborative and content-based filtering.

    Args:
        user_id (Optional[str]): The ID of the user for collaborative filtering. Defaults to None.
        product_id (Optional[str]): The ID of the product for content-based filtering. Defaults to None.
        limit (int): Number of recommendations to return. Defaults to 10.

    Returns:
        dict: A dictionary containing recommendations, algorithm type, and count.

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    if not user_id and not product_id:
        # Return popular products
        popular = products_df.sort_values('createdAt', ascending=False).head(limit)
        recommendations = [{
            'id': row['id'],
            'name': row['name'],
            'category': row['category'],
            'price': float(row['price']) if pd.notna(row['price']) else None,
            'mrp': float(row['mrp']) if pd.notna(row['mrp']) else None,
            'images': row['images'] if isinstance(row['images'], list) else [],
            'storeId': row['storeId'],
            'score': 1.0
        } for _, row in popular.iterrows()]
        return {
            "recommendations": recommendations,
            "algorithm": "popular",
            "count": len(recommendations)
        }
    if not hybrid_recommender:
        # Fallback to content-based only
        if product_id and content_recommender:
            return await get_similar_products(product_id, limit)
        else:
            raise HTTPException(
                status_code=503,
                detail="Hybrid model not available. Try /api/recommendations/similar/{product_id}"
            )
    try:
        recommendations = hybrid_recommender.recommend(
            user_id=user_id,
            product_id=product_id,
            top_n=limit
        )
        return {
            "recommendations": recommendations,
            "algorithm": "hybrid",
            "count": len(recommendations)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/api/recommendations/retrain")
async def retrain_models(secret_key: str = None) -> dict:
    """
    Retrains all recommendation models and updates global model variables.

    Args:
        secret_key (str): Optional secret key for authentication. Defaults to None.

    Returns:
        dict: A dictionary with retraining status, product count, interaction count, and model update status.

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    # Optional: Add authentication
    # if secret_key != os.getenv("RETRAIN_SECRET_KEY"):
    #     raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        print("\n" + "=" * 50)
        print("Starting model retraining...")
        print("=" * 50)
        engine = get_engine()
        # Reload data
        print("\nLoading fresh data from database...")
        new_products_df = load_products(engine)
        print(f"  - Loaded {len(new_products_df)} products")
        interactions_df = load_user_ratings_interactions(engine)
        print(f"  - Loaded {len(interactions_df)} interactions")
        # Retrain content-based
        print("\nRetraining content-based recommender...")
        new_content = ContentBasedRecommender()
        new_content.fit(new_products_df)
        # Retrain collaborative
        new_collab = None
        if len(interactions_df) >= 50:
            print("\nRetraining collaborative recommender...")
            new_collab = CollaborativeRecommender()
            new_collab.fit(interactions_df, n_factors=30, n_iterations=10)
        else:
            print(f"\n⚠ Not enough interaction data ({len(interactions_df)} records)")
        # Update global models
        global content_recommender, collab_recommender, hybrid_recommender, products_df
        content_recommender = new_content
        collab_recommender = new_collab
        products_df = new_products_df
        if collab_recommender:
            hybrid_recommender = HybridRecommender(
                content_recommender,
                collab_recommender,
                products_df
            )
            print("✓ Hybrid recommender updated")
        # Save models
        print("\nSaving updated models...")
        models_path = Path("models")
        with open(models_path / "content_model.pkl", "wb") as f:
            pickle.dump(content_recommender, f)
        if collab_recommender:
            with open(models_path / "collab_model.pkl", "wb") as f:
                pickle.dump(collab_recommender, f)
        with open(models_path / "products_df.pkl", "wb") as f:
            pickle.dump(products_df, f)
        print("=" * 50)
        print("Retraining complete!")
        print("=" * 50)
        return {
            "message": "Models retrained successfully",
            "products_count": len(products_df),
            "interactions_count": len(interactions_df),
            "models_updated": {
                "content": True,
                "collaborative": collab_recommender is not None,
                "hybrid": hybrid_recommender is not None
            }
        }
    except Exception as e:
        print(f"\n Retraining failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/api/recommendations/popular")
async def get_popular_products(limit: int = 10) -> dict:
    """
    Returns a list of popular or trending products based on recency.

    Args:
        limit (int): Number of popular products to return. Defaults to 10.

    Returns:
        dict: A dictionary containing popular product recommendations, algorithm type, and count.

    Author: Pham Viet Hung
    Date: November 05, 2025
    """
    if products_df is None:
        raise HTTPException(status_code=503, detail="Products data not loaded")
    # Return most recent products (you can customize this logic)
    popular = products_df.sort_values('createdAt', ascending=False).head(limit)
    recommendations = [{
        'id': row['id'],
        'name': row['name'],
        'category': row['category'],
        'price': float(row['price']) if pd.notna(row['price']) else None,
        'mrp': float(row['mrp']) if pd.notna(row['mrp']) else None,
        'images': row['images'] if isinstance(row['images'], list) else [],
        'storeId': row['storeId'],
        'inStock': row['inStock']
    } for _, row in popular.iterrows()]
    return {
        "recommendations": recommendations,
        "algorithm": "popular",
        "count": len(recommendations)
    }

# ============================================
# 6. MAIN ENTRY POINT
# ============================================
if __name__ == "__main__":
    # For development
    uvicorn.run(
        "recommendation_engine:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )