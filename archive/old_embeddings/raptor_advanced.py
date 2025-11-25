"""
Advanced RAPTOR Implementation with Clustering
"""
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict
import uuid
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class RaptorBuilder:
    """Build RAPTOR hierarchy with clustering"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 db_url: str = None, gemini_api_key: str = None):
        self.model = SentenceTransformer(model_name)
        
        # Use individual connection parameters (Transaction pooler)
        self.db_config = {
            'user': os.getenv("user"),
            'password': os.getenv("password"),
            'host': os.getenv("host"),
            'port': int(os.getenv("port", "5432")),
            'dbname': os.getenv("dbname")
        }
        
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    
    def cluster_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> Dict[int, List[int]]:
        """Cluster chunks using K-means"""
        if len(chunks) <= 3:
            return {0: list(range(len(chunks)))}
        
        # Auto-determine clusters: aim for 3-7 chunks per cluster
        n_clusters = max(2, min(len(chunks) // 5, 10))
        
        X = np.array(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Group indices by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        print(f"  Created {len(clusters)} clusters from {len(chunks)} chunks")
        return clusters
    
    
    def summarize_cluster(self, chunk_texts: List[str]) -> str:
        """Summarize cluster with Gemini"""
        combined = "\n\n---\n\n".join(chunk_texts)
        
        if len(combined) > 8000:
            combined = combined[:8000] + "...[truncated]"
        
        prompt = f"""Summarize this research content, preserving key details:

{combined}

Provide a 2-3 paragraph summary:"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Warning: Summarization error: {e}")
            return combined[:2000]
    
    
    def build_hierarchy_with_clustering(self, chunks: List[Dict], source_file: str, user_id: str) -> int:
        """Build RAPTOR hierarchy using clustering"""
        print(f"\nBuilding RAPTOR Hierarchy with Clustering...")
        
        current_level = chunks
        level_num = 0
        
        while len(current_level) > 1:
            level_num += 1
            print(f"\n  Creating Level {level_num} from {len(current_level)} chunks...")
            
            # Get embeddings
            texts = [c['text'] for c in current_level]
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()
            
            # Cluster
            clusters = self.cluster_chunks(current_level, embeddings)
            
            # Summarize each cluster
            next_level = []
            for cluster_id, indices in clusters.items():
                print(f"    Summarizing cluster {cluster_id + 1}/{len(clusters)}...")
                
                cluster_texts = [current_level[idx]['text'] for idx in indices]
                summary = self.summarize_cluster(cluster_texts)
                
                summary_chunk = {
                    'text': summary,
                    'id': str(uuid.uuid4()),
                    'children': [current_level[idx]['id'] for idx in indices]
                }
                next_level.append(summary_chunk)
            
            # Store level
            self._store_level(next_level, level_num, source_file, user_id)
            current_level = next_level
        
        print(f"\nRAPTOR hierarchy complete: {level_num} levels")
        return level_num
    
    
    def _store_level(self, chunks: List[Dict], level_num: int, source_file: str, user_id: str):
        """Store hierarchy level in database"""
        texts = [c['text'] for c in chunks]
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        data_to_insert = []
        for chunk, vector in zip(chunks, vectors):
            metadata = {
                "source_file": source_file,
                "children": chunk.get('children', [])
            }
            data_to_insert.append((
                chunk['id'],
                chunk['text'],
                vector.tolist(),
                json.dumps(metadata),
                user_id,
                level_num
            ))
        
        conn = psycopg2.connect(**self.db_config)
        try:
            with conn.cursor() as cur:
                execute_values(cur, 
                    "INSERT INTO documents (doc_id, content, embedding, metadata, user_id, hierarchy_level) VALUES %s",
                    data_to_insert)
                conn.commit()
            print(f"  Stored Level {level_num}: {len(data_to_insert)} summaries")
        finally:
            conn.close()