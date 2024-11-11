import numpy as np
from deepface import DeepFace
import chromadb
from chromadb.config import Settings
import os
from PIL import Image

class FaceSearchSystem:
    def __init__(self, collection_name="face_embeddings"):
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="face_database"
        ))
        
        # Create or get collection
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity for face embeddings
        )
        
    def extract_face_embedding(self, image_path):
        """Extract facial embedding from an image using DeepFace"""
        try:
            # Get embedding using DeepFace
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet",
                enforce_detection=True
            )
            
            # DeepFace returns a list of embeddings, we take the first one
            return np.array(embedding[0]['embedding'])
            
        except Exception as e:
            print(f"Error extracting embedding: {str(e)}")
            return None
    
    def add_face(self, image_path, metadata=None):
        """Add a face to the database"""
        try:
            # Generate embedding
            embedding = self.extract_face_embedding(image_path)
            
            if embedding is not None:
                # Convert embedding to list for ChromaDB
                embedding_list = embedding.tolist()
                
                # Generate an ID (you might want to use a more sophisticated method)
                doc_id = str(abs(hash(str(embedding_list))))
                
                # Add to ChromaDB
                self.collection.add(
                    embeddings=[embedding_list],
                    ids=[doc_id],
                    metadatas=[metadata or {}]
                )
                
                return doc_id
            
        except Exception as e:
            print(f"Error adding face: {str(e)}")
            return None
    
    def search_similar_faces(self, query_image_path, n_results=5):
        """Search for similar faces in the database"""
        try:
            # Get embedding for query image
            query_embedding = self.extract_face_embedding(query_image_path)
            
            if query_embedding is not None:
                # Search in ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results
                )
                
                return results
            
        except Exception as e:
            print(f"Error searching faces: {str(e)}")
            return None

