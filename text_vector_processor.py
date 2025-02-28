from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
import json
import numpy as np

class TextVectorProcessor:
    def __init__(self, storage_dir="legal_text_vectorstore"):
        """Initialize the Text Vector Processor with a storage directory."""
        self.storage_dir = storage_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorstore = None
        self.chunks = None
        self.vectors = None

    def process_text_file(self, text_path):
        """Process text file and create vector embeddings."""
        try:
            # Check if the file exists
            if not os.path.exists(text_path):
                raise FileNotFoundError(f"The file {text_path} does not exist")

            # Read the text file
            with open(text_path, 'r', encoding='utf-8') as file:
                text_content = file.read()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.chunks = text_splitter.create_documents([text_content])

            if not self.chunks:
                raise ValueError("No text chunks were created from the file")

            # Create embeddings
            self.vectors = self.embeddings.embed_documents([chunk.page_content for chunk in self.chunks])
            
            # Create FAISS vectorstore for similarity search
            self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
            return self.vectorstore

        except Exception as e:
            raise Exception(f"Error processing text file: {str(e)}")

    def save_vectorstore(self, name):
        """Save the vector store to local storage in both FAISS and JSON formats."""
        if self.vectorstore is None or self.chunks is None or self.vectors is None:
            raise ValueError("No vectorstore to save. Process text file first.")
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Save FAISS vectorstore
        faiss_path = os.path.join(self.storage_dir, name)
        self.vectorstore.save_local(faiss_path)
        
        # Prepare data for JSON
        vector_data = {
            'chunks': [{
                'content': chunk.page_content,
                'metadata': chunk.metadata,
                'embedding': self.vectors[i].tolist() if isinstance(self.vectors[i], np.ndarray) else self.vectors[i]
            } for i, chunk in enumerate(self.chunks)]
        }
        
        # Save as JSON
        json_path = os.path.join(self.storage_dir, f"{name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, ensure_ascii=False, indent=2)
        
        return json_path

    def load_vectorstore(self, name):
        """Load a vector store from local storage."""
        # Try loading FAISS format first
        faiss_path = os.path.join(self.storage_dir, name)
        if os.path.exists(faiss_path):
            self.vectorstore = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            return self.vectorstore
        
        # If FAISS format not found, try JSON format
        json_path = os.path.join(self.storage_dir, f"{name}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                vector_data = json.load(f)
            
            # Convert JSON data back to documents
            documents = [Document(page_content=chunk['content'], metadata=chunk['metadata']) 
                        for chunk in vector_data['chunks']]
            
            # Create new FAISS vectorstore
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            return self.vectorstore
            
        raise FileNotFoundError(f"No vectorstore found at {faiss_path} or {json_path}")

    def similarity_search(self, query, k=3):
        """Perform similarity search on the loaded vectorstore."""
        if self.vectorstore is None:
            raise ValueError("No vectorstore loaded. Load or process text file first.")
        
        return self.vectorstore.similarity_search(query, k=k)