from text_vector_processor import TextVectorProcessor

def main():
    # Initialize the Text Vector Processor
    processor = TextVectorProcessor(storage_dir="xxx")
    
    # Path to the legal text file
    text_path = r".txt"
    
    try:
        # Process the document and create vector embeddings
        print("Processing the legal text...")
        vectorstore = processor.process_text_file(text_path)
        
        # Save the vector store for future use
        print("Saving vector embeddings...")
        save_path = processor.save_vectorstore("legal_text_embeddings")
        print(f"Vector embeddings saved at: {save_path}")
        
        # Test the similarity search
        print("\nTesting similarity search...")
        query = "¿Cuáles son los derechos del consumidor?"
        results = processor.similarity_search(query, k=3)
        
        print("\nSearch Results:")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(doc.page_content)
            print("---")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()