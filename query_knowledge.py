from text_vector_processor import TextVectorProcessor
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class KnowledgeQuerySystem:
    def __init__(self, vector_name="legal_knowledge", api_key=None):
        """Initialize the query system with the vector store name and Groq API key."""
        if api_key is None:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key is None:
                raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Get the absolute path for vector store
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.store_path = os.path.join(base_dir, "legal_text_vectorstore")
        
        # Initialize text vector processor
        self.processor = TextVectorProcessor(storage_dir=self.store_path)
        self.api_key = api_key
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
    
    def process_documents(self, docs_path):
        """Process PDF documents from a directory or single file."""
        try:
            if not os.path.exists(docs_path):
                raise FileNotFoundError(f"The path {docs_path} does not exist")

            # Load documents based on whether it's a file or directory
            if os.path.isfile(docs_path):
                loader = PyPDFLoader(docs_path)
                documents = loader.load()
            else:
                loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()

            if not documents:
                raise ValueError("No documents were loaded from the provided path")

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            # Create embeddings and store in FAISS
            self.vectorstore = FAISS.from_documents(chunks, self.processor.embeddings)
            
            # Save the vectorstore
            self.vectorstore.save_local(os.path.join(self.store_path, "case_documents"))
            
            # Initialize the chat model and memory
            self.llm = ChatGroq(temperature=0, api_key=self.api_key)
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create the conversational chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(),
                memory=self.memory,
                return_source_documents=True,
                output_key="answer"
            )
            
            return True
            
        except Exception as e:
            raise Exception(f"Error processing documents: {str(e)}")
    
    def analyze_case(self, question=None):
        """Analyze the case documents and provide a comprehensive analysis."""
        if self.vectorstore is None:
            return {"error": "No documents have been processed. Please process documents first."}
        
        try:
            # If no specific question is provided, ask for a general case analysis
            if question is None:
                question = "Please provide a comprehensive analysis of this legal case, including key facts, legal issues, and potential conclusions based on the available documents."
            
            result = self.qa_chain.invoke({"question": question})
            return {
                'analysis': result['answer'],
                'source_documents': [{
                    'content': doc.page_content,
                    'metadata': doc.metadata
                } for doc in result['source_documents']]
            }
        except Exception as e:
            return {"error": str(e)}

def main():
    # Example usage
    try:
        # Initialize the knowledge query system with the Groq API key
        query_system = KnowledgeQuerySystem(api_key="")
        
        print(".")
        
        # Ask for the documents directory
        docs_path = input("\nPlease enter the directory path or PDF file containing the case documents: ")
        
        # Process the documents
        print("\nProcessing documents...")
        query_system.process_documents(docs_path)
        print("Documents processed successfully.")
        
        # Get initial case analysis
        print("\nGenerating initial case analysis...")
        analysis = query_system.analyze_case()
        
        if 'analysis' in analysis:
            print("\nCase Analysis:")
            print(analysis['analysis'])
            print("\nConsulted Sources:")
            for source in analysis['source_documents']:
                print(f"\nDocument: {source['metadata'].get('source', 'Unknown')}")
                print(f"Page: {source['metadata'].get('page', 'N/A')}")
        else:
            print(f"Error: {analysis['error']}")
        
        print("\nYou can ask specific questions about the case. Type 'exit' to end.")
        
        while True:
            # Get user input
            question = input("\nPlease enter your question: ")
            
            # Check if user wants to exit
            if question.lower() in ['exit', 'quit']:
                print("\n")
                break
            
            # Process the question
            response = query_system.analyze_case(question)
            
            if 'analysis' in response:
                print("\nResponse:")
                print(response['analysis'])
                print("\nConsulted Sources:")
                for source in response['source_documents']:
                    print(f"\nDocument: {source['metadata'].get('source', 'Unknown')}")
                    print(f"Page: {source['metadata'].get('page', 'N/A')}")
            else:
                print(f"Error: {response['error']}")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()