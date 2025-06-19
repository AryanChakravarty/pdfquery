import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
import google.generativeai as genai

# Get the directory where policy_rag.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Debug prints
print("Current working directory:", os.getcwd())
print("Script location:", current_dir)
print("Looking for .env file in:", os.path.join(current_dir, ".env"))
print("Directory contents:", os.listdir(current_dir))

# Load environment variables from the correct directory
load_dotenv(dotenv_path=os.path.join(current_dir, ".env"))

# Get API key and verify it
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Configure Gemini with explicit API key
try:
    # Disable ADC and use API key only
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""  # Clear any ADC
    genai.configure(api_key=api_key)
    print("Gemini configuration successful with API key!")
except Exception as e:
    print(f"Error configuring Gemini: {str(e)}")
    raise

class PolicyRAG:
    def __init__(self, pdf_directory: str, persist_directory: str = "./policy_chroma"):
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
            task_type="retrieval_document"
        )
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None

    def load_and_chunk_documents(self) -> List:
        """Load PDFs and split them into chunks."""
        documents = []
        pdf_files = Path(self.pdf_directory).glob("*.pdf")
        
        print("\nProcessing PDF files:")
        for pdf_path in pdf_files:
            print(f"Loading {pdf_path.name}...")
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                # Add metadata about the source file
                for doc in docs:
                    doc.metadata["source"] = pdf_path.name
                    # Add page number to metadata
                    if "page" in doc.metadata:
                        doc.metadata["page_number"] = doc.metadata["page"]
                documents.extend(docs)
                print(f"Successfully loaded {len(docs)} pages from {pdf_path.name}")
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")

        if not documents:
            raise ValueError("No documents were loaded successfully")

        print("\nSplitting documents into chunks...")
        # Split documents into chunks with larger size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,     # Increased from 1000 to include more context
            chunk_overlap=400,   # Increased from 200 for better context continuity
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def create_vector_store(self, chunks: List):
        """Create and persist the vector store."""
        print("Creating vector store...")
        try:
            # Create the vector store with persistence
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            # In newer versions of Chroma, persistence is handled automatically
            print(f"Vector store created and saved to {self.persist_directory}")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise

    def initialize_llm(self):
        """Initialize the Gemini model with LangChain wrapper."""
        print("Initializing Gemini model...")
        try:
            # Initialize Gemini 1.5 Flash 8B model - most cost-effective option with better free tier limits
            self.llm = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-flash-8b",  # Using the most cost-effective model
                temperature=0.1,  # Lower temperature for more focused, factual responses
                google_api_key=api_key,
                convert_system_message_to_human=False,
                max_output_tokens=8192,  # Maximum output length
                top_p=0.95,  # Balanced between creativity and focus
                top_k=40  # Good balance for policy document processing
            )
            print("Model initialized successfully!")
        except Exception as e:
            print(f"Detailed error: {str(e)}")
            raise RuntimeError(f"Error initializing model: {str(e)}")

    def setup_qa_chain(self):
        """Set up the QA chain."""
        if not self.vectorstore or not self.llm:
            raise ValueError("Vector store and LLM must be initialized first")
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}  # Increased from 4 to retrieve more relevant documents
        )
        
        # Create a proper PromptTemplate object
        prompt_template = PromptTemplate(
            template="""You are a helpful assistant that provides detailed answers about medical policies. 
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Provide a comprehensive and detailed response, including specific details, requirements, and any relevant conditions.
            
            Context: {context}
            
            Question: {question}
            
            Detailed Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

    def query(self, question: str) -> Dict:
        """Query the RAG system."""
        if not self.llm or not self.vectorstore:
            raise ValueError("LLM and vector store must be initialized first")
        
        try:
            # Get relevant documents with increased k for better context
            docs = self.vectorstore.similarity_search(
                question,
                k=4  # Increased from 2 to get more context
            )
            
            if not docs:
                return {
                    "answer": "I couldn't find any relevant information in the policy documents to answer your question.",
                    "sources": []
                }
            
            # Create a context-aware prompt that enforces policy-only answers but with better guidance
            context = "\n\n".join([
                f"Document {i+1} (from {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page_number', 'Unknown')}):\n{doc.page_content}" 
                for i, doc in enumerate(docs)
            ])
            
            prompt = PromptTemplate(
                template="""You are a helpful assistant that answers questions about medical policies based on the provided policy documents.
                IMPORTANT RULES:
                1. ONLY answer questions that can be answered using the provided policy documents.
                2. If the question is not about medical policies or cannot be answered using the provided documents, respond with: "I can only answer questions about medical policies based on the provided documents. This question appears to be outside that scope."
                3. For policy-specific questions (like about benefits, coverage, or services), carefully search through ALL provided context to find relevant information.
                4. If you find policy information but it's not exactly matching the question, explain what information is available and where it can be found.
                5. If the documents don't contain enough information to answer the question, say so and mention what information is available.
                6. When discussing specific policy items (like preventive care services), include the policy number or section reference if available.
                7. For questions about specific policies (like Colorado policy), make sure to only use information from that specific policy document.

                Policy Documents:
                {context}

                Question: {question}

                Answer:""",
                input_variables=["context", "question"]
            )
            
            # Use the new RunnablePassthrough pattern instead of LLMChain
            chain = (
                {"context": lambda x: x["context"], "question": lambda x: x["question"]}
                | prompt
                | self.llm
            )
            
            # Generate response
            response = chain.invoke({"context": context, "question": question})
            answer = response.content
            
            # Get sources with more detail
            sources = [f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})" for doc in docs]
            sources = list(set(sources))  # Remove duplicates
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": "I encountered an error while trying to answer your question.",
                "sources": []
            }

    def summarize_policy(self, policy_name: str) -> str:
        """Generate a summary of a specific policy."""
        if not self.llm:
            raise ValueError("LLM not initialized")
            
        # Find relevant documents
        docs = self.vectorstore.similarity_search(
            f"Find information about {policy_name}",
            k=2  # Keep it small for focused summaries
        )
        
        if not docs:
            return "I couldn't find any relevant information to summarize."
        
        # Create a context-aware prompt for summarization
        context = "\n\n".join([f"Section {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        prompt = PromptTemplate(
            template="""You are a helpful assistant that summarizes medical policies based on the provided policy documents.
            IMPORTANT RULES:
            1. ONLY summarize information that is present in the provided policy documents.
            2. Do not add any external knowledge or make assumptions.
            3. Focus on key points, requirements, and conditions.
            4. Provide a clear, structured summary.

            Policy Sections:
            {context}

            Please provide a clear and structured summary of the key points about {policy_name}:""",
            input_variables=["context", "policy_name"]
        )
        
        try:
            # Use the new RunnablePassthrough pattern instead of LLMChain
            chain = (
                {"context": lambda x: x["context"], "policy_name": lambda x: x["policy_name"]}
                | prompt
                | self.llm
            )
            
            # Generate summary
            response = chain.invoke({"context": context, "policy_name": policy_name})
            return response.content
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "I encountered an error while trying to summarize the policy."

    def load_or_create_vector_store(self):
        """Load existing vector store or create new one if it doesn't exist."""
        try:
            # Check if vector store exists and try to load it
            if os.path.exists(self.persist_directory):
                print("Loading existing vector store...")
                try:
                    self.vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings
                    )
                    print("Vector store loaded successfully!")
                    return True
                except Exception as e:
                    print(f"Error loading vector store: {e}")
                    print("Will create new vector store...")
            
            # Create new vector store
            print("Creating new vector store with Gemini embeddings...")
            chunks = self.load_and_chunk_documents()
            self.create_vector_store(chunks)
            print("Vector store created successfully!")
            return False
            
        except Exception as e:
            print(f"Error with vector store: {e}")
            raise RuntimeError(f"Failed to initialize vector store: {e}")

def main():
    # Initialize the RAG system
    rag = PolicyRAG(
        pdf_directory="public_policies - Copy",
        persist_directory="./policy_chroma"
    )
    
    try:
        # Load or create vector store
        rag.load_or_create_vector_store()
        
        # Initialize model
        print("Initializing language model...")
        rag.initialize_llm()
        rag.setup_qa_chain()
        
        # Interactive query loop
        print("\nPolicy RAG System Ready! Available commands:")
        print("- Type 'summarize [policy_name]' to get a summary of a specific policy")
        print("- Type 'exit' to quit")
        print("- Type any other text to ask a question about the policies")
        
        while True:
            user_input = input("\nEnter your query: ").strip()
            
            if user_input.lower() == 'exit':
                break
                
            try:
                if user_input.lower().startswith('summarize '):
                    policy_name = user_input[10:].strip()
                    summary = rag.summarize_policy(policy_name)
                    print("\nSummary:", summary)
                else:
                    result = rag.query(user_input)
                    print("\nAnswer:", result["answer"])
                    print("\nSources:", ", ".join(result["sources"]))
            except Exception as e:
                print(f"Error processing query: {e}")
                
    except Exception as e:
        print(f"Error initializing system: {e}")

if __name__ == "__main__":
    main() 