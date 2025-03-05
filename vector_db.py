# Set environment variable before imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", persist_directory="./chroma_db", dry_run=False):
        """Initialize the vector store with the specified embedding model."""
        self.dry_run = dry_run
        logging.debug(f"[__init__] Initializing VectorStore with model: {model_name}, dry_run: {self.dry_run}")
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        self.persist_directory = persist_directory
        
        if not self.dry_run:
            # Try to load existing DB
            if os.path.exists(persist_directory):
                try:
                    self.db = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=self.embedding_function,
                        client_settings=chromadb.config.Settings(
                            anonymized_telemetry=False,
                            is_persistent=True
                        )
                    )
                    logging.info("[__init__] Loaded existing vector store from disk")
                except Exception as e:
                    logging.error(f"[__init__] Error loading existing vector store: {e}")
                    self.db = None
            else:
                self.db = None
        else:
            self.db = None
            logging.info("[__init__] Dry run mode - skipping DB load")

    def initialize_from_pdf(self, pdf_path, chunk_size=300, chunk_overlap=50):
        """Initialize vector DB with PDF content."""
        if self.dry_run:
            logging.info(f"[initialize_from_pdf] Dry run mode - skipping PDF initialization for {pdf_path}")
            return True

        logging.debug(f"[initialize_from_pdf] Loading PDF: {pdf_path}")
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            logging.debug(f"[initialize_from_pdf] Loaded {len(documents)} pages from PDF")

            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
            docs = text_splitter.split_documents(documents)
            logging.debug(f"[initialize_from_pdf] Split into {len(docs)} chunks")

            self.db = Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_function,
                persist_directory=self.persist_directory,
                client_settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            logging.debug("[initialize_from_pdf] Vector store initialized successfully")
            return True

        except Exception as e:
            logging.error(f"[initialize_from_pdf] Error initializing vector store: {e}")
            return False

    def search(self, query, n_results=3):
        """Search the vector store for relevant documents."""
        if self.dry_run:
            logging.info(f"[search] Dry run mode - returning mock results for query: {query}")
            return "Mock result: This is a dry run response."

        logging.debug(f"[search] Searching for: {query}")
        try:
            if not self.db:
                logging.error("[search] Vector store not initialized")
                return ""

            docs = self.db.similarity_search(query, k=n_results)
            context = "\n".join(doc.page_content for doc in docs)
            logging.debug(f"[search] Found {len(docs)} relevant documents")
            return context

        except Exception as e:
            logging.error(f"[search] Error searching vector store: {e}")
            return ""

    def __del__(self):
        """Cleanup when the object is destroyed."""
        logging.debug("[__del__] Vector store cleanup complete")