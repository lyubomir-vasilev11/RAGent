import time
from vector_db import VectorStore
from utils import call_rag, call_tavily_web_search, assess_confidence, synthesize_information
import logging

def main(user_query, initialize=False, pdf_path=None, dry_run=False):
    if initialize:
        logging.debug("[main] Initialization mode.")
    elif user_query is not None:
        logging.debug(f"[main] Starting process with query: {user_query[:100]}...")
    else:
        logging.debug("[main] No user query provided.")

    # Continue with initialization branch
    vector_store = VectorStore(persist_directory="./chroma_db")
    
    if initialize and pdf_path:
        logging.info(f"[main] Initializing with PDF: {pdf_path}")
        vector_store.initialize_from_pdf(pdf_path)
        return "PDF processed and ready for questions!"
    elif not vector_store.db and not initialize:
        logging.error("[main] No vector store found")
        return "Error: Vector store not initialized. Please provide a PDF file first."
    elif initialize:
        logging.error("[main] PDF path not provided for initialization")
        return "Please provide a PDF file to initialize the system."
    
    # Process the user query if provided
    internal_response = vector_store.search(user_query)
    
    logging.debug("[main] Getting RAG response")
    rag_response = call_rag(user_query, internal_response, dry_run)
    
    if isinstance(rag_response, list):
        rag_response = " ".join(str(item) for item in rag_response)
    
    logging.debug("[main] Assessing confidence")
    if assess_confidence(rag_response, dry_run):
        logging.info("[main] Returning high-confidence response")
        return rag_response
    
    logging.debug("[main] Confidence low, performing web search")
    web_results = call_tavily_web_search(user_query, dry_run)
    time.sleep(1)
    
    logging.debug("[main] Synthesizing information")
    synthesized_response = synthesize_information(rag_response, web_results, dry_run)
    
    logging.info("[main] Returning synthesized response")
    return synthesized_response
