from kani.engines.openai import OpenAIEngine
from litellm import completion
import asyncio
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


from rna_rag.rag.agents import SpokeAI
from rna_rag.rag.text_search import QdrantVectorDatabase
from rna_rag.rag.prompts import INPUT_FORMAT_FOR_SPOKE_AI, INPUT_FORMAT_FOR_VECTOR_DB, MERGE_CONTEXTS_FORMAT, RAG_TEMPLATE


async def get_spoke_context(question: str, data: str, agent: SpokeAI) -> str:
    logger.info("Starting SPOKE knowledge graph retrieval")
    
    formatted_input = INPUT_FORMAT_FOR_SPOKE_AI.format(question=question, data=data)
    logger.debug(f"Formatted input for SPOKE: {formatted_input}")

    messages = []
    async for msg in agent.full_round(formatted_input):
        messages.append(msg)

    last_message = messages[-1]
    logger.info("SPOKE knowledge graph retrieval completed")
    logger.debug(f"SPOKE response: {last_message.text}")
    return last_message.text


def get_literature_context(question: str, data: str, qdrant_database: QdrantVectorDatabase, k:int = None) -> str:
    """Get context from literature using vector database."""
    # Use environment variable if k is not specified
    k = k or int(os.getenv("RAG_RESULTS_COUNT", 5))
    
    logger.info("Starting literature retrieval from vector database")
    logger.debug(f"Number of results to retrieve: {k}")
    
    formatted_input = INPUT_FORMAT_FOR_VECTOR_DB.format(question=question, data=data)
    logger.debug(f"Formatted input for vector search: {formatted_input}")
    
    results = qdrant_database.hybrid_search(formatted_input, k=k)
    context = _format_qdrant_results(results)
    
    logger.info("Literature retrieval completed")
    return context

async def pipeline(question: str, data: str, answering_model: str = None) -> str:
    """
    Main RAG pipeline for scRNA-seq data analysis.
    
    Args:
        question: User's question about their scRNA-seq data
        data: Description of the user's scRNA-seq data sample
        answering_model: LLM model to use for final answer generation
    
    Returns:
        Comprehensive answer combining knowledge graph and literature evidence
    """
    logger.info("Starting RAG pipeline")
    
    # Use default model from env if not specified
    answering_model = answering_model or os.getenv("DEFAULT_ANSWERING_MODEL")
    logger.info(f"Answering model: {answering_model}")

    # Initialize components
    logger.info("Initializing components")
    spoke_model = os.getenv("SPOKE_MODEL")
    agent = SpokeAI(engine=OpenAIEngine(model=spoke_model))
    qdrant_database = QdrantVectorDatabase.load(os.getenv("QDRANT_DATABASE_PATH"))

    # Retrieve context from both sources
    logger.info("Retrieving context from vector database")
    rag_results_count = int(os.getenv("RAG_RESULTS_COUNT", 5))
    vector_context = get_literature_context(question, data, qdrant_database, k=rag_results_count)
    logger.info("Retrieving context from SPOKE knowledge graph")
    spoke_context = await get_spoke_context(question, data, agent)
    
    # Combine contexts for final answer generation
    logger.info("Combining contexts for final answer generation")
    final_input = MERGE_CONTEXTS_FORMAT.format(
        question=question, 
        data=data, 
        vector_context=vector_context, 
        spoke_context=spoke_context
    )
    logger.debug(f"Final input for answer generation: {final_input}")

    # Generate the final answer using the answering model
    logger.info("Generating final answer")
    temperature = float(os.getenv("RAG_TEMPERATURE", 0.2))
    max_tokens = int(os.getenv("RAG_MAX_TOKENS", 1500))
    
    answer = completion(
        model=answering_model,
        messages=[{"role": "user", "content": final_input}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    final_answer = answer['choices'][0]['message']['content']
    logger.info("Pipeline completed successfully")
    logger.debug(f"Final answer: {final_answer}")
    
    return final_answer


def _format_qdrant_results(results):
    formatted_results = []
    for point in results.points:
        payload = point.payload
        source_document = payload["file_name"]
        content = payload["document"]
        formatted_results.append(f"Source: {source_document}\n\nContent:\n{content}")
    return "\n\n---\n\n".join(formatted_results)


if __name__ == "__main__":
    from dotenv import load_dotenv

    # set logging to debug
    logging.basicConfig(level=logging.DEBUG)

    load_dotenv()
    
    # Example usage
    question = "What pathways may be active with such gene expressions in these cells?"
    data = "3-month-old female mouse bone marrow granulocytopoietic cells, appearing normal. Showing expression of genes like Rab20, Rpl38, and Usp9y."
    
    # Use asyncio.run() only in the main block
    answer = asyncio.run(pipeline(question, data))
    print("\nFinal Answer:")
    print("=" * 80)
    print(answer)

