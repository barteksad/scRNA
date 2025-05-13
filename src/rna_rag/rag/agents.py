import logging
from enum import Enum
from typing import List, Dict, Annotated
from kani import Kani, ai_function, ChatMessage, AIParam

from rna_rag.rag.prompts import SPOKE_AI_SYSTEM_PROMPT
from rna_rag.spoke_api import SpokeAPI



class SpokeAI(Kani):
    """Kani agent to interact with the SPOKE API and process data."""

    def __init__(self, engine, spoke: SpokeAPI = None):
        system_prompt = SPOKE_AI_SYSTEM_PROMPT
        super().__init__(engine, system_prompt=system_prompt)
        self.spoke = SpokeAPI() if spoke is None else spoke

    @ai_function()
    async def get_neighborhood(
            self,
            node_type: Annotated[str, AIParam(desc="Node type to search. One of 'Gene', 'Disease', 'Compound', 'Pathway', 'Reaction', 'CellType', 'MolecularFunction'.")],
            attribute: Annotated[str, AIParam(desc="Attribute to match")],
            value: Annotated[str, AIParam(desc="Attribute value to match. Gene names should be in uppercase, like TP53, BRCA1, CDK2.")],
            node_filters: Annotated[
                List[str], AIParam(desc="Types of nodes to be included in search results. One of 'Gene', 'Disease', 'Compound', 'Pathway', 'Reaction', 'CellType', 'MolecularFunction'.")],
            edge_filters: Annotated[List[str], AIParam(desc="Types of edges to be included in search results. ")],
    ) -> List[Dict]:
        """Retrieve connected nodes/edges around a specific entity in SPOKE."""
        result = self.spoke.get_neighborhood(str(node_type), attribute, value, depth=1, node_filters=node_filters, edge_filters=edge_filters)
        logging.debug(f"Neighborhood data:\n{result}")
        return result

    #TODO: function that trims properties of nodes and edges to only include relevant information from get_neighborhood method
    #
    # @ai_function()
    # async def extract_sources(
    #     self,
    #     raw_data: Annotated[List[Dict], AIParam(desc="Raw JSON output from SPOKE")]
    # ) -> str:
    #     """Extract literature sources from raw SPOKE data (e.g., PubMed IDs, journal references)."""
    #     sources = set()
    #     for item in raw_data:
    #         if "literature" in item.get("properties", {}):
    #             sources.update(item["properties"]["literature"])
    #     return "Relevant sources: " + ", ".join(sources) if sources else "No sources found."


class AnsweringModel(Kani):
    """Kani agent to synthesize answers using processed SPOKE data."""

    @ai_function()
    async def generate_answer(
            self,
            question: Annotated[str, AIParam(desc="Original user question")],
            context: Annotated[str, AIParam(desc="Processed SPOKE data/sources")]
    ) -> str:
        """Generate a final answer using the provided context."""
        # Format the prompt with context
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        response = await self.chat_round(prompt)
        return response.content
