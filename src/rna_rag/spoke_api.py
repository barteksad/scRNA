from typing import List, Dict

import networkx as nx

import requests


class SpokeAPI:
    """
    A client for interacting with the SPOKE (Scalable Precision Medicine Knowledge Engine) API.

    SPOKE integrates heterogeneous biological data into a graph database, connecting entities like
    genes, compounds, diseases, and more through various relationships.

    Attributes:
        base_url (str): Base URL for the SPOKE API (defaults to production endpoint)

    Note:
        Use `get_types()` to retrieve valid node/edge types and `get_metagraph()` for detailed
        schema information to help construct filters.
    """

    def __init__(self, base_url: str = "https://spoke.rbvi.ucsf.edu/api/v1/"):
        self.base_url = base_url

    def _request(self, method, endpoint_template, path_params=None, query_params=None):
        """Helper method to handle HTTP requests."""
        if path_params is None:
            path_params = {}
        if query_params is None:
            query_params = {}

        endpoint = endpoint_template.format(**path_params)
        url = f"{self.base_url}{endpoint}"

        response = requests.request(method, url, params=query_params)
        response.raise_for_status()
        return response.json()

    def get_metagraph(self) -> list:
        """
        Get the complete SPOKE metagraph structure.

        Returns:
            list: JSGraph object containing all nodes and edges in the SPOKE graph
        """
        return self._request("GET", "metagraph")

    def get_types(self) -> list:
        """
        Get valid node/edge types and default queries.

        Returns:
            list: Array of arrays containing [node_type, edge_type, default_query]
        """
        return self._request("GET", "types")

    def get_node(self, node_type: str, attribute: str, value: str, **query_params) -> list:
        """
        Retrieve details for a specific node.

        Args:
            node_type: Valid node type (e.g., 'Gene', 'Compound') from get_types()
            attribute: Node property to match (e.g., 'name', 'identifier')
            value: Value of the specified property to match
            **query_params: Additional query parameters

        Returns:
            list: Array of node objects matching the criteria
        """
        return self._request(
            "GET",
            "node/{node_type}/{attribute}/{value}",
            {"node_type": node_type, "attribute": attribute, "value": value},
            query_params,
        )

    def get_neighborhood(
        self,
        node_type: str,
        attribute: str,
        value: str,
        node_filters: list = None,
        edge_filters: list = None,
        depth: int = None,
        cutoff_Compound_max_phase: float = None,
        cutoff_Protein_source: list = None,
        cutoff_DaG_diseases_sources: list = None,
        cutoff_DaG_textmining: float = None,
        cutoff_CtD_phase: float = None,
        cutoff_PiP_confidence: float = None,
        cutoff_ACTeG_level: list = None,
        **query_params
    ) -> list:
        """
        Retrieve neighborhood graph around a node with filtering options.

        Args:
            node_type: Valid node type from get_types()
            attribute: Node property to match
            value: Property value to match
            node_filters: List of node types to include (e.g., ['Gene', 'Compound'])
            edge_filters: List of edge types to include (e.g., ['INTERACTS', 'TREATS'])
            depth: Expansion depth (default=1)
            cutoff_Compound_max_phase: Max clinical phase for compounds (0-4)
            cutoff_Protein_source: Protein sources ['SwissProt', 'TrEMBL']
            cutoff_DaG_diseases_sources: Disease sources ['knowledge', 'experiments', 'textmining']
            cutoff_DaG_textmining: Min textmining z-score (default=3)
            cutoff_CtD_phase: Compound-Disease phase cutoff
            cutoff_PiP_confidence: Protein interaction confidence (0-1, default=0.7)
            cutoff_ACTeG_level: Protein expression levels ['Not detected', 'Low', 'Medium', 'High']
            **query_params: Additional parameters

        Returns:
            list: JSGraph object containing neighborhood nodes and edges
        """
        params = {
            "node_filters": node_filters,
            "edge_filters": edge_filters,
            "depth": depth,
            "cutoff_Compound_max_phase": cutoff_Compound_max_phase,
            "cutoff_Protein_source": cutoff_Protein_source,
            "cutoff_DaG_diseases_sources": cutoff_DaG_diseases_sources,
            "cutoff_DaG_textmining": cutoff_DaG_textmining,
            "cutoff_CtD_phase": cutoff_CtD_phase,
            "cutoff_PiP_confidence": cutoff_PiP_confidence,
            "cutoff_ACTeG_level": cutoff_ACTeG_level,
            **query_params
        }
        return self._request(
            "GET",
            "neighborhood/{node_type}/{attribute}/{value}",
            {"node_type": node_type, "attribute": attribute, "value": value},
            {k: v for k, v in params.items() if v is not None},
        )

    def get_version(self) -> dict:
        """
        Get version information and database timestamps.

        Returns:
            dict: Version metadata including:
                - version (str): SPOKE version
                - snapshot_timestamp (str): Database snapshot timestamp
                - timestamps (dict): Individual component timestamps
        """
        return self._request("GET", "version")

    def expand_node(
        self,
        node_type: str,
        node_id: int,
        node_ids: list = None,
        node_filters: list = None,
        edge_filters: list = None,
        depth: int = None,
        **cutoff_params
    ) -> list:
        """
        Expand a node by its internal ID with context from existing nodes.

        Args:
            node_type: Valid node type from get_types()
            node_id: Internal node ID to expand
            node_ids: List of existing node IDs in the network (for context-aware expansion)
            node_filters: List of node types to include
            edge_filters: List of edge types to include
            depth: Expansion depth (default=1)
            **cutoff_params: Filtering parameters (see get_neighborhood docs)

        Returns:
            list: JSGraph object with expanded nodes/edges
        """
        params = {
            "node_ids": node_ids,
            "node_filters": node_filters,
            "edge_filters": edge_filters,
            "depth": depth,
            **cutoff_params
        }
        return self._request(
            "POST",
            "expand/{node_type}/{node_id}",
            {"node_type": node_type, "node_id": node_id},
            {k: v for k, v in params.items() if v is not None},
        )

    def sea_search(self, smiles_or_zinc: str, **query_params) -> list:
        """
        Perform a SEA (Similarity Ensemble Approach) search.

        Args:
            smiles_or_zinc: SMILES string or ZINC ID (must start with 'zinc')
            **query_params: Filtering parameters (see get_neighborhood docs)

        Returns:
            list: JSGraph object with related compounds and targets
        """
        return self._request(
            "GET",
            "sea/{smiles_or_zinc}",
            {"smiles_or_zinc": smiles_or_zinc},
            query_params,
        )

    def search_by_type(self, node_type: str, query: str, **query_params) -> list:
        """
        Search nodes of a specific type using Lucene query syntax.

        Args:
            node_type: Valid node type from get_types()
            query: Lucene query string (e.g., 'name:IL7R', 'identifier:7157')
            **query_params: Additional search parameters

        Returns:
            list: Search results with scores
        """
        return self._request(
            "GET",
            "search/{node_type}/{query}",
            {"node_type": node_type, "query": query},
            query_params,
        )

    def search_global(self, query: str, **query_params) -> list:
        """
        Global search across all node types using Lucene query syntax.

        Args:
            query: Lucene query string (e.g., 'cancer', 'name:aspirin')
            **query_params: Additional search parameters

        Returns:
            list: Search results with scores across all node types
        """
        return self._request(
            "GET",
            "search/{query}",
            {"query": query},
            query_params,
        )

    def get_gene(self, gene_symbol: str, attribute: str = "name", **query_params) -> list:
        """
        Helper to fetch gene information by symbol.

        Args:
            gene_symbol: Gene symbol (e.g., 'BRCA1', 'IL7R')
            attribute: Property to match (default='name')

        Returns:
            list: Gene node information
        """
        return self.get_node("Gene", attribute, gene_symbol, **query_params)

    def get_drug(self, drug_name: str, attribute: str = "name", **query_params) -> list:
        """
        Helper to fetch drug information by name.

        Args:
            drug_name: Drug name (e.g., 'aspirin', 'imatinib')
            attribute: Property to match (default='name')

        Returns:
            list: Compound node information
        """
        return self.get_node("Compound", attribute, drug_name, **query_params)



def create_networkx_graph(spoke_api_graph_response: List[Dict]) -> nx.Graph:
    G = nx.Graph()

    for item in spoke_api_graph_response:
        if "source" in item["data"]:  # It's an edge
            edge_data = item["data"]
            G.add_edge(edge_data["source"], edge_data["target"], **edge_data["properties"])
        else:  # It's a node
            node_data = item["data"]
            G.add_node(node_data["id"], **node_data["properties"])

    return G
