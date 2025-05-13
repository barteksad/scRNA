import logging
import time
from typing import List, Dict, Optional

import requests

from rna_rag.utils import batch_list

NCBI_EMAIL = "p.wojciechow@student.uw.edu.pl"


def convert_article_id(source_ids: List[str], batch_size: int = 200, source_id_type='doi', target_id_type='pmcid') -> Dict[str, Optional[str]]:
    """
    Convert multiple DOIs to PubMed IDs in batches

    Args:
        dois: List of DOIs to convert
        batch_size: Maximum number of DOIs to process in a single request (max 200)

    Returns:
        Dictionary mapping DOIs to PMIDs (None if not found)
        :param source_id_type:
        :param target_id_type:
    """
    results = {}


    # Process DOIs in batches
    for doi_batch in batch_list(source_ids, batch_size):
        # Join DOIs with commas for the API request
        doi_string = ','.join(doi_batch)

        # Use the NCBI ID converter API
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?email={NCBI_EMAIL}&ids={doi_string}&format=json"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if 'records' in data:
                # Create a mapping of DOIs to PMIDs
                for record in data['records']:
                    source_id = record.get(source_id_type)
                    target_id = record.get(target_id_type)
                    if source_id:
                        results[source_id] = target_id
        # except Exception as e:
        #     logging.error(f"Error converting batch of DOIs to PMIDs: {e}")

        # Add a small delay to respect API rate limits
        time.sleep(0.5)

    return results
