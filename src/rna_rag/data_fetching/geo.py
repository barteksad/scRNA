import sys
from Bio import Entrez
import time
from typing import List, Dict, Optional

Entrez.email = "p.wojciechow@student.uw.edu.pl"


def get_geo_entries_with_articles(search_query: str, batch_size: int = 100, max_entries: Optional[int] = None) -> List[Dict]:
    """
    Fetch GEO entries that have associated articles, processing in batches.
    
    Args:
        search_query: Search query for GEO entries
        batch_size: Number of entries to process in each batch (default: 100)
        max_entries: Maximum number of entries to process (default: None, process all entries)
        
    Returns:
        List of dictionaries containing GEO entry information
    """
    Entrez.email = "p.wojciechow@student.uw.edu.pl"
    
    # First, get the total count of entries
    handle = Entrez.esearch(db="gds", term=search_query, retmax=0)
    search_results = Entrez.read(handle)
    handle.close()
    
    total_count = int(search_results["Count"])
    if max_entries is not None:
        total_count = min(total_count, max_entries)
        
    all_entries = []
    
    # Process in batches
    for start in range(0, total_count, batch_size):
        try:
            # Adjust batch_size for the last batch if max_entries is set
            current_batch_size = min(
                batch_size,
                total_count - start if max_entries is not None else batch_size
            )
            
            # Get batch of IDs
            handle = Entrez.esearch(
                db="gds",
                term=search_query,
                retstart=start,
                retmax=current_batch_size
            )
            search_results = Entrez.read(handle)
            handle.close()
            
            if not search_results["IdList"]:
                break
                
            # Get details for the batch
            handle = Entrez.esummary(db="gds", id=",".join(search_results["IdList"]))
            records = Entrez.read(handle)
            handle.close()
            
            # Process each record
            for record in records:
                if not record.get('PubMedIds'):
                    continue
                    
                entry = {
                    'Accession': record['Accession'],
                    'Title': record['title'],
                    'Summary': record['summary'],
                    'URL': f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={record['Accession']}",
                    'Taxon': record['taxon'],
                    'Platform': record.get('GPL', ''),
                    'Modification_Date': record['PDAT'],
                    'PMIDs': record['PubMedIds']
                }
                all_entries.append(entry)
                
                # Check if we've reached max_entries
                if max_entries is not None and len(all_entries) >= max_entries:
                    return all_entries
            
            # Respect NCBI's rate limit (maximum 3 requests per second)
            time.sleep(0.34)
            
        except Exception as e:
            print(f"Error processing batch starting at {start}: {e}")
            continue
            
    return all_entries