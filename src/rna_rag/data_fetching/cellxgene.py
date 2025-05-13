import cellxgene_census
from typing import List, Dict
import logging

from rna_rag.data_fetching.utils import convert_article_id


def get_cellxgene_datasets() -> List[Dict]:
    """
    Fetch all Cell x Gene datasets
    
    Returns:
        List of dictionaries containing dataset information
    """
    logging.info("Fetching all Cell x Gene datasets")
    
    try:
        # Open Census data
        census = cellxgene_census.open_soma(census_version="latest")
        
        # Get dataset information
        datasets_df = census["census_info"]["datasets"].read().concat().to_pandas()
        
        # Convert to desired format
        results = []
        for _, row in datasets_df.iterrows():
            dataset = {
                'Accession': row['dataset_id'],
                'Title': row['dataset_title'],
                'URL': row.get('collection_url'),
                'Taxon': row.get('organism'),
                'Summary': row.get('tissue'),
                'Platform': row.get('assay'),
                'Modification_Date': row.get('dataset_version_date'),
                'DOI': row.get('collection_doi'),
                'Citation': row.get('citation'),
                'Source': 'cellxgene'
            }
            results.append(dataset)
        
        census.close()
        logging.info(f"Found {len(results)} Cell x Gene datasets")
        return results
        
    except Exception as e:
        logging.error(f"Error fetching Cell x Gene datasets: {e}")
        return []


def get_cellxgene_entries_with_articles() -> List[Dict]:
    """
    Fetch all Cell x Gene datasets and convert DOIs to PMIDs
    
    Returns:
        List of dictionaries containing dataset information with PMIDs
    """
    datasets = get_cellxgene_datasets()
    
    # Extract all DOIs
    dois = [dataset.get('DOI', '') for dataset in datasets]
    
    # Convert DOIs to PMIDs in batches
    doi_to_pmid_map = convert_article_id(dois)
    
    # Update datasets with PMIDs
    for dataset in datasets:
        doi = dataset.get('DOI', '')
        if doi and doi in doi_to_pmid_map:
            pmcid = doi_to_pmid_map[doi]
            if pmcid:
                dataset['PMIDs'] = [pmcid]
            else:
                dataset['PMIDs'] = []
        else:
            dataset['PMIDs'] = []
    
    # Return only datasets with associated PMIDs
    return [dataset for dataset in datasets if dataset['PMIDs']]