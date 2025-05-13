import argparse
import os
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import time
from Bio import Entrez
import requests

from markitdown import MarkItDown
from tqdm import tqdm

from rna_rag.data_fetching.geo import get_geo_entries_with_articles
from rna_rag.data_fetching.cellxgene import get_cellxgene_datasets
from rna_rag.data_fetching.database import RNADatabase, Article, ArticleDownload
from rna_rag.aws import download_pmc_files_by_id_list
from rna_rag.data_fetching.utils import convert_article_id

# TODO: add filtering to Cell x Gene entries
# TODO: investigate cell x gene access to full text articles

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Entrez.email = "p.wojciechow@student.uw.edu.pl"

# Cell x Gene filters - hardcoded constants
CELLXGENE_ORGANISM = "mus_musculus"
CELLXGENE_ASSAYS = [
    "10x 3' transcription profiling",
    "10x 3' v2",
    "10x 3' v3",
    "10x 5' transcription profiling",
    "10x 5' v1",
    "10x 5' v2",
    "10x gene expression flex"
]

GEO_QUERY = '(scRNA-seq[All Fields] OR sc-RNA-seq[All Fields] OR single-cell RNA-seq[All Fields] OR single-cell RNA-sequencing[All Fields]) AND "Mus musculus"[porgn]'


def download_articles(articles: List[Article], download_dir: str) -> List[ArticleDownload]:
    """
    1. Try to get PMC IDs for all articles if possible.
    2. For found PMC IDs, download TXT files. If not found, process such files as in point 3.
    3. For leftover articles, use paperscraper to download XML files based on DOIs.
    """

    Path(download_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Identify articles without PMIDs
    articles_without_pmid = [article for article in articles if not article.pmcid]
    dois_to_convert = [article.doi for article in articles_without_pmid]

    # Step 2: Convert DOIs to PMIDs in batches
    pmid_map = convert_article_id(dois_to_convert, source_id_type='doi', target_id_type='pmcid')
    for article in articles_without_pmid:
        if article.doi in pmid_map:
            article.pmcid = pmid_map[article.doi]

    # Step 3: Separate articles with PMIDs and without PMIDs
    articles_with_pmid = [article for article in articles if article.pmcid]
    articles_without_pmid = [article for article in articles if not article.pmcid]

    # Step 4: Download articles with PMIDs using PMC
    success_downloads_pmc, failed_to_download_pmc = download_articles_from_pmc(articles_with_pmid, download_dir)

    success_downloads_doi, failed_to_download_doi = download_articles_by_doi(
        articles_without_pmid + failed_to_download_pmc, download_dir)

    all_downloaded_articles = success_downloads_pmc + success_downloads_doi

    assert len(articles) == len(all_downloaded_articles) + len(failed_to_download_doi)
    return all_downloaded_articles


def download_articles_from_pmc(articles: List[Article], articles_dir: str) -> Tuple[
    List[ArticleDownload], List[Article]]:
    """Download articles for given PMC IDs."""

    pmcids = [article.pmcid for article in articles]
    download_result = download_pmc_files_by_id_list(pmcids, articles_dir, file_format='txt')

    assert len(download_result) == len(pmcids)

    downloaded_articles = []
    articles_failed_to_download = []

    for article, download_result in zip(articles, download_result):
        if download_result is None:
            articles_failed_to_download.append(article)
        else:
            # Update article with download result
            downloaded_articles.append(ArticleDownload.from_article(article, str(download_result)))

    return downloaded_articles, articles_failed_to_download


def download_articles_by_doi(articles: List[Article], articles_dir: str) -> Tuple[List[ArticleDownload], List[Article]]:

    #TODO: downloading from biorxiv is not possible, but soon it will be: https://github.com/jannisborn/paperscraper/pull/80
    # download_results = []
    #
    # for article in tqdm(articles):
    #     pdf_url = query_unpaywall_for_link(article.doi)
    #     if pdf_url is None:
    #         download_results.append(None)
    #         continue
    #
    #     response = requests.get(pdf_url)
    #     if response.status_code == 200:
    #         filepath = os.path.join(articles_dir, f"{article.doi.replace('/', '_')}.pdf")
    #         with open(filepath, 'wb') as f:
    #             f.write(response.content)
    #
    #         md = MarkItDown(enable_plugins=False)  # Set to True to enable plugins
    #         result = md.convert(filepath)
    #
    #         # save result.text into txt file
    #         txt_filepath = os.path.join(articles_dir, f"{article.doi.replace('/', '_')}.txt")
    #         with open(txt_filepath, 'w') as f:
    #             f.write(result.text_content)
    #
    #         download_results.append(txt_filepath)
    #
    #         # remove pdf file
    #         os.remove(filepath)
    #
    #     else:
    #         print(f"Failed to download file for doi {article.doi}. Status code: {response.status_code}")
    #         download_results.append(None)
    #
    # assert len(download_results) == len(articles)
    #
    # downloaded_articles = []
    # articles_failed_to_download = []
    #
    # for article, download_result in zip(articles, download_results):
    #     if download_result is None:
    #         articles_failed_to_download.append(article)
    #     else:
    #         # Update article with download result
    #         downloaded_articles.append(ArticleDownload.from_article(article, str(download_result)))



    # return downloaded_articles, articles_failed_to_download
    return [], articles


def query_unpaywall_for_link(doi: str) -> str | None:
    email = 'p.wojciechow@student.uw.edu.pl'
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"

    response = requests.get(url)
    data = response.json()
    if data.get("is_oa"):
        return data["best_oa_location"]["url_for_pdf"]
    else:
        return None

def get_articles_from_geo(batch_size: int = 100, limit: Optional[int] = None) -> List[Article]:
    entries = get_geo_entries_with_articles(GEO_QUERY, batch_size=batch_size, max_entries=limit)
    pmids = [str(int(entry['PMIDs'][0])) for entry in entries]

    dois_map = convert_article_id(pmids, source_id_type='pmid', target_id_type='doi')
    dois = [dois_map.get(pmid) for pmid in pmids]

    pmcids_map = convert_article_id(pmids, source_id_type='pmid', target_id_type='pmcid')
    pmcids = [pmcids_map.get(pmid) for pmid in pmids]

    articles = []

    dropped_entries = []

    for entry, pmid, doi, pmcid in zip(entries, pmids, dois, pmcids):
        if doi is None:
            dropped_entries.append(entry)
            continue
        articles.append(Article(
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            source_repository='geo',
            source_dataset={'title': entry['Accession'],
                            'url': entry['URL']},
        ))

    # TODO: fix id converted dependence on PMC, see the notes.md for more details how to fix it
    logging.info(f"Dropped {len(dropped_entries)} entries from GEO because of lacking DOI.")
    return articles


def get_articles_from_cellxgene() -> List[Article]:
    datasets = get_cellxgene_datasets()
    datasets = [dataset for dataset in datasets if dataset.get('DOI')]

    articles = []
    for dataset in datasets:
        articles.append(Article(
            doi=dataset['DOI'],
            pmcid=None,
            source_repository='cellxgene',
            source_dataset={'title': dataset['Accession'],
                            'url': dataset['URL']},
        ))

    return articles


def main(db_path: str, articles_dir: str,
         update_geo_data: bool = False,
         geo_batch_size: int = 10000, geo_max_entries: Optional[int] = None,
         update_cellxgene_data: bool = False):
    """
    Update RNA dataset articles from various sources
    
    This script can update datasets from multiple sources (GEO, Cell x Gene) and download
    their associated full-text articles. Use the appropriate flags to specify which sources
    to update.
    
    Examples:
        # Update only GEO datasets
        python -m rna_rag.data_fetching.update_rna_articles --db_path db.sqlite --articles_dir articles --update_geo --geo_search_query "RNA-seq"
        
        # Update only Cell x Gene datasets
        python -m rna_rag.data_fetching.update_rna_articles --db_path db.sqlite --articles_dir articles --update_cellxgene
        
        # Update both sources
        python -m rna_rag.data_fetching.update_rna_articles --db_path db.sqlite --articles_dir articles --update_geo --geo_search_query "RNA-seq" --update_cellxgene
    """
    # Initialize database
    db = RNADatabase(db_path, articles_dir)

    Path(articles_dir).mkdir(parents=True, exist_ok=True)

    articles = []

    if update_geo_data:
        geo_articles = get_articles_from_geo(geo_batch_size, geo_max_entries)
        articles += get_articles_from_geo(geo_batch_size, geo_max_entries)
        logging.info(f"Found {len(geo_articles)} GEO articles")
    if update_cellxgene_data:
        cellxgene_articles = get_articles_from_cellxgene()
        logging.info(f"Found {len(cellxgene_articles)} Cell x Gene articles")
        articles += cellxgene_articles

    logging.info(f"Found {len(articles)} total articles potentially to be downloaded.")

    downloaded_articles = download_articles(articles, articles_dir)

    db.upsert_downloaded_articles(downloaded_articles)
    logging.info(f"Downloaded {len(downloaded_articles)} articles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update RNA dataset articles from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update only GEO datasets
  python -m rna_rag.data_fetching.update_rna_articles --db_path db.sqlite --articles_dir articles --update_geo --geo_search_query "RNA-seq"
  
  # Update only Cell x Gene datasets
  python -m rna_rag.data_fetching.update_rna_articles --db_path db.sqlite --articles_dir articles --update_cellxgene
  
  # Update both sources
  python -m rna_rag.data_fetching.update_rna_articles --db_path db.sqlite --articles_dir articles --update_geo --geo_search_query "RNA-seq" --update_cellxgene
        """
    )

    parser.add_argument("--db_path", type=str, required=True,
                        help="Path to SQLite database file")
    parser.add_argument("--articles_dir", type=str, required=True,
                        help="Directory to store downloaded articles")

    # GEO-specific arguments
    parser.add_argument("--update_geo", action="store_true",
                        help="Update GEO datasets"),
    parser.add_argument("--geo_batch_size", type=int, default=10000,
                        help="Number of GEO entries to process in each batch")
    parser.add_argument("--geo_max_entries", type=int, default=None,
                        help="Maximum number of GEO entries to process")

    # Cell x Gene-specific arguments
    parser.add_argument("--update_cellxgene", action="store_true",
                        help="Update Cell x Gene datasets")

    args = parser.parse_args()


    if not args.update_geo and not args.update_cellxgene:
        parser.error("At least one of --update_geo or --update_cellxgene must be specified")

    main(
        db_path=args.db_path,
        articles_dir=args.articles_dir,
        update_geo_data=args.update_geo,
        geo_batch_size=args.geo_batch_size,
        geo_max_entries=args.geo_max_entries,
        update_cellxgene_data=args.update_cellxgene
    )
