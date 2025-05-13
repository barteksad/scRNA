import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Literal
from pathlib import Path
import json
import logging
from pydantic import BaseModel, HttpUrl


class SourceDataset(BaseModel):
    title: str
    url: Optional[HttpUrl] = None


class Article(BaseModel):
    doi: str
    pmid: Optional[str] = None
    pmcid: Optional[str] = None  # this one is used to download full text from pubmed central
    source_repository: Literal['geo', 'cellxgene']
    source_dataset: SourceDataset


class ArticleDownload(Article):
    filename: str

    @classmethod
    def from_article(cls, article: Article, filename: str) -> 'ArticleDownload':
        """Create an ArticleDownload instance from an Article instance"""
        return cls(
            doi=article.doi,
            pmcid=article.pmcid,
            source_repository=article.source_repository,
            source_dataset=article.source_dataset,
            filename=filename
        )


class RNADatabase:
    def __init__(self, db_path: str, articles_dir: str):
        self.db_path = db_path
        self.articles_dir = str

        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        Path(articles_dir).mkdir(parents=True, exist_ok=True)

        if not db_file.exists():
            logging.info(f"Creating new database at {db_path}")
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS downloaded_articles (
                    doi TEXT PRIMARY KEY,
                    pmcid TEXT,
                    url TEXT,
                    source_repository TEXT,
                    source_dataset_title TEXT,
                    source_dataset_url TEXT,
                    last_updated TIMESTAMP
                )
            ''')

            conn.commit()

    def upsert_downloaded_articles(self, articles: List[ArticleDownload]):
        """Insert or update downloaded articles"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for article in articles:
                cursor.execute('''
                    INSERT OR REPLACE INTO downloaded_articles 
                    (doi, pmcid, url, source_repository, source_dataset_title, source_dataset_url, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.doi,
                    str(article.pmcid) if article.pmcid is not None else None,
                    str(article.source_dataset.url) if article.source_dataset.url is not None else None,
                    article.source_repository,
                    article.source_dataset.title,
                    str(article.source_dataset.url) if article.source_dataset.url is not None else None,
                    datetime.now().isoformat()
                ))
            conn.commit()
            logging.info(f"Inserted/updated {len(articles)} articles in the database")

    def filter_already_downloaded_articles(self, articles: List[Article]):
        """Filter out articles that are already downloaded"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            downloaded_dois = {row[0] for row in cursor.execute('SELECT doi FROM downloaded_articles')}

        # Filter out articles that are already downloaded
        return [article for article in articles if article.doi not in downloaded_dois]

    def check_database_integrity(self, path_to_files: str, delete_missing_entries: bool = False) -> List[
        ArticleDownload]:
        """Checks if downloaded files exist. Returns a list of missing files."""
        missing_files = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT doi, pmcid, url, source_repository, source_dataset_title, source_dataset_url FROM downloaded_articles')
            rows = cursor.fetchall()

            for row in rows:
                doi, pmcid, url, source_repository, source_dataset_title, source_dataset_url = row
                file_path = Path(path_to_files) / f"{doi}.pdf"

                if not file_path.exists():
                    missing_files.append(ArticleDownload(
                        doi=doi,
                        pmcid=pmcid,
                        filename=str(file_path),
                        url=url,
                        source_repository=source_repository,
                        source_dataset=SourceDataset(title=source_dataset_title, url=source_dataset_url)
                    ))
                    if delete_missing_entries:
                        cursor.execute('DELETE FROM downloaded_articles WHERE doi = ?', (doi,))

            conn.commit()

        return missing_files
