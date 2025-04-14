from typing import List
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import bs4
import pandas as pd


class SingleCellMeta:
    def __init__(self, obs_cols: List[str]):
        self.obs_cols = obs_cols
        # Configure retry strategy
        self.session = requests.Session()
        retries = Retry(
            total=10,  # number of retries
            backoff_factor=2,  # wait 1, 2, 4 seconds between retries
            status_forcelist=[500, 502, 503, 504],  # HTTP status codes to retry on
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def get_metadata(self, obs: pd.Series, var: pd.Series, source_id: str) -> str:
        # meta = "\n".join([f"{col}: {obs[col]}" for col in self.obs_cols])
        meta = "\n".join([f"{col}: {obs[col]}" for col in obs.index])
        extra_meta = self.fetch_additional_metadata(source_id)

        return f"{meta}\n{extra_meta}"

    def fetch_additional_metadata(self, source_id: str) -> str:
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={source_id}",
                    timeout=30,  # Add timeout
                )
                response.raise_for_status()  # Raise an exception for bad status codes
                data = response.text
                break
            except (requests.exceptions.RequestException, ConnectionResetError) as e:
                if attempt == max_retries - 1:  # Last attempt
                    return f"Error fetching metadata: {str(e)}"
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue

        try:
            soup = bs4.BeautifulSoup(data, "html.parser")
            rows = soup.find_all("tr", valign="top")
            extra_meta = []

            for row in rows:
                cells = row.find_all("td")
                if len(cells) == 2:
                    label = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if label and value:
                        if label.strip().lower() == "Submission date".lower():
                            break
                        extra_meta.append(f"{label}: {value}")

            return "\n".join(extra_meta)
        except Exception as e:
            print(f"Error parsing metadata: {str(e)}")
            return "No metadata found"
