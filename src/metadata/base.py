from typing import List

import bs4
import pandas as pd
import requests


class SingleCellMeta:
    def __init__(self, obs_cols: List[str]):
        self.obs_cols = obs_cols

    def get_metadata(self, obs: pd.Series, var: pd.Series, source_id: str) -> str:
        # meta = "\n".join([f"{col}: {obs[col]}" for col in self.obs_cols])
        meta = "\n".join([f"{col}: {obs[col]}" for col in obs.index])
        extra_meta = self.fetch_additional_metadata(source_id)

        return f"{meta}\n{extra_meta}"

    def fetch_additional_metadata(self, source_id: str) -> str:
        data = requests.get(
            f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={source_id}"
        ).text
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
