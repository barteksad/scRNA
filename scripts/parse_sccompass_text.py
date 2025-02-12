import bs4
import requests

data = requests.get('https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5743383').text

soup = bs4.BeautifulSoup(data, 'html.parser', from_encoding='utf-8')

# Find all table rows
rows = soup.find_all('tr', valign="top")

# Extract label-value pairs
for row in rows:
    # Find the label (first td) and value (second td)
    cells = row.find_all('td')
    if len(cells) == 2:
        label = cells[0].get_text(strip=True)
        value = cells[1].get_text(strip=True)
        if label and value:
            print(f"{label}: {value}")