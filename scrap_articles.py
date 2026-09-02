import json
import arxiv
import pandas as pd
from pathlib import Path
import time
from urllib.request import urlretrieve

df = pd.read_csv("minerva_papers_arxiv.csv", dtype={"arxiv_id": str})
output_dir = Path("papers")
output_dir.mkdir(exist_ok=True)
all_metadata = []
client = arxiv.Client()

for _, row in df.iterrows():
    filepath = output_dir / f"{row['arxiv_id']}.pdf"
    if filepath.exists():
        continue

    try:
        search = arxiv.Search(id_list=[row["arxiv_id"]])
        paper = next(client.results(search))
        urlretrieve(paper.pdf_url, str(filepath))
        print(f"OK: {row['arxiv_id']} - {paper.title}")

        # Bonus : récupère direct les métadonnées propres
        metadata = {
            "id": row["arxiv_id"],
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "published": paper.published.isoformat(),
            "summary": paper.summary,
            "doc_type": row.get("doc_type", "primary_research"),
        }

        all_metadata.append(metadata)

    except Exception as e:
        print(f"ECHEC: {row['arxiv_id']} - {e}")

    time.sleep(3) #avoid dumping the arxiv server too fast


with open("papers_metadata.json", "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)