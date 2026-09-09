import json
import time
from pathlib import Path
from urllib.request import urlretrieve

import arxiv
import pandas as pd

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
            # doc_type vient directement du CSV : "primary_research" (mesure MINERvA)
            # ou "background" (document de référence générale, ex: white paper).
            # Se propage automatiquement jusqu'au prompt final du RAG.
            "doc_type": row.get("doc_type", "primary_research"),
        }

        all_metadata.append(metadata)

        # tu peux stocker ça dans un JSON/CSV à part pour la Phase 3

    except Exception as e:  # noqa: BLE001 -- volontaire : on veut continuer sur les autres papiers même en cas d'erreur inattendue (réseau, PDF corrompu, timeout arXiv...)
        print(f"ECHEC: {row['arxiv_id']} - {e}")

    time.sleep(3)


with open("papers_metadata.json", "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)