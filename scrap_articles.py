import json
import time
from pathlib import Path
from urllib.request import urlretrieve

import arxiv
import pandas as pd

df = pd.read_csv("minerva_papers_arxiv.csv", dtype={"arxiv_id": str})
output_dir = Path("papers")
output_dir.mkdir(exist_ok=True)
metadata_path = Path("papers_metadata.json")

# Charge les métadonnées déjà présentes, pour ne pas les écraser quand on
# relance le script sur un CSV qui a des papiers déjà téléchargés.
if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)
else:
    all_metadata = []

already_have_metadata = {m["id"] for m in all_metadata}

client = arxiv.Client()

for _, row in df.iterrows():
    filepath = output_dir / f"{row['arxiv_id']}.pdf"

    if row["arxiv_id"] in already_have_metadata:
        continue  # déjà téléchargé ET déjà dans les métadonnées, rien à faire

    if filepath.exists():
        # Le PDF existe mais ses métadonnées manquent (ex: fichier
        # récupéré autrement, ou papers_metadata.json perdu) -- on
        # re-fetch juste les métadonnées sans re-télécharger le PDF.
        pass
    else:
        try:
            search = arxiv.Search(id_list=[row["arxiv_id"]])
            paper = next(client.results(search))
            urlretrieve(paper.pdf_url, str(filepath))
            print(f"OK: {row['arxiv_id']} - {paper.title}")
        except Exception as e:  # noqa: BLE001 -- volontaire : on veut continuer sur les autres papiers même en cas d'erreur inattendue (réseau, PDF corrompu, timeout arXiv...)
            print(f"ECHEC téléchargement: {row['arxiv_id']} - {e}")
            continue

    try:
        search = arxiv.Search(id_list=[row["arxiv_id"]])
        paper = next(client.results(search))

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

    except Exception as e:  # noqa: BLE001 -- volontaire : idem, continuer sur les autres papiers
        print(f"ECHEC métadonnées: {row['arxiv_id']} - {e}")

    time.sleep(3)


with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

print(f"\nTotal: {len(all_metadata)} entrées dans papers_metadata.json")