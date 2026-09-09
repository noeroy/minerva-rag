# MINERvA RAG — Recherche dans la littérature scientifique

Système de RAG (Retrieval-Augmented Generation) permettant d'interroger en langage naturel un corpus de 51 papiers de l'experience neutrino  MINERvA  et un white paper de référence, avec réponses sourcées et vérifiables.

![Démo de l'interface](docs/demo.gif)

---

## Installation et usage

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**1. Récupérer les papiers** (liste dans `minerva_papers_arxiv.csv`) :
```bash
python scrap_articles.py
```

**2. Chunker et indexer** (idempotent — ne retraite que les nouveaux papiers ajoutés au CSV) :
```bash
python chunk_papers.py
```

**3. Configurer la clé API Mistral** :
```bash
export MISTRAL_API_KEY="votre_clé"
```

**4. Lancer l'interface** :
```bash
streamlit run app.py
```

---

## Stack

Python · unstructured · pdfminer.six · tiktoken · langchain-text-splitters · sentence-transformers (bge-small-en-v1.5) · ChromaDB · Mistral AI · Streamlit
