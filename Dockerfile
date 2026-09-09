FROM python:3.12-slim

WORKDIR /app

# Dépendances système nécessaires à unstructured/pdfminer (pas utilisées au
# runtime de l'app Streamlit elle-même, seulement si on veut aussi lancer
# le pipeline d'indexation dans le conteneur -- gardées pour la flexibilité).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py chunk_utils.py chunk_papers.py scrap_articles.py ./
# L'index ChromaDB pré-construit est copié dans l'image -- construit en
# amont (localement ou en CI), pas régénéré à chaque déploiement (trop
# lent, nécessiterait les PDF + un appel réseau à Hugging Face pour le
# modèle d'embedding).
COPY chroma_db/ ./chroma_db/

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8080}/_stcore/health || exit 1

# Cloud Run injecte PORT dynamiquement (8080 par défaut) -- le conteneur
# DOIT écouter dessus, pas sur un port fixe codé en dur.
ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
