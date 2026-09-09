import os
import time

import chromadb
import streamlit as st
from mistralai.client import Mistral
from mistralai.client.errors.sdkerror import SDKError
from sentence_transformers import SentenceTransformer

# Choix des modèles : bge-small-en-v1.5 pour les embeddings (local, gratuit),
# Mistral pour la synthèse finale (seul appel payant du pipeline).
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("minerva_papers")

mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Garde-fou : évite qu'un lien public ne génère une facture API incontrôlée.
# Limite par session utilisateur (pas globale) -- suffisant pour une démo.
MAX_QUERIES_PER_SESSION = 10


def call_mistral_with_retry(prompt, max_retries=3, base_delay=2):
    """Appelle Mistral avec retry en cas de rate limit (429).
    Attente exponentielle : 2s, 4s, 8s entre les tentatives."""
    for attempt in range(max_retries):
        try:
            return mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
            )
        except SDKError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                st.info(f"Limite de débit atteinte, nouvelle tentative dans {delay}s...")
                time.sleep(delay)
            else:
                raise


def build_context(results):
    parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        label = "Document de référence générale" if meta.get("doc_type") == "background" else "Mesure MINERvA"
        parts.append(f"[{label} — Source: arXiv:{meta['arxiv_id']} - {meta['title']} - Section: {meta['section']}]\n{doc}")
    return "\n\n---\n\n".join(parts)


def rag_answer_mistral(user_question, search_query=None, n_results=8):
    query_to_search = search_query if search_query else user_question

    query_embedding = model.encode([query_to_search])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)

    context = build_context(results)

    prompt = f"""Voici des passages extraits de papiers scientifiques MINERvA :

{context}

---

Question : {user_question}

Instructions :
- Réponds d'abord UNIQUEMENT à partir des passages fournis ci-dessus.
  Cite chaque affirmation avec le format (arXiv:XXXX.XXXXX).
  Si les passages ne permettent pas de répondre complètement, dis-le explicitement
  dans cette partie.
- Distingue les "Documents de référence générale" (contexte pédagogique) des "Mesures MINERvA"
  (résultats de recherche spécifiques) quand tu cites tes sources.
- Si tu as des connaissances générales pertinentes qui vont au-delà de ces passages,
  ajoute une section séparée intitulée "### Au-delà des sources fournies" —
  clairement distincte de la réponse basée sur les sources, sans mélanger les deux.
- Réponds en Français."""

    response = call_mistral_with_retry(prompt)

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {"arxiv_id": m["arxiv_id"], "title": m["title"], "section": m["section"]}
            for m in results["metadatas"][0]
        ],
    }


st.title("MINERvA RAG — Recherche dans la littérature scientifique")
st.caption(
    "Corpus : 51 papiers de mesure MINERvA + 1 white paper de référence (NuSTEC) "
    "— Embeddings locaux + synthèse Mistral"
)

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

remaining = MAX_QUERIES_PER_SESSION - st.session_state.query_count
st.caption(f"Requêtes restantes pour cette session : {remaining}/{MAX_QUERIES_PER_SESSION}")

question = st.text_input("Pose ta question sur la physique MINERvA :")

if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
    st.warning(
        "Limite de requêtes atteinte pour cette session de démo. "
        "Recharge la page pour repartir avec un nouveau quota."
    )
elif st.button("Chercher") and question:
    st.session_state.query_count += 1
    with st.spinner("Recherche en cours..."):
        result = rag_answer_mistral(question)

    st.markdown("### Réponse")
    st.write(result["answer"])

    st.markdown("### Sources utilisées")
    for s in result["sources"]:
        with st.expander(f"{s['arxiv_id']} — {s['title']}"):
            st.write(f"Section : {s['section']}")
            st.markdown(f"[Voir le PDF](https://arxiv.org/abs/{s['arxiv_id']})")