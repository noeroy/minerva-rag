import chromadb
import os
#from mistralai import Mistral
from mistralai.client import Mistral
from sentence_transformers import SentenceTransformer
import streamlit as st


#choice of Model, we're using a small MistralAI model here
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("minerva_papers")

mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def build_context(results):
    parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        label = "Document de référence générale" if meta.get("doc_type") == "background" else "Mesure MINERvA"
        parts.append(f"[{label} — Source: arXiv:{meta['arxiv_id']} - {meta['title']} - Section: {meta['section']}]\n{doc}")
    return "\n\n---\n\n".join(parts)


def rag_answer_mistral(user_question, search_query=None, n_results=8):
    # Si pas de requête de recherche fournie, utilise directement la question
    query_to_search = search_query if search_query else user_question

    query_embedding = model.encode([query_to_search])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)

    context = build_context(results)

    # for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    #     print("CONTEXTE: ")
    #     print(f"[{meta['arxiv_id']} - {meta['section']}]")
    #     print(doc)
    #     print("---") 

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
- Réponds en Francais."""

    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {"arxiv_id": m["arxiv_id"], "title": m["title"], "section": m["section"]}
            for m in results["metadatas"][0]
        ]
    }

st.title("MINERvA RAG — Recherche dans la littérature scientifique")
st.caption("Corpus : 51 papiers de mesure MINERvA + 1 white paper de référence (NuSTEC) — Embeddings locaux + synthèse Mistral")

question = st.text_input("Pose ta question sur la physique MINERvA :")

if st.button("Chercher") and question:
    with st.spinner("Recherche en cours..."):
        result = rag_answer_mistral(question)

    st.markdown("### Réponse")
    st.write(result["answer"])

    st.markdown("### Sources utilisées")
    for s in result["sources"]:
        with st.expander(f"{s['arxiv_id']} — {s['title']}"):
            st.write(f"Section : {s['section']}")
            st.markdown(f"[Voir le PDF](https://arxiv.org/abs/{s['arxiv_id']})")

# result = rag_answer_mistral("What is the flux uncertainty in the MINERvA medium energy beam and how is it constrained?")
# print(result["answer"])