import json
import re
from pathlib import Path

import tiktoken
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer


_ENCODER = tiktoken.get_encoding("cl100k_base")

METADATA_PATH = Path("papers_metadata.json")
PAPERS_DIR = Path("papers")
CHUNKS_PATH = Path("all_chunks.json")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "minerva_papers"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def token_length(text):
    return len(_ENCODER.encode(text))


# Sometimes the text is garbled in a vertical way, with one character per line. This function detects such cases.
def is_garbled_vertical(text):
    words = text.split()
    if len(words) < 3:
        return False
    single_char_ratio = sum(1 for w in words if len(w) == 1) / len(words)
    return single_char_ratio > 0.5


# Filters out true section titles (e.g., 'I. INTRODUCTION', 'VII. RESULTS').
def looks_like_section_title(text):
    text = text.strip()
    if len(text) < 4 or len(text) > 80:
        return False
    if is_garbled_vertical(text):
        return False
    if re.match(r"^[IVX]+\.\s", text) or text.isupper():
        return True
    return False


# Extracts structured elements from a PDF using unstructured, with a fallback to pdfminer if necessary.
# Fallback seems to be needed in cases where pages are full of images.
def extract_elements_with_fallback(pdf_path):
    elements = partition_pdf(filename=str(pdf_path), strategy="fast")

    if len(elements) > 0:
        return elements, "unstructured"

    text = extract_text(str(pdf_path))
    fallback_element = {"type": "NarrativeText", "text": text, "page": 1}
    return [fallback_element], "pdfminer_fallback"


# Filters out noise and assigns a section to each element.
def clean_elements(elements, extraction_method="unstructured"):
    cleaned = []
    current_section = "Front Matter" if extraction_method == "unstructured" else "Unstructured Fallback"

    for el in elements:
        if isinstance(el, dict):
            el_type = el["type"]
            text = el["text"].strip()
            page = el["page"]
        else:
            el_type = type(el).__name__
            text = str(el).strip()
            page = el.metadata.page_number

        if is_garbled_vertical(text):
            continue
        if el_type in ("Header", "Footer") and len(text) < 15:
            continue
        if len(text) < 10:
            continue

        if el_type == "Title" and looks_like_section_title(text):
            current_section = text

        cleaned.append({
            "text": text,
            "type": el_type,
            "section": current_section,
            "page": page,
            "extraction_method": extraction_method,
        })

    return cleaned


# Groups elements by section and splits them into fixed-size chunks (in tokens).
def build_chunks(cleaned_elements, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=token_length
    )

    sections = {}
    for el in cleaned_elements:
        key = (el["section"], el["extraction_method"])
        sections.setdefault(key, []).append(el)

    all_chunks = []
    for (section_name, method), elements in sections.items():
        full_text = " ".join(el["text"] for el in elements)
        pages = sorted(set(el["page"] for el in elements))

        for chunk_text in splitter.split_text(full_text):
            all_chunks.append({
                "text": chunk_text,
                "section": section_name,
                "pages": pages,
                "extraction_method": method,
            })

    return all_chunks


# Loads already generated chunks if they exist, to avoid reprocessing.
def load_existing_chunks(output_path):
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

#pdf to chunk. Only treats papers that have not yet been chunked.
def run_chunking():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    all_chunks_with_meta = load_existing_chunks(CHUNKS_PATH)
    #gets the id of the chunk already processed
    already_processed = {c["arxiv_id"] for c in all_chunks_with_meta}

    for entry in metadata:
        arxiv_id = entry["id"]

        if arxiv_id in already_processed:
            continue

        pdf_path = PAPERS_DIR / f"{arxiv_id}.pdf"
        if not pdf_path.exists():
            print(f"MANQUANT: {arxiv_id}")
            continue

        elements, method = extract_elements_with_fallback(pdf_path)
        cleaned = clean_elements(elements, extraction_method=method)
        chunks = build_chunks(cleaned)

        for c in chunks:
            c["arxiv_id"] = arxiv_id
            c["title"] = entry["title"]
            c["doc_type"] = entry.get("doc_type", "primary_research")

        all_chunks_with_meta.extend(chunks)
        print(f"OK: {arxiv_id} - {len(chunks)} chunks ({method})")

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks_with_meta, f, ensure_ascii=False, indent=2)

    fallback_count = sum(1 for c in all_chunks_with_meta if c["extraction_method"] == "pdfminer_fallback")
    print(f"\nTotal chunks: {len(all_chunks_with_meta)} sur {len(metadata)} papiers")
    print(f"{fallback_count} chunks issus du fallback pdfminer")

    return all_chunks_with_meta

# Indexes the chunks into ChromaDB, only adding new chunks that aren't already present.
def run_indexing(all_chunks_with_meta):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # IDs déjà présents dans la base -> évite de ré-indexer
    existing_ids = set(collection.get()["ids"])

    # Chaque chunk a un ID stable basé sur son contenu (arxiv_id + position),
    # pas juste sa position dans la liste globale (qui bougerait si l'ordre change)
    to_index = []
    for c in all_chunks_with_meta:
        chunk_id = f"{c['arxiv_id']}_{c['section']}_{hash(c['text']) & 0xffffffff}"
        if chunk_id not in existing_ids:
            to_index.append((chunk_id, c))

    if not to_index:
        print("Rien de nouveau à indexer.")
        return

    texts = [c["text"] for _, c in to_index]
    print(f"Encodage de {len(texts)} nouveaux chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    collection.add(
        ids=[cid for cid, _ in to_index],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {
                "arxiv_id": c["arxiv_id"],
                "title": c["title"],
                "section": c["section"],
                "extraction_method": c["extraction_method"],
                "doc_type": c.get("doc_type", "primary_research"),
            }
            for _, c in to_index
        ],
    )
    print(f"{len(to_index)} nouveaux chunks indexés")
    print(f"Total dans la collection: {collection.count()}")


if __name__ == "__main__":
    chunks = run_chunking()
    run_indexing(chunks)