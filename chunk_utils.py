import re

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ENCODER = tiktoken.get_encoding("cl100k_base")


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