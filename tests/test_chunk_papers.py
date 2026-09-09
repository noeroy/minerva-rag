import sys
from pathlib import Path

# Permet d'importer chunk_papers.py depuis la racine du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunk_utils import (
    token_length,
    is_garbled_vertical,
    looks_like_section_title,
    build_chunks,
)


def test_token_length_basic():
    assert token_length("hello world") > 0
    assert token_length("") == 0


def test_token_length_longer_text_more_tokens():
    short = "physics"
    long = "physics measurement cross section neutrino interaction"
    assert token_length(long) > token_length(short)


def test_is_garbled_vertical_detects_single_chars():
    vertical_text = "] x e - p e h ["
    assert is_garbled_vertical(vertical_text) is True


def test_is_garbled_vertical_ignores_normal_text():
    normal_text = "This is a normal sentence about neutrino physics."
    assert is_garbled_vertical(normal_text) is False


def test_is_garbled_vertical_short_text_not_flagged():
    # Moins de 3 mots : pas assez d'info pour juger, doit rester False
    assert is_garbled_vertical("a b") is False


def test_looks_like_section_title_roman_numeral():
    assert looks_like_section_title("I. INTRODUCTION") is True
    assert looks_like_section_title("VII. RESULTS") is True


def test_looks_like_section_title_all_caps():
    assert looks_like_section_title("CONCLUSIONS") is True


def test_looks_like_section_title_rejects_normal_sentence():
    assert looks_like_section_title("This is a regular sentence.") is False


def test_looks_like_section_title_rejects_too_long():
    long_text = "I. " + "word " * 30
    assert looks_like_section_title(long_text) is False


def test_build_chunks_respects_chunk_size():
    elements = [
        {
            "text": "word " * 1000,  # texte volontairement long
            "section": "TEST",
            "extraction_method": "unstructured",
            "page": 1,
        }
    ]
    chunks = build_chunks(elements, chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 1  # doit avoir été découpé en plusieurs morceaux
    for c in chunks:
        assert token_length(c["text"]) <= 120  # marge pour l'overlap/découpe


def test_build_chunks_groups_by_section():
    elements = [
        {"text": "premier morceau de texte", "section": "A", "extraction_method": "unstructured", "page": 1},
        {"text": "deuxième morceau de texte", "section": "B", "extraction_method": "unstructured", "page": 1},
    ]
    chunks = build_chunks(elements, chunk_size=500, chunk_overlap=50)

    sections = {c["section"] for c in chunks}
    assert sections == {"A", "B"}