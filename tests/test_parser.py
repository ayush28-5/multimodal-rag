"""
Unit tests for src/parser.py

Covers text chunking and the 6-heuristic flowchart/image classifier.
classify_image is tested against synthetically generated images so
no real photos/flowcharts are needed as fixtures.
"""
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PIL import Image, ImageDraw

from src.parser import extract_text_chunks, classify_image
from src.config import MIN_TEXT_LENGTH, CHUNK_SIZE, CHUNK_OVERLAP


# ============================================
# extract_text_chunks
# ============================================

def test_extract_text_chunks_below_min_length_returns_empty():
    short_text = "too short"
    assert len(short_text) < MIN_TEXT_LENGTH
    assert extract_text_chunks(short_text, page_num=1, source_file="doc.pdf") == []


def test_extract_text_chunks_empty_string_returns_empty():
    assert extract_text_chunks("", page_num=1, source_file="doc.pdf") == []
    assert extract_text_chunks(None, page_num=1, source_file="doc.pdf") == []


def test_extract_text_chunks_single_chunk_metadata():
    text = "x" * (MIN_TEXT_LENGTH + 10)
    chunks = extract_text_chunks(text, page_num=3, source_file="report.pdf")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["content_type"] == "text"
    assert chunk["page"] == 3
    assert chunk["source_file"] == "report.pdf"
    assert chunk["chunk_index"] == 0
    assert chunk["text"] == text


def test_extract_text_chunks_splits_long_text_with_overlap():
    # Long enough to require multiple CHUNK_SIZE-sized windows
    text = "a" * (CHUNK_SIZE * 3)
    chunks = extract_text_chunks(text, page_num=1, source_file="doc.pdf")

    assert len(chunks) > 1
    # chunk_index should increment sequentially
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    # Each chunk shouldn't exceed the configured chunk size
    assert all(len(c["text"]) <= CHUNK_SIZE for c in chunks)


def test_extract_text_chunks_drops_trailing_fragment_under_min_length():
    # Construct text so the final leftover window is short and should be dropped
    text = "a" * CHUNK_SIZE + "b" * 5  # trailing fragment well under MIN_TEXT_LENGTH
    chunks = extract_text_chunks(text, page_num=1, source_file="doc.pdf")
    # No chunk's text should be the tiny trailing fragment
    assert all(len(c["text"]) >= MIN_TEXT_LENGTH for c in chunks)


# ============================================
# classify_image — synthetic fixtures
# ============================================

def _make_flowchart_like_image(path, size=(400, 300)):
    """Mostly white background, few flat colors, sharp box/line edges — should read as 'flowchart'."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # A couple of solid-colored boxes with sharp edges, connected by a line
    draw.rectangle([40, 40, 160, 100], outline=(0, 0, 0), fill=(220, 220, 255), width=3)
    draw.rectangle([240, 180, 360, 240], outline=(0, 0, 0), fill=(220, 255, 220), width=3)
    draw.line([160, 70, 240, 210], fill=(0, 0, 0), width=3)
    img.save(path)


def _make_photo_like_image(path, size=(400, 300)):
    """Dense random noise across many colors — should read as 'image', not 'flowchart'."""
    random.seed(42)
    img = Image.new("RGB", size)
    pixels = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            pixels[x, y] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
    img.save(path)


@pytest.fixture
def tmp_image_dir(tmp_path):
    return tmp_path


def test_classify_image_flowchart_like(tmp_image_dir):
    path = os.path.join(tmp_image_dir, "flowchart.png")
    _make_flowchart_like_image(path)
    assert classify_image(path) == "flowchart"


def test_classify_image_photo_like(tmp_image_dir):
    path = os.path.join(tmp_image_dir, "photo.png")
    _make_photo_like_image(path)
    assert classify_image(path) == "image"


def test_classify_image_handles_missing_file_gracefully(tmp_image_dir):
    missing_path = os.path.join(tmp_image_dir, "does_not_exist.png")
    # Should not raise — falls back to "image" on any error
    assert classify_image(missing_path) == "image"


def test_classify_image_solid_white_is_flowchart(tmp_image_dir):
    # Pure white, single color -> high white ratio, very few colors, low gradient
    path = os.path.join(tmp_image_dir, "blank.png")
    Image.new("RGB", (300, 300), color=(255, 255, 255)).save(path)
    assert classify_image(path) == "flowchart"
