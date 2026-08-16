# Multimodal RAG Assistant

[![CI](https://github.com/ayush28-5/multimodal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ayush28-5/multimodal-rag/actions/workflows/ci.yml)

A local, multimodal Retrieval-Augmented Generation (RAG) system that answers questions from PDFs, Word docs, spreadsheets, and images using locally-run LLMs (Ollama) and a vision-language model (Florence-2) for image understanding.

## Features

- Multi-format document parsing (PDF, DOCX, PPTX, XLSX, CSV, HTML, images)
- Image classification (flowchart vs. photo) using a custom heuristic classifier
- Vector search with ChromaDB
- Conversational query rewriting for follow-up questions
- Streaming responses via a local LLM (Phi-3 Mini through Ollama)

## Testing & CI

- 46 unit tests (`pytest`) covering core logic — chunking, retrieval dedup, query rewriting, image classification
- All model/network calls mocked for fast, reliable tests
- Continuous Integration via GitHub Actions — runs the full test suite on every push

## Tech Stack

Python · LangChain · ChromaDB · Ollama · Streamlit · Florence-2 · pytest · GitHub Actions
