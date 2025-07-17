#!/bin/bash

echo "Installing packages for Python 3.12 compatibility..."

# Core dependencies
pip install langchain langchain-community langchain-openai chromadb sentence-transformers

# PDF processing (without camelot-py)
pip install PyMuPDF pdf2image pytesseract tabula-py

# Image processing
pip install Pillow opencv-python

# Data processing  
pip install pandas openpyxl

# ML/AI
pip install torch transformers scikit-learn

# LLM and agents
pip install openai

# Utilities
pip install tqdm python-dotenv

# Storage and caching
pip install diskcache lmdb

# NLP
pip install nltk

# Alternative to camelot-py for Python 3.12
pip install pdfplumber

echo "Installation complete!"