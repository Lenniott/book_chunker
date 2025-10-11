#!/bin/bash
# Complete extraction and vectorization workflow

echo "=== Complete Workflow: Extract + Vectorize ==="
echo ""
echo "This example shows the full pipeline:"
echo "1. Extract PDF to JSON"
echo "2. Analyze the content"
echo "3. Vectorize and load into Qdrant"
echo ""

# Step 1: Extract with hierarchy
echo "Step 1: Extracting book.pdf with hierarchy detection..."
./book_extractor.py book.pdf --extract-images --output my_book.json

# Step 2: Analyze
echo ""
echo "Step 2: Analyzing content structure..."
./vectorize.py my_book.json --analyze-only --chunk-size 3

# Step 3: Vectorize
echo ""
echo "Step 3: Vectorizing with nested sections..."
./vectorize.py my_book.json --collection my_books --chunk-size 3

echo ""
echo "✅ Complete! Your book is now searchable in Qdrant."
echo ""
echo "Collection: my_books"
echo "JSON: my_book.json"

