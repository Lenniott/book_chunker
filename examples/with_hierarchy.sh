#!/bin/bash
# Extraction with hierarchy detection example

echo "=== Book Extraction with Automatic Hierarchy Detection ==="
echo ""
echo "This example shows the hybrid extraction approach:"
echo "1. Bookmarks provide chapter structure"
echo "2. Font analysis detects sections/subsections within chapters"
echo "3. Properly nested JSON output with multiple levels"
echo ""

# Extract with hierarchy (default)
echo "Extracting book.pdf with hierarchy detection..."
./book_extractor.py book.pdf --output book_hierarchical.json

echo ""
echo "Done! Hierarchical structure saved to: book_hierarchical.json"
echo ""
echo "To view the nested structure:"
echo "  cat book_hierarchical.json | jq '.outline[0]' | less"
echo ""
echo "To see just the hierarchy (no paragraphs):"
echo "  cat book_hierarchical.json | jq '.outline[] | {title, page, sections: .sections | length}'"
echo ""
echo "Compare with simple extraction (no hierarchy):"
echo "  ./book_extractor.py book.pdf --no-hierarchy --output book_simple.json"
echo "  diff <(jq . book_hierarchical.json) <(jq . book_simple.json)"

