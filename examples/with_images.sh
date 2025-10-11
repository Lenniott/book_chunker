#!/bin/bash
# Extraction with images example

echo "=== Book Extraction with Images ==="
echo ""
echo "This example extracts a PDF with embedded images:"
echo "1. Extract PDF structure and content"
echo "2. Extract and embed all images as base64"
echo "3. Save to JSON"
echo ""

# Extract with images and hierarchy
echo "Extracting book.pdf with images and hierarchy detection..."
./book_extractor.py book.pdf --extract-images

echo ""
echo "Done! Output includes:"
echo "  ✓ Hierarchical structure (chapters → sections → subsections)"
echo "  ✓ Embedded images (base64)"
echo "  ✓ Smart paragraphs"
echo ""
echo "Note: Files with images will be much larger!"
echo "Check file size: ls -lh book_structure_v2.json"

