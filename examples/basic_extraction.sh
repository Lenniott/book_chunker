#!/bin/bash
# Basic PDF extraction example

echo "=== Basic Book Extraction ==="
echo ""
echo "This example shows the simplest workflow:"
echo "1. Extract a PDF to JSON using bookmarks for structure"
echo "2. Review the JSON output"
echo ""

# Extract the PDF
echo "Extracting book.pdf with automatic hierarchy detection..."
./book_extractor.py book.pdf

# Show the result
echo ""
echo "Done! Output saved to: book_structure_v2.json"
echo ""
echo "To view the JSON:"
echo "  cat book_structure_v2.json | less"
echo ""
echo "Or use jq for prettier output:"
echo "  cat book_structure_v2.json | jq . | less"
echo ""
echo "To see the hierarchy structure:"
echo "  cat book_structure_v2.json | jq '.outline[] | {title, sections: .sections | length}'"

