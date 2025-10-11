#!/bin/bash
# Batch processing multiple books

echo "=== Batch Processing Multiple Books ==="
echo ""

# Create output directory
mkdir -p extracted_books

# Extract all PDFs in current directory
echo "Extracting all PDFs..."
for pdf in *.pdf; do
    if [ -f "$pdf" ]; then
        basename="${pdf%.pdf}"
        echo "Processing: $pdf"
        python book_extractor.py "$pdf" --output "extracted_books/${basename}.json"
    fi
done

# Analyze all extracted books
echo ""
echo "Analyzing extracted content..."
for json in extracted_books/*.json; do
    if [ -f "$json" ]; then
        echo "Analyzing: $json"
        python vectorize.py "$json" --analyze-only
    fi
done

# Vectorize all books into one collection
echo ""
echo "Vectorizing all books into collection 'my_library'..."
python vectorize.py extracted_books/*.json --collection my_library --chunk-size 5

echo ""
echo "✅ Batch processing complete!"
echo "All books are in collection: my_library"

