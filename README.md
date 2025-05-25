# Book Chunker

This project extracts and structures the content of a book PDF into a hierarchical JSON format, using the PDF's outline (table of contents) and robust paragraph extraction.

## Features
- Extracts the outline (TOC) and metadata from a PDF.
- Uses PyMuPDF (fitz) for accurate paragraph/block extraction.
- Outputs a hierarchical JSON structure matching the book's sections and chapters.
- Supports extracting a specific page range or the entire book.

## Requirements
- Python 3.7+
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Extract the full book structure:
```bash
python book_processer.py path/to/book.pdf
```

Extract a specific page range (e.g., pages 88 to 112):
```bash
python book_processer.py path/to/book.pdf "[88,112]"
```

Show the table of contents (outline):
```bash
python book_processer.py path/to/book.pdf toc
```

## Output
- The script outputs a JSON file (`book_structure.json` or `book_structure_START_END.json`) with the following structure:
  - `title`, `author`, `total_pages`: Book metadata
  - `outline`: Hierarchical list of sections/chapters, each with:
    - `title`, `page`: Section title and start page
    - `children`: Nested subsections (if any)
    - `paragraphs`: List of paragraphs (only for leaf sections or gaps between children)

## Notes
- The script uses PyPDF2 for outline and metadata extraction, and PyMuPDF (fitz) for robust paragraph/block extraction.
- Paragraphs are smartly merged to avoid awkward splits.
- For parent sections with children, only gaps strictly between children are included for paragraph extraction.
- If a page range is specified, extraction is strictly limited to those pages.

## Example Output (truncated)
```json
{
  "title": "A Course in Miracles: Original Edition",
  "author": "Helen Schucman",
  "total_pages": 1626,
  "outline": [
    {
      "title": "Chapter 3. Retraining the Mind",
      "page": 88,
      "children": [
        {
          "title": "I. Introduction",
          "page": 88,
          "paragraphs": [
            "This is a course in mind training..."
          ]
        }
      ]
    }
  ]
}
``` 