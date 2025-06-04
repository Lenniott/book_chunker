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
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Book Structure Schema",
  "description": "Schema defining the structure of the book chunker output JSON",
  "type": "object",
  "required": ["title", "author", "total_pages", "outline"],
  "properties": {
    "title": {
      "type": "string",
      "description": "The full title of the book as extracted from the PDF metadata. This is typically found in the PDF's document information dictionary."
    },
    "author": {
      "type": "string",
      "description": "The author(s) of the book as extracted from the PDF metadata. Multiple authors may be separated by commas or other delimiters depending on the PDF's formatting."
    },
    "total_pages": {
      "type": "integer",
      "minimum": 1,
      "description": "The total number of pages in the PDF document. This is used for validation and to ensure all page references are within bounds."
    },
    "outline": {
      "type": "array",
      "description": "A hierarchical array representing the book's table of contents structure. Each element represents a top-level section or chapter.",
      "items": {
        "type": "object",
        "required": ["title", "page"],
        "properties": {
          "title": {
            "type": "string",
            "description": "The title of the section or chapter as it appears in the book's table of contents. This is extracted directly from the PDF's outline structure."
          },
          "page": {
            "type": "integer",
            "minimum": 1,
            "description": "The page number where this section or chapter begins. This is a 1-based index (first page is 1, not 0)."
          },
          "children": {
            "type": "array",
            "description": "An array of subsections or subchapters that belong to this section. This creates the hierarchical structure of the book.",
            "items": {
              "$ref": "#/properties/outline/items"
            }
          },
          "paragraphs": {
            "type": "array",
            "description": "An array of text paragraphs that belong to this section. These are only included for leaf nodes (sections without children) or for content that appears between subsections in parent nodes. Each paragraph is a complete thought or section of text as it appears in the book.",
            "items": {
              "type": "string",
              "description": "A single paragraph of text from the book. Paragraphs are extracted using PyMuPDF's block detection and are merged intelligently to avoid awkward splits. They maintain the original formatting and line breaks as they appear in the book."
            }
          }
        }
      }
    }
  },
  "examples": [
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
  ],
  "additionalProperties": false
}
``` 




