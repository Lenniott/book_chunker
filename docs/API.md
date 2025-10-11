# API Reference

Complete reference for Book Chunker's main scripts and utilities.

## Command Line Scripts

### book_extractor.py

Extract structured content from PDF files to JSON.

#### Synopsis

```bash
python book_extractor.py PDF_PATH [OPTIONS]
```

#### Arguments

- `PDF_PATH` (required): Path to PDF file

#### Options

- `--pages, -p PAGE_RANGE`: Extract specific page range (format: "START-END")
- `--output, -o FILENAME`: Custom output JSON filename
- `--extract-images, -i`: Extract and embed images as base64
- `--toc`: Show table of contents and exit (no extraction)
- `--verbose, -v`: Enable verbose logging

#### Examples

```bash
# Basic extraction
python book_extractor.py book.pdf

# With images
python book_extractor.py book.pdf --extract-images

# Specific pages
python book_extractor.py book.pdf --pages 10-50

# Custom output
python book_extractor.py book.pdf --output chapter1.json

# Show TOC only
python book_extractor.py book.pdf --toc

# Verbose mode
python book_extractor.py book.pdf --verbose
```

#### Output

Creates a JSON file with this structure:

```json
{
  "title": "string",
  "author": "string",
  "total_pages": number,
  "outline": [
    {
      "title": "string",
      "page": number,
      "paragraphs": ["string", ...],
      "images": [
        {
          "format": "string",
          "base64": "string",
          "width": number,
          "height": number,
          "size_bytes": number
        }
      ],
      "children": [...]
    }
  ],
  "extraction_info": {
    "source": "string",
    "page_range": "string",
    "images_extracted": boolean,
    "extractor_version": "string"
  }
}
```

#### Return Codes

- `0`: Success
- `1`: Error (file not found, invalid format, etc.)

---

### vectorize.py

Load JSON files into Qdrant vector database.

#### Synopsis

```bash
python vectorize.py JSON_FILE [JSON_FILE ...] [OPTIONS]
```

#### Arguments

- `JSON_FILE` (required): One or more JSON files to process

#### Options

- `--collection, -c NAME`: Qdrant collection name (required unless --analyze-only)
- `--chunk-size, -s SIZE`: Paragraphs per chunk (0 = whole sections, default: 0)
- `--analyze-only, -a`: Analyze content without vectorizing
- `--verbose, -v`: Enable verbose logging

#### Examples

```bash
# Analyze only
python vectorize.py book.json --analyze-only

# Vectorize whole sections
python vectorize.py book.json --collection my_books

# Vectorize with chunking
python vectorize.py book.json --collection my_books --chunk-size 3

# Multiple books
python vectorize.py book1.json book2.json book3.json --collection library

# With verbose output
python vectorize.py book.json --collection my_books --verbose
```

#### Output

**Analysis Mode:**
Prints statistics to stdout:
```
📊 Content Analysis:
  Book: Title by Author
  Total sections: 150
  Sections with content: 145
  Total paragraphs: 2500
  Total images: 50
  Estimated vectors: 2500
  Chunking: Whole sections
  Largest section: Chapter 5
  Largest section size: 15,000 characters
```

**Vectorization Mode:**
Creates vectors in Qdrant and prints summary:
```
✅ Vectorization complete!
  Collection: my_books
  Vectors created: 2500
  Total size: 15.36 MB
  Average vector size: 6.29 KB
```

#### Return Codes

- `0`: Success
- `1`: Error (connection failed, invalid JSON, etc.)

---

## Python API

### Utilities Module

Import utilities for use in Python scripts.

```python
from utils import (
    extract_images_from_page,
    extract_paragraphs_from_pages,
    clean_text,
    get_page_number,
    print_outline_tree
)
```

#### extract_images_from_page()

Extract images from a PDF page.

```python
def extract_images_from_page(fitz_doc, page_num: int) -> List[Dict]
```

**Parameters:**
- `fitz_doc`: PyMuPDF document object
- `page_num`: Page number (1-indexed)

**Returns:**
List of image dictionaries:
```python
[
  {
    "image_index": 0,
    "format": "png",
    "base64": "iVBORw0KGgo...",
    "width": 800,
    "height": 600,
    "size_bytes": 45000,
    "extraction_method": "multiple_fallback"
  },
  ...
]
```

**Example:**
```python
import fitz
from utils import extract_images_from_page

doc = fitz.open("book.pdf")
images = extract_images_from_page(doc, 1)
print(f"Found {len(images)} images on page 1")
```

#### extract_paragraphs_from_pages()

Extract paragraphs from page range with smart merging.

```python
def extract_paragraphs_from_pages(
    reader,
    start_page: int,
    end_page: int,
    fitz_doc=None
) -> List[str]
```

**Parameters:**
- `reader`: PyPDF2 reader object (compatibility, not used)
- `start_page`: Starting page (1-indexed)
- `end_page`: Ending page (1-indexed, inclusive)
- `fitz_doc`: PyMuPDF document object

**Returns:**
List of paragraph strings.

**Example:**
```python
import PyPDF2
import fitz
from utils import extract_paragraphs_from_pages

pdf_reader = PyPDF2.PdfReader("book.pdf")
fitz_doc = fitz.open("book.pdf")

paragraphs = extract_paragraphs_from_pages(
    pdf_reader, 1, 10, fitz_doc
)
print(f"Extracted {len(paragraphs)} paragraphs from pages 1-10")
```

#### clean_text()

Clean and normalize text.

```python
def clean_text(text: str) -> str
```

**Parameters:**
- `text`: Raw text string

**Returns:**
Cleaned text with normalized whitespace.

**Example:**
```python
from utils import clean_text

raw = "This  has   extra    spaces\n\n\nand newlines"
cleaned = clean_text(raw)
# Result: "This has extra spaces and newlines"
```

#### get_page_number()

Get page number from PDF bookmark destination.

```python
def get_page_number(dest, reader) -> Optional[int]
```

**Parameters:**
- `dest`: Bookmark destination object
- `reader`: PyPDF2 reader object

**Returns:**
Page number (1-indexed) or None if not found.

**Example:**
```python
import PyPDF2
from utils import get_page_number

reader = PyPDF2.PdfReader("book.pdf")
outline = reader.outline

for item in outline:
    if hasattr(item, 'title'):
        page_num = get_page_number(item, reader)
        print(f"{item.title}: Page {page_num}")
```

#### print_outline_tree()

Print formatted tree view of PDF outline.

```python
def print_outline_tree(outline, reader, level=0, parent_is_last_list=None)
```

**Parameters:**
- `outline`: PDF outline structure
- `reader`: PyPDF2 reader object
- `level`: Current depth (internal use)
- `parent_is_last_list`: Parent tracking (internal use)

**Returns:**
None (prints to stdout)

**Example:**
```python
import PyPDF2
from utils import print_outline_tree

reader = PyPDF2.PdfReader("book.pdf")
print_outline_tree(reader.outline, reader)
```

Output:
```
├── Chapter 1 [Page 1]
│   ├── Section 1.1 [Page 2]
│   └── Section 1.2 [Page 5]
├── Chapter 2 [Page 10]
    └── Section 2.1 [Page 11]
```

---

### Core Functions

#### process_pdf()

Main PDF processing function (from book_extractor.py).

```python
def process_pdf(
    pdf_path: str,
    mode: str = 'json',
    page_range: Optional[Tuple[int, int]] = None,
    output_file: Optional[str] = None,
    extract_images: bool = False
) -> Optional[Dict]
```

**Parameters:**
- `pdf_path`: Path to PDF file
- `mode`: 'json' or 'toc'
- `page_range`: Optional (start, end) tuple
- `output_file`: Optional output filename
- `extract_images`: Whether to extract images

**Returns:**
Dictionary with extracted content, or None on error.

**Example:**
```python
from book_extractor import process_pdf

result = process_pdf(
    "book.pdf",
    mode='json',
    page_range=(10, 50),
    extract_images=True
)

if result:
    print(f"Extracted {len(result['outline'])} sections")
```

#### vectorize_sections()

Vectorize book sections and store in Qdrant (from vectorize.py).

```python
def vectorize_sections(
    book_data: Dict,
    collection_name: str,
    chunk_size: int = 0
) -> Dict[str, int]
```

**Parameters:**
- `book_data`: Book JSON data
- `collection_name`: Qdrant collection name
- `chunk_size`: Paragraphs per chunk (0 = whole sections)

**Returns:**
Statistics dictionary:
```python
{
    "vector_count": 2500,
    "total_size_bytes": 15728640,
    "avg_vector_size": 6291.456
}
```

**Example:**
```python
import json
from vectorize import vectorize_sections, load_book_json

book_data = load_book_json("book_structure.json")
stats = vectorize_sections(book_data, "my_collection", chunk_size=3)

print(f"Created {stats['vector_count']} vectors")
```

#### analyze_content()

Analyze content without vectorizing (from vectorize.py).

```python
def analyze_content(
    book_data: Dict,
    chunk_size: int = 0
) -> Dict[str, Any]
```

**Parameters:**
- `book_data`: Book JSON data
- `chunk_size`: Paragraphs per chunk (0 = whole sections)

**Returns:**
Analysis dictionary:
```python
{
    "book_title": "Title",
    "book_author": "Author",
    "total_sections": 150,
    "sections_with_content": 145,
    "total_paragraphs": 2500,
    "total_images": 50,
    "estimated_chunks": 2500,
    "largest_section": "Chapter 5",
    "largest_section_size": 15000
}
```

**Example:**
```python
from vectorize import analyze_content, load_book_json

book_data = load_book_json("book_structure.json")
analysis = analyze_content(book_data, chunk_size=3)

print(f"Book will create ~{analysis['estimated_chunks']} vectors")
```

---

## Environment Variables

Configuration via `.env` file:

```bash
# Required for vectorization
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
OPENAI_API_KEY=sk-your-key

# Optional: PostgreSQL (database module)
PG_DBNAME=book_library
PG_USER=postgres
PG_PASSWORD=your_password
PG_HOST=localhost
PG_PORT=5432
```

**Access in Python:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
openai_key = os.getenv("OPENAI_API_KEY")
```

---

## JSON Schema

### Book Structure JSON

```typescript
interface BookStructure {
  title: string;
  author: string;
  total_pages: number;
  outline: Section[];
  extraction_info: {
    source: string;
    page_range: string;
    images_extracted: boolean;
    extractor_version: string;
  };
}

interface Section {
  title: string;
  page: number;
  paragraphs?: string[];
  images?: Image[];
  children?: Section[];
}

interface Image {
  image_index: number;
  format: string;
  base64: string;
  width: number;
  height: number;
  size_bytes: number;
  extraction_method: string;
}
```

### Qdrant Metadata

Metadata stored in each vector's payload:

```typescript
interface VectorMetadata {
  book_title: string;
  book_author: string;
  section_title: string;
  section_path: string;  // e.g., "1.2.3"
  page: number;
  parent_path?: string;
  paragraph_count: number;
  text: string;
  has_images: boolean;
  image_count: number;
  chunking_method: "whole_section" | "paragraph_groups";
  
  // If chunked:
  chunk_index?: number;
  total_chunks?: number;
  is_first_chunk?: boolean;
  is_last_chunk?: boolean;
  paragraphs_per_chunk?: number;
}
```

---

## Error Handling

All functions may raise:

- `FileNotFoundError`: File not found
- `json.JSONDecodeError`: Invalid JSON
- `Exception`: Generic errors with descriptive messages

Always wrap calls in try-except:

```python
try:
    result = process_pdf("book.pdf")
except FileNotFoundError:
    print("PDF not found")
except Exception as e:
    print(f"Error: {e}")
```

---

## Logging

Configure logging level:

```python
import logging

# Set to DEBUG for verbose output
logging.basicConfig(level=logging.DEBUG)

# Or use --verbose flag in CLI
python book_extractor.py book.pdf --verbose
```

Log files created:
- `book_extraction.log` - Extraction logs
- `vectorization.log` - Vectorization logs

---

## Extension Points

### Custom Paragraph Merging

Modify `utils/paragraph_extraction.py`:

```python
# Adjust merging thresholds
new_paragraph = (
    vertical_gap > YOUR_THRESHOLD or
    indent_difference > YOUR_VALUE or
    your_custom_logic()
)
```

### Custom Image Processing

Modify `utils/image_extraction.py`:

```python
def extract_images_from_page(fitz_doc, page_num):
    # Add custom processing
    images = original_extraction(fitz_doc, page_num)
    
    for img in images:
        img['custom_field'] = process(img)
    
    return images
```

### Custom Vectorization

Modify `vectorize.py`:

```python
def vectorize_text(text):
    # Use different embedding model
    response = your_model.embed(text)
    return response.embedding, len(response.embedding) * 4
```

---

## Version Compatibility

- Python: 3.7+
- PyPDF2: 2.0+
- PyMuPDF: 1.19+
- Qdrant: 1.0+
- OpenAI: 1.0+

Check versions:
```bash
python --version
pip list | grep -E 'PyPDF|PyMuPDF|qdrant|openai'
```

