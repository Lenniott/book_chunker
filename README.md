# Book Chunker

Extract structured content from PDF books and load into vector databases for semantic search.

## Overview

Book Chunker is a robust pipeline for processing PDF books into searchable, structured content. It uses PDF bookmarks (table of contents) for reliable structure extraction, combined with smart paragraph detection and optional image extraction.

### Key Features

- 🎯 **Hybrid extraction**: Bookmarks + Font analysis for complete hierarchy
- 🔄 **Automatic hierarchy detection**: Discovers 2-5 levels of nested structure
- 📚 **Universal compatibility**: Works across different book formats (no hardcoded sizes)
- 🖼️ **Image extraction**: Optional base64 embedding of images
- 📝 **Smart paragraph grouping**: Intelligent merging across page boundaries
- 🔍 **Vector search ready**: Direct integration with Qdrant
- 📊 **Chunking support**: Split large sections into manageable chunks
- 🎯 **JSON-first workflow**: Always outputs JSON, vectorization optional

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd book_chunker

# Create virtual environment (already done for you!)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies (already done!)
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

**Good news:** The scripts are already configured to use the virtual environment automatically!

```bash
# Just run directly (no need to activate venv first!)
./book_extractor.py your_book.pdf

# For development work (optional - to run other Python commands):
source venv/bin/activate
# or use the helper: source activate.sh
```

### Basic Usage

```bash
# 1. Extract PDF to JSON with automatic hierarchy detection
./book_extractor.py your_book.pdf

# 2. Review the nested JSON structure
cat book_structure_v2.json | jq . | less

# 3. (Optional) Vectorize and load into Qdrant
./vectorize.py book_structure_v2.json --collection my_books
```

**Note:** Scripts automatically use the virtual environment. No need to activate it first!

### Hierarchy Detection

By default, the extractor automatically detects nested structure:

```bash
# Automatic hierarchy (default)
./book_extractor.py book.pdf

# Simple extraction (bookmarks only, no hierarchy)
./book_extractor.py book.pdf --no-hierarchy
```

**What gets detected:**
- Chapters from PDF bookmarks (reliable Level 1)
- Sections from font analysis (Level 2: larger fonts, bold, ALL CAPS)
- Subsections from font patterns (Level 3+: numbered, title case)

See `docs/HIERARCHY_DETECTION.md` for details.

### With Images

```bash
# Extract with images embedded
python book_extractor.py your_book.pdf --extract-images

# Note: Files with images will be much larger!
```

## Main Scripts

### `book_extractor.py` - PDF to JSON

The primary extraction tool. Converts PDFs to structured JSON using bookmarks for hierarchy.

```bash
# Basic extraction
python book_extractor.py book.pdf

# With images
python book_extractor.py book.pdf --extract-images

# Specific page range
python book_extractor.py book.pdf --pages 10-50

# Custom output filename
python book_extractor.py book.pdf --output my_book.json

# Show table of contents only
python book_extractor.py book.pdf --toc
```

**Output:** JSON file with hierarchical structure matching the book's TOC.

### `vectorize.py` - JSON to Vector Database

Loads structured JSON into Qdrant for semantic search.

```bash
# Analyze content without vectorizing
python vectorize.py book_structure.json --analyze-only

# Vectorize whole sections
python vectorize.py book_structure.json --collection my_books

# Vectorize with chunking (3 paragraphs per chunk)
python vectorize.py book_structure.json --collection my_books --chunk-size 3

# Batch process multiple books
python vectorize.py book1.json book2.json book3.json --collection my_library
```

**Requires:** OpenAI API key and Qdrant instance (see Configuration below)

## JSON Output Format

### With Hierarchy Detection (Default)

```json
{
  "title": "Book Title",
  "author": "Author Name",
  "total_pages": 500,
  "outline": [
    {
      "title": "Chapter 1: Introduction",
      "page": 1,
      "paragraphs": ["Intro paragraph before sections..."],
      "sections": [
        {
          "title": "MAJOR SECTION",
          "page": 5,
          "paragraphs": ["Section content..."],
          "sections": [
            {
              "title": "Subsection 1.1",
              "page": 8,
              "paragraphs": ["Subsection content..."],
              "sections": []
            }
          ]
        }
      ]
    }
  ]
}
```

**Note:** Properly nested structure with sections at multiple levels!

## Configuration

### Environment Variables

Create a `.env` file with the following:

```bash
# Required for vectorization
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
OPENAI_API_KEY=sk-your-key

# Optional: PostgreSQL (if using database module)
PG_DBNAME=book_library
PG_USER=postgres
PG_PASSWORD=your_password
PG_HOST=localhost
PG_PORT=5432
```

### Setting up Qdrant

**Option 1: Docker (easiest)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option 2: Qdrant Cloud**
Sign up at https://cloud.qdrant.io and get your URL and API key.

## Examples

See the `examples/` directory for complete workflow scripts:

- `basic_extraction.sh` - Simple PDF to JSON
- `with_images.sh` - Extraction with images
- `with_vectorization.sh` - Complete pipeline
- `batch_processing.sh` - Process multiple books

## Project Structure

```
book_chunker/
├── book_extractor.py          # Main extraction script
├── vectorize.py               # Vectorization script
├── utils/                     # Reusable utilities
│   ├── image_extraction.py
│   ├── paragraph_extraction.py
│   └── outline_processing.py
├── examples/                  # Usage examples
├── experimental/              # Advanced features
├── database/                  # PostgreSQL integration (optional)
├── tests/                     # Test files
└── docs/                      # Additional documentation
```

## Advanced Features

### Experimental Tools

The `experimental/` directory contains advanced processing tools:
- **font_aware_processor.py**: Font-based structure detection
- **section_processor.py**: Post-processing restructuring
- **node_combiner.py**: Merge related paragraphs

See `experimental/README.md` for details.

### PostgreSQL Integration

The `database/` module provides optional PostgreSQL storage for book content. Most users won't need this - Qdrant stores all necessary metadata.

See `database/README.md` for setup instructions.

## Workflow Comparison

### Simple Workflow (Recommended)
```bash
PDF → book_extractor.py → JSON → vectorize.py → Qdrant
```
All metadata stored in Qdrant payload. Simple, fast, easy to manage.

### Advanced Workflow (Optional)
```bash
PDF → book_extractor.py → JSON → db_loader.py → PostgreSQL
                                              ↓
                                         vectorize.py → Qdrant
```
Separate storage and vectors. Use if you need complex queries or multi-user system.

## Requirements

- Python 3.7+
- PyPDF2
- PyMuPDF (fitz)
- OpenAI API key (for vectorization)
- Qdrant instance (for vectorization)

See `requirements.txt` for complete list.

## Troubleshooting

### "No bookmarks found"
Some PDFs lack bookmarks. Try:
1. Check with: `python book_extractor.py book.pdf --toc`
2. Use `experimental/font_aware_processor.py` for font-based extraction

### "Failed to vectorize"
- Check OpenAI API key in `.env`
- Verify API key has credits
- Check network connection

### "Qdrant connection failed"
- Ensure Qdrant is running: `docker ps`
- Check QDRANT_URL in `.env`
- Verify port 6333 is accessible

### Images not extracting
- Some PDF images are embedded in non-standard ways
- Check logs for specific extraction errors
- Images may be vector graphics (not extractable as raster)

## Contributing

Contributions welcome! Please:
1. Check existing issues
2. Create feature branch
3. Add tests for new features
4. Update documentation
5. Submit pull request

## License

[Your chosen license]

## Changelog

See `CHANGELOG.md` for version history and updates.

## Support

- Documentation: See `docs/` directory
- Issues: [GitHub Issues]
- Discussions: [GitHub Discussions]

## Acknowledgments

Built with:
- PyPDF2 for PDF parsing
- PyMuPDF for text and image extraction
- OpenAI for embeddings
- Qdrant for vector search
