# Architecture

This document describes how Book Chunker works internally.

## System Design

### High-Level Overview

```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│         book_extractor.py                       │
│  ┌────────────────────────────────────────────┐ │
│  │ 1. PyPDF2: Read bookmarks (TOC)            │ │
│  │ 2. PyMuPDF: Extract text blocks            │ │
│  │ 3. Smart paragraph grouping                │ │
│  │ 4. Optional: Extract images                │ │
│  │ 5. Build hierarchical structure            │ │
│  └────────────────────────────────────────────┘ │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  JSON File   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│         vectorize.py                            │
│  ┌────────────────────────────────────────────┐ │
│  │ 1. Load JSON                               │ │
│  │ 2. Flatten hierarchical structure          │ │
│  │ 3. Optional: Split into chunks             │ │
│  │ 4. OpenAI: Generate embeddings             │ │
│  │ 5. Qdrant: Store vectors + metadata        │ │
│  └────────────────────────────────────────────┘ │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│    Qdrant    │
│ (Vector DB)  │
└──────────────┘
```

## Core Components

### 1. Extraction Pipeline (book_extractor.py)

#### Phase 1: Structure Detection
- Uses PyPDF2 to read PDF bookmarks (TOC)
- Bookmarks provide reliable top-level structure
- Each bookmark maps to a page number

#### Phase 2: Content Extraction
- Uses PyMuPDF (fitz) to extract text blocks
- Blocks include position, text, and bounding box
- Sorts blocks by vertical then horizontal position

#### Phase 3: Paragraph Assembly
Smart merging logic:
- Detects vertical gaps between blocks
- Tracks indentation changes
- Respects sentence endings
- Handles page boundaries intelligently

Key algorithm:
```python
for each text block:
    if significant_gap OR indent_change OR sentence_end:
        finalize_current_paragraph()
        start_new_paragraph()
    else:
        append_to_current_paragraph()
```

#### Phase 4: Image Extraction (Optional)
Multiple fallback methods:
1. `extract_image()` - Direct image extraction
2. `Pixmap()` - Create pixmap from reference
3. Page rendering - Render image area

All images converted to base64 for JSON embedding.

#### Phase 5: JSON Assembly
- Recursive tree building
- Handles nested sections (children)
- Fills gaps between children
- Attaches paragraphs and images

### 2. Vectorization Pipeline (vectorize.py)

#### Phase 1: Structure Flattening
- Converts hierarchical JSON to flat list
- Maintains path information (e.g., "1.2.3")
- Preserves parent context

#### Phase 2: Chunking (Optional)
If `--chunk-size N`:
- Split sections with >N paragraphs
- Maintain paragraph boundaries
- Track chunk position (first/last/middle)

#### Phase 3: Embedding Generation
- Combine paragraphs with `\n\n` separator
- Truncate if >30,000 characters
- Call OpenAI text-embedding-3-small
- Returns 1536-dimensional vector

#### Phase 4: Metadata Assembly
Each vector includes:
```json
{
  "book_title": "...",
  "book_author": "...",
  "section_title": "...",
  "section_path": "1.2.3",
  "page": 42,
  "text": "full text content",
  "paragraph_count": 5,
  "image_count": 2,
  "chunk_index": 0,
  "total_chunks": 3
}
```

#### Phase 5: Vector Storage
- Generate unique UUID for each vector
- Create Qdrant PointStruct
- Batch upload to collection
- All metadata in payload (no separate DB needed)

## Utility Modules

### utils/paragraph_extraction.py
Responsible for smart text extraction:
- Loads pages via PyMuPDF
- Sorts blocks by position
- Implements merging logic
- Handles special cases (lists, quotes, etc.)

### utils/image_extraction.py
Handles image extraction:
- Multiple extraction methods
- Format detection (PNG, JPEG, etc.)
- Dimension extraction
- Base64 encoding

### utils/outline_processing.py
Bookmark/TOC utilities:
- Page number resolution
- Tree visualization
- Outline traversal

## Design Decisions

### Why Bookmarks First?

**Pros:**
- Reliable structure from PDF metadata
- Fast - no complex analysis needed
- Matches reader's TOC view
- Standard across PDFs

**Cons:**
- Some PDFs lack bookmarks
- May have shallow hierarchy

**Solution:** Hybrid approach (future)
- Start with bookmarks
- Optionally enhance with font analysis

### Why JSON as Intermediate Format?

**Benefits:**
1. **Inspectable**: Human-readable, debuggable
2. **Flexible**: Can be processed further
3. **Portable**: Works across systems
4. **Versionable**: Git-friendly
5. **Decoupled**: Extraction separate from vectorization

**Tradeoffs:**
- Larger file size (especially with images)
- Extra I/O step

Decision: Benefits outweigh costs for development and debugging.

### Why No PostgreSQL by Default?

**Original design** used PostgreSQL to store:
- Book metadata
- Content nodes
- Paragraphs
- Vector metadata

**New design** stores everything in Qdrant payload:
- Simpler setup
- Fewer moving parts
- All data co-located with vectors
- Fast retrieval

**When to use PostgreSQL:**
- Multi-user system
- Complex relational queries
- Content versioning
- Separate storage/vectors

## Performance Characteristics

### Extraction
- **Speed**: ~1-5 pages/second (depends on complexity)
- **Memory**: ~50-200MB per book
- **Bottleneck**: Text block sorting, paragraph merging

### Image Extraction
- **Speed**: ~0.5-2 seconds/image
- **Memory**: Depends on image size
- **Bottleneck**: Image decoding, base64 encoding

### Vectorization
- **Speed**: ~1-5 sections/second
- **Memory**: ~100-500MB
- **Bottleneck**: OpenAI API calls (rate limited)

### Chunking Impact
- More chunks = more API calls
- More chunks = better granularity
- Recommendation: 3-5 paragraphs per chunk

## Error Handling

### Extraction Errors
- Missing bookmarks → Log warning, try to extract anyway
- Image extraction fails → Log error, continue with text
- Invalid page range → Exit with clear message

### Vectorization Errors
- OpenAI API error → Retry with backoff
- Text too long → Truncate with warning
- Qdrant connection fail → Exit with connection details

## Future Enhancements

### Planned
1. **Hybrid extraction**: Bookmarks + font analysis
2. **Font-based sub-sectioning**: Detect deeper hierarchy
3. **Figure detection**: Link images to mentions
4. **Table extraction**: Parse tables to structured data
5. **Incremental updates**: Only vectorize changed sections

### Under Consideration
1. **Multiple LLM support**: Cohere, Anthropic, etc.
2. **Local embeddings**: Sentence Transformers
3. **Multi-language**: Non-English books
4. **OCR support**: Scanned PDFs
5. **Streaming**: Process very large books

## Testing Strategy

### Unit Tests
- Paragraph merging logic
- Image extraction methods
- Path generation

### Integration Tests
- End-to-end extraction
- JSON format validation
- Vector upload

### Regression Tests
- Sample PDFs with known outputs
- Edge cases (no TOC, images only, etc.)

## Deployment Considerations

### Development
```bash
python book_extractor.py book.pdf
python vectorize.py book_structure.json --collection dev
```

### Production
- Use environment variables for API keys
- Run Qdrant in Docker/cloud
- Batch process with error handling
- Monitor API rate limits
- Log to file for debugging

## Troubleshooting Guide

### Common Issues

**Extraction hangs**
- Large PDF with many images
- Solution: Run without `--extract-images` first

**Weird paragraph breaks**
- PDF has unusual formatting
- Solution: Adjust merging thresholds in code

**Missing sections**
- Bookmark page numbers incorrect
- Solution: Check with `--toc`, manually fix JSON

**Vectorization fails**
- OpenAI rate limit
- Solution: Add retry logic, use chunking

## Contributing

When modifying architecture:
1. Update this document
2. Consider backward compatibility
3. Add migration path
4. Update tests
5. Benchmark performance impact

