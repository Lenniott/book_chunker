# CHANGELOG

## [2.0.0] - 2025-10-11 - Hierarchy Detection System

### Added
- **🎯 Hybrid extraction**: Combines PDF bookmarks with font analysis for complete structure
- **🔄 Automatic hierarchy detection**: Discovers 2-5 nested levels automatically
- **📊 Pattern-based nesting**: Uses document flow to infer parent-child relationships
- **🌐 Universal compatibility**: Font-size agnostic, works across different book formats
- **🛠️ Analysis tool**: `list_font_patterns.py` for debugging hierarchy detection
- **📚 Comprehensive docs**: New `HIERARCHY_DETECTION.md` explaining the algorithm

### Changed
- **`book_extractor.py` now uses hybrid extraction by default**
- Added `--no-hierarchy` flag for simple bookmark-only extraction
- Improved paragraph extraction to work within hierarchical sections
- Output filename changed to `book_structure_v2.json` to indicate v2 format
- Reorganized project structure (moved old tools to experimental/)

### Technical Details
- **Two-phase algorithm**:
  - Phase 1: Font size filtering (body text = 7+ consecutive lines)
  - Phase 2: Pattern stacking (document order determines nesting)
- **Body text identification**: Statistical analysis of consecutive line groups
- **Pattern repetition**: Same pattern appearing = sibling section
- **Tested on 5 different book formats**: All pass validation

### Fixed
- Multi-line headings now properly merged (e.g., "A Case for Load Training")
- Font size detection no longer hardcoded (adapts to each book)
- Paragraph extraction properly scoped to section boundaries
- Subsections correctly nest under their parent sections

### Migration
- Old `book_extractor.py` → `experimental/book_extractor_simple.py`
- Test/analysis scripts → `tools/` directory
- See `MIGRATION.md` for upgrade guide

---

# CHANGELOG

## [2025-06-24] - Content Analysis and Paragraph-Based Chunking

### Added
- **Content Structure Analysis Mode**: New `--analyze-only` flag that analyzes how content will be combined without actually vectorizing
  - Shows section-by-section breakdown of paragraphs, images, and text statistics
  - Identifies large sections that would be truncated
  - Outputs detailed JSON analysis files for inspection
  
- **Paragraph-Based Chunking**: New `--chunk-size` parameter for vectorization
  - Default `--chunk-size 0`: Uses whole sections (original behavior)
  - `--chunk-size N`: Splits sections into chunks of N paragraphs each
  - Preserves paragraph boundaries when chunking
  - Maintains image associations with their respective paragraph chunks
  
- **Enhanced Metadata for Chunks**: When using chunking, each vector includes:
  - `chunk_index`: Which chunk within the section (0, 1, 2...)
  - `total_chunks_in_section`: How many chunks the section was split into
  - `is_first_chunk`/`is_last_chunk`: Boolean flags for chunk position
  - `chunking_method`: "paragraph_groups"
  - `paragraphs_per_chunk`: The chunk size used
  - `chunk_text_preview`: First 200 characters of combined text
  
- **Chunked Analysis Mode**: `--analyze-only --chunk-size N` shows:
  - How sections would be split into chunks
  - Chunk-level statistics and content previews
  - Distribution of chunks per section
  - No large chunks (>30k chars) when properly chunked

### Fixed
- **CRITICAL SQL Bug - Cartesian Product**: Fixed major bug in `get_sections_with_content()` query that was creating a cartesian product between paragraphs and images
  - **Before**: LOW BACK PAIN showed 1,659 paragraphs (79 real paragraphs × 21 images = 1,659)
  - **After**: LOW BACK PAIN correctly shows 79 paragraphs and 21 images
  - **Impact**: Reduced total "paragraphs" from 20,527 to 2,643 (real count)
  - **Solution**: Used separate CTEs for paragraphs and images instead of cross-joining them
- **Large Section Handling**: Chunking eliminates the token limit issues that occurred with massive sections

### Technical Details
- **Analysis Files**: 
  - Whole sections: `content_analysis_TIMESTAMP.json`
  - Chunked: `chunked_analysis_Npara_TIMESTAMP.json`
- **Vector IDs**: Chunks use format `{section_id}_chunk_{index}` for unique identification
- **PostgreSQL Storage**: Chunk metadata stored with `search_level = 'chunk'` vs `'section'`

### Usage Examples
```bash
# Analyze without vectorizing (whole sections)
python qdrant.py --book "Built from Broken" --analyze-only

# Analyze with 3-paragraph chunks
python qdrant.py --book "Built from Broken" --analyze-only --chunk-size 3

# Vectorize with 5-paragraph chunks
python qdrant.py --book "Built from Broken" --collection my_collection --chunk-size 5

# Vectorize whole sections (original behavior)
python qdrant.py --book "Built from Broken" --collection my_collection
```

### Statistics from Analysis
- **Book**: Built from Broken (45MB JSON, 144 sections)
- **Before Fix**: 20,527 "paragraphs" (cartesian product), 6,876 chunks, 21,921 "images"
- **After Fix**: 2,643 real paragraphs, 916 chunks, 301 real images
- **3-Paragraph Chunks**: 916 chunks, 0 large chunks, avg 6.5 chunks/section
- **Content**: 2,643 paragraphs, 301 images, 713k characters, 112k words

---

## [2025-06-23] - Image Support and Database Loading

### Added
- Enhanced PostgreSQL schema with images table
- Image metadata inclusion in vector payloads
- Book loading with hierarchical node paths
- Complete Built from Broken book processing

### Technical
- Vector database (Qdrant) integration with image metadata
- PostgreSQL storage of book content and image data
- Node path assignment using materialized path pattern

--- 