# Experimental Features

This directory contains advanced and experimental tools for PDF processing. These are **optional** and not required for basic extraction and vectorization.

## Contents

### `font_aware_processor.py`
Advanced PDF processor that uses font analysis to detect document structure.

**When to use:**
- PDFs with shallow or missing bookmarks
- Need deeper hierarchy detection beyond TOC
- Want detailed font and style analysis

**Features:**
- Font-based structure detection (size, bold, case)
- List item detection
- Figure/table reference detection
- Image extraction with metadata

**Usage:**
```bash
python experimental/font_aware_processor.py book.pdf
```

### `section_processor.py`
Post-processes extracted JSON to create additional section layers based on content patterns.

**When to use:**
- Want to further subdivide sections
- Have consistent heading patterns in content
- Need to reorganize structure after extraction

**Usage:**
```bash
# Analyze what would be created
python experimental/section_processor.py analyze --file book_structure.json --criteria 'case_type="all_caps",is_bold=true'

# Apply the restructuring
python experimental/section_processor.py process --file book_structure.json --criteria 'case_type="all_caps",is_bold=true'
```

### `node_combiner.py`
Combines adjacent paragraphs based on dual criteria (e.g., figure captions + descriptions).

**When to use:**
- Want to merge related paragraphs
- Combine list headers with lists
- Merge figure references with captions

**Usage:**
```bash
# Find potential combinations
python experimental/node_combiner.py analyze --first 'text=":$"' --second 'is_list_item=true'

# Apply combinations
python experimental/node_combiner.py combine --first 'is_figure=true' --second 'font_size>10'
```

### `add_node_paths.py`
Utility to add hierarchical paths to nodes in JSON (e.g., "1.2.3").

**Usage:**
```bash
python experimental/add_node_paths.py book_structure.json
```

## Note

These tools were developed during the evolution of this project. The main `book_extractor.py` incorporates the most reliable techniques. Use these experimental tools when you need additional control or specialized processing.

## Support

These experimental tools may have:
- Less documentation
- Fewer tests
- Evolving APIs
- Edge cases

Use at your own discretion. Check the source code for detailed implementation.

