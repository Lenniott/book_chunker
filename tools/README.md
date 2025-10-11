# Analysis Tools

Development tool for analyzing PDF font patterns when hierarchy detection needs debugging.

## Tool: `list_font_patterns.py`

Lists font patterns in document order with automatic line merging.

### Purpose

Use this when:
- Hierarchy detection fails on a PDF
- Need to understand a book's font structure
- Debugging why sections aren't being detected
- Analyzing a new/unusual book format

### Usage

```bash
python tools/list_font_patterns.py <pdf_file> [start_page] [end_page] [--show-body]
```

**Examples:**
```bash
# Analyze pages 10-50
python tools/list_font_patterns.py book.pdf 10 50

# Analyze entire book
python tools/list_font_patterns.py book.pdf

# Include body text in output
python tools/list_font_patterns.py book.pdf 10 50 --show-body
```

### What It Shows

1. **Body text baseline** - Automatically identified
2. **Font patterns in order** - Exactly as they appear in the document
3. **Merged multi-line headings** - Consecutive same-size lines combined
4. **Pattern summary** - Unique patterns with character counts

### Sample Output

```
================================================================================
FONT PATTERN LISTING: book.pdf
Pages 10-30 of 567
================================================================================

📘 Body Text (baseline): 15.0pt norm Mixed
   Total: 15,099 characters

================================================================================
FONT PATTERNS IN DOCUMENT ORDER
================================================================================

[p     10] 30.0pt BOLD → PART 1 Principles and Strategy
[p     11] 30.0pt BOLD → 1 A Case for Load Training (3 lines)
[p     14] 21.0pt BOLD → HEALING LOADS
[p     16] 21.0pt BOLD → STRENGTH, MUSCLE, AND MORTALITY
[p     20] 21.0pt BOLD → Bodybuilding
[p     21] 21.0pt BOLD → Progressive Strength Training
```

### Understanding the Output

**Pattern format:**
```
[page] size weight case → text
```

**Line merging:**
- `(3 lines)` = This heading was split across 3 lines in the PDF
- Lines with same font size are automatically merged

**Body text:**
- Marked as `[BODY]` if shown with `--show-body`
- Hidden by default to reduce noise

### Using the Results

1. **Identify heading sizes:**
   - Look for patterns larger than body text
   - Note which sizes appear as headings

2. **Understand hierarchy:**
   - Larger sizes = higher level
   - Same size with different case = sub-levels
   - Document order shows nesting

3. **Debug extraction:**
   - If headings missing → check if detected here
   - If wrong level → check pattern order
   - If not detected → may be same size as body text

### When Hierarchy Detection Fails

If `book_extractor.py` produces flat structure:

1. **Run this tool:**
   ```bash
   python tools/list_font_patterns.py problem_book.pdf 20 100
   ```

2. **Check:**
   - Is body text correctly identified?
   - Are headings larger than body text?
   - Do heading patterns repeat in a logical way?

3. **Solutions:**
   - If headings = body text size → Manual structuring needed
   - If patterns illogical → Use `--no-hierarchy` flag
   - If partially working → File an issue with output

## Why Only One Tool?

The `book_extractor.py` now works automatically on most books. This single tool provides all the debugging capability needed for edge cases without cluttering the project with multiple similar analysis scripts.

For most users: **You'll never need this tool** - the extractor just works!

## See Also

- `docs/HIERARCHY_DETECTION.md` - Algorithm explanation
- `docs/TROUBLESHOOTING.md` - Common issues
- `tests/test_hybrid_extraction.py` - Algorithm validation tests
