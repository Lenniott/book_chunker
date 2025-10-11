# Hierarchy Detection

Comprehensive guide to the automatic hierarchy detection system in Book Chunker.

## Overview

The hierarchy detection system automatically discovers and extracts the nested structure of books (chapters, sections, subsections) using a **hybrid approach**:

1. **PDF Bookmarks** → Reliable top-level structure (chapters)
2. **Font Analysis** → Deeper structure within chapters (sections, subsections)

This works across different book formats without hardcoding font sizes or styles.

## The Problem

PDFs vary widely in their bookmark depth:
- Some books: 3 levels in bookmarks (chapters → sections → subsections)
- Other books: 1 level in bookmarks (chapters only)
- But content often has 3-5 actual hierarchy levels

**Example:** A book might have bookmarks for chapters only, but within each chapter there are sections (ALL CAPS headings), subsections (numbered items), and sub-subsections.

## The Solution: Hybrid Approach

### Phase 0: Bookmarks (Reliable Skeleton)
```
From PDF TOC:
- Chapter 1: Introduction (page 10)
- Chapter 2: Methods (page 25)
- Chapter 3: Results (page 50)
```

### Phase 1: Font Analysis (Find Headings)
Within each chapter, identify heading candidates:

**Algorithm:**
1. Collect all text lines with font metadata
2. Group consecutive same-size lines (merge multi-line headings)
3. Identify body text: Font sizes where max consecutive group >= 7 lines
4. Filter headings: Sizes > body text AND groups <= 2 lines

**Result:** Clean list of heading candidates with their patterns

### Phase 2: Build Hierarchy (Pattern Stacking)
Use document order to infer nesting:

**Algorithm:**
1. Track pattern stack: `[(size, bold, case), ...]`
2. For each heading:
   - If pattern seen before → Pop to that level (sibling)
   - If pattern new → Push to stack (child)
3. Build nested tree structure

**Result:** Properly nested hierarchy

## How It Works (Detailed)

### Step 1: Baseline Identification

**Goal:** Find body text font size

**Method:** Font sizes that appear in long consecutive runs

```python
# Example from book_2.pdf:
15.0pt → max group: 103 consecutive lines → BODY TEXT
11.2pt → max group: 106 consecutive lines → BODY TEXT (footnotes)
21.0pt → max group: 2 consecutive lines → HEADING candidate
30.0pt → max group: 2 consecutive lines → HEADING candidate
```

**Why this works:** Body text dominates the content. Headings are short, standalone.

### Step 2: Line Merging

**Goal:** Combine multi-line headings into single items

**Method:** Merge consecutive lines with same font size

```
Before merging:
  Line 1: 30pt "A Case for"
  Line 2: 30pt "Load Training"

After merging:
  Item: 30pt "A Case for Load Training" (2 lines)
```

**Why this works:** PDF line breaks are arbitrary (page width). Same-size consecutive lines are likely one logical heading.

### Step 3: Heading Filtering

**Goal:** Separate headings from body text variants

**Rules:**
```python
is_heading = (
    font_size > max_body_size AND
    line_count <= 2
)
```

**Examples:**
```
✅ 30pt BOLD "Chapter 1" (1 line) → HEADING
✅ 21pt BOLD "Introduction" (2 lines) → HEADING  
❌ 15pt normal (103 lines) → BODY TEXT
❌ 21pt BOLD (15 lines) → BODY TEXT (long paragraph in big font)
```

### Step 4: Pattern Stacking

**Goal:** Build hierarchy from document order

**Concept:** When reading top-to-bottom:
- First time seeing pattern → It's a child of current context
- Seeing pattern again → It's a sibling of its previous occurrence

**Example:**

```
Document flow:
1. 30pt BOLD "Chapter 1"
2. 21pt BOLD ALL_CAPS "SECTION A"
3. 21pt BOLD Title "Subsection 1"
4. 21pt BOLD Title "Subsection 2"
5. 21pt BOLD ALL_CAPS "SECTION B"  ← Same as #2!
6. 30pt BOLD "Chapter 2"            ← Same as #1!

Pattern stack evolution:
1. Stack: [30pt-BOLD] → Level 1
2. Stack: [30pt-BOLD, 21pt-BOLD-CAPS] → Level 2 (child of #1)
3. Stack: [30pt-BOLD, 21pt-BOLD-CAPS, 21pt-BOLD-Title] → Level 3 (child of #2)
4. Stack: [30pt-BOLD, 21pt-BOLD-CAPS, 21pt-BOLD-Title] → Level 3 (sibling of #3)
5. Stack: [30pt-BOLD, 21pt-BOLD-CAPS] → Level 2 (sibling of #2, pop #3)
6. Stack: [30pt-BOLD] → Level 1 (sibling of #1, pop #2)

Result:
Chapter 1
  ├─ SECTION A
  │   ├─ Subsection 1
  │   └─ Subsection 2
  └─ SECTION B
Chapter 2
```

### Step 5: Paragraph Extraction

**Goal:** Extract content for each section

**Method:**
1. Section starts at its page
2. Section ends before next section's page (or parent's end)
3. If section has children, extract only paragraphs before first child
4. Use smart paragraph merging (from `utils/paragraph_extraction.py`)

**Result:** Each section has its own paragraphs, properly scoped

## Real-World Examples

### Book 2: Built from Broken

**Bookmarks:** 1 level (chapters only)

**Font analysis detected:**
- Level 1: 30pt (chapter titles - also in bookmarks)
- Level 2: 21pt BOLD ALL_CAPS (sections like "HEALING LOADS")
- Level 3: 21pt BOLD Title (subsections like "Bodybuilding")
- Level 3: 21pt BOLD Mixed (numbered like "1. Posture")

**Result:** 3 levels extracted (1 from bookmarks, 2-3 from fonts)

### Book 3: Rehab Guide

**Bookmarks:** 2 levels (parts → chapters)

**Font analysis detected:**
- Level 3: 27.5pt ALL_CAPS (major sections)
- Level 4: 16.9pt ALL_CAPS (subsections)
- Level 5: 21.2pt lower (minor sections)

**Result:** 5 levels extracted (2 from bookmarks, 3-5 from fonts)

### Book 5: Managing Priorities

**Bookmarks:** 2 levels (parts → chapters)

**Font analysis detected:**
- Level 3: 25pt BOLD Title (sections)
- Level 4: 21.2pt BOLD (subsections)
- Level 5: Nested subsections

**Result:** 4 levels extracted (2 from bookmarks, 3-4 from fonts)

## Key Principles

### 1. Relative, Not Absolute

Don't hardcode "15pt = body text". Instead:
- Find what's most common → that's body text
- Everything larger + rare → headings
- Works across any book format

### 2. Document Flow Matters

Visual hierarchy follows reading order:
- First occurrence of pattern = child
- Repeated pattern = sibling
- Based on gestalt principles of document design

### 3. Font Size is Primary

Within same font size, bold/case distinguish sub-levels:
- 21pt BOLD ALL_CAPS (first) → Section
- 21pt BOLD Title (after) → Subsection of that section

### 4. Consecutive = Same Item

PDF line breaks are arbitrary:
- "A Case for" + "Load Training" = ONE heading
- Merge by font size only
- Ignore case differences in merging

## Configuration

### Default Behavior

```bash
./book_extractor.py book.pdf
# Hierarchy detection: ENABLED by default
```

### Disable Hierarchy Detection

```bash
./book_extractor.py book.pdf --no-hierarchy
# Simple extraction: bookmarks + paragraphs only
```

### Parameters (Internal)

Can be adjusted in `book_extractor.py`:

```python
# Body text threshold
max_lines_threshold = 7  # Consecutive lines to identify body text

# Heading line limit  
max_heading_lines = 2  # Headings must be <= 2 lines after merging

# Page proximity for merging
page_window = 1  # Merge lines within 1 page of each other
```

## Output Format

### With Hierarchy

```json
{
  "title": "Book Title",
  "outline": [
    {
      "title": "Chapter 1",
      "page": 10,
      "sections": [
        {
          "title": "Section 1.1",
          "page": 12,
          "paragraphs": ["...", "..."],
          "sections": [
            {
              "title": "Subsection 1.1.1",
              "page": 15,
              "paragraphs": ["...", "..."],
              "sections": []
            }
          ]
        }
      ],
      "paragraphs": ["...", "..."]
    }
  ]
}
```

### Without Hierarchy

```json
{
  "title": "Book Title",
  "outline": [
    {
      "title": "Chapter 1",
      "page": 10,
      "paragraphs": ["...", "...", "..."]
    }
  ]
}
```

## Limitations

### What Works Well

✅ Non-fiction books with clear visual hierarchy  
✅ Technical books and textbooks  
✅ Books with consistent heading styles  
✅ Multiple heading levels (2-5)  

### What Doesn't Work

❌ Fiction books (no clear hierarchy)  
❌ Books with inconsistent formatting  
❌ Scanned PDFs without font metadata  
❌ Books where headings = same size as body text  

### Known Issues

1. **Quotes in large bold fonts** may be detected as headings
2. **Figure labels** might appear as sections
3. **TOC entries** might be detected if similar to actual headings
4. **References** sections may have unusual structure

Most issues can be resolved by adjusting thresholds or adding filters.

## Debugging

### Check Body Text Detection

```bash
python tools/extract_heading_patterns.py problem_book.pdf 20 50
```

Look at "Body text sizes" - does it correctly identify the main text size?

### Visualize Font Patterns

```bash
python tools/list_font_patterns.py problem_book.pdf 20 50
```

See patterns in order - are the headings obvious?

### Test Hierarchy

```bash
python tools/infer_hierarchy.py problem_book.pdf 20 100
```

Check if hierarchy makes sense - proper nesting?

### Extract and Inspect

```bash
./book_extractor.py problem_book.pdf --output debug.json
cat debug.json | jq '.outline[0]' | less
```

Review actual extraction - does structure match the book?

## Algorithm Evolution

This algorithm was developed through iteration:

1. **v0.1**: Font analysis only (unreliable)
2. **v0.5**: Bookmarks only (incomplete)
3. **v1.0**: Hybrid with hardcoded sizes (not universal)
4. **v2.0**: Relative detection + pattern stacking (current)

See `FONT_ANALYSIS_RESULTS.md` and test scripts for development history.

## Future Enhancements

Possible improvements:
- Machine learning for pattern classification
- Manual rule configuration files
- Interactive hierarchy refinement
- Confidence scoring for uncertain headings
- Special handling for common edge cases

## References

- Gestalt principles in document design
- Pattern stacking algorithm (custom)
- Statistical text analysis
- PyMuPDF font metadata extraction

