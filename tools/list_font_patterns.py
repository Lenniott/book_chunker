#!/Users/benjamin/Desktop/CODE/book_chunker/venv/bin/python3
"""
List Font Patterns - Show font patterns in document order

Simple listing of font configurations page by page, so you can:
1. See patterns in context
2. Identify what's a heading vs body text
3. Create your own hierarchy rules
4. Understand the document structure visually
"""

import sys
import fitz  # PyMuPDF
from collections import Counter
import re


def analyze_text_case(text):
    """Determine the case pattern of text."""
    text = text.strip()
    if not text:
        return "empty"
    
    clean_text = re.sub(r'[^\w\s]', '', text)
    if not clean_text:
        return "punct"
    
    words = clean_text.split()
    if not words:
        return "empty"
    
    if all(word.isupper() for word in words):
        return "ALL_CAPS"
    elif all(word.islower() for word in words):
        return "lower"
    elif all(len(word) > 0 and word[0].isupper() and (len(word) == 1 or word[1:].islower()) for word in words):
        return "Title"
    elif len(words) > 0 and words[0][0].isupper():
        return "Sentence"
    else:
        return "Mixed"


def is_bold(flags):
    return bool(flags & 2**4)


def is_italic(flags):
    return bool(flags & 2**1)


def list_font_patterns(pdf_path, start_page=1, end_page=None, show_body_text=False):
    """
    List all font patterns in document order, page by page.
    
    Args:
        pdf_path: Path to PDF
        start_page: Start page (1-indexed)
        end_page: End page (1-indexed, None = all pages)
        show_body_text: Whether to show common body text patterns
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    if end_page is None:
        end_page = total_pages
    
    start_idx = start_page - 1
    end_idx = min(end_page, total_pages)
    
    print(f"\n{'='*80}")
    print(f"FONT PATTERN LISTING: {pdf_path}")
    print(f"Pages {start_page}-{end_idx} of {total_pages}")
    print(f"{'='*80}\n")
    
    # First pass: identify baseline (most common pattern)
    pattern_char_counts = Counter()
    
    for page_num in range(start_idx, end_idx):
        page = doc.load_page(page_num)
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                line_text = ""
                line_spans = []
                
                for span in line["spans"]:
                    span_text = span["text"].strip()
                    if span_text:
                        line_text += span_text + " "
                        line_spans.append(span)
                
                line_text = line_text.strip()
                
                if line_text and line_spans:
                    span = line_spans[0]
                    font_size = round(span["size"], 1)
                    bold = is_bold(span["flags"])
                    italic = is_italic(span["flags"])
                    case_type = analyze_text_case(line_text)
                    
                    pattern = (font_size, bold, italic, case_type)
                    pattern_char_counts[pattern] += len(line_text)
    
    # Find baseline (body text)
    baseline_pattern = max(pattern_char_counts.items(), key=lambda x: x[1])
    baseline_key = baseline_pattern[0]
    baseline_chars = baseline_pattern[1]
    
    size, bold, italic, case = baseline_key
    bold_str = "BOLD" if bold else "norm"
    italic_str = "+italic" if italic else ""
    
    print(f"📘 Body Text (baseline): {size}pt {bold_str}{italic_str} {case}")
    print(f"   Total: {baseline_chars:,} characters")
    print(f"\n{'='*80}")
    print(f"FONT PATTERNS IN DOCUMENT ORDER")
    print(f"{'='*80}\n")
    print("Legend: [Page] Pattern → Text")
    print(f"{'-'*80}\n")
    
    # Second pass: collect and merge consecutive same-pattern lines
    pattern_items = []  # List of (page, pattern, combined_text)
    
    for page_num in range(start_idx, end_idx):
        page = doc.load_page(page_num)
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                line_text = ""
                line_spans = []
                
                for span in line["spans"]:
                    span_text = span["text"].strip()
                    if span_text:
                        line_text += span_text + " "
                        line_spans.append(span)
                
                line_text = line_text.strip()
                
                if line_text and line_spans:
                    span = line_spans[0]
                    font_size = round(span["size"], 1)
                    bold = is_bold(span["flags"])
                    italic = is_italic(span["flags"])
                    case_type = analyze_text_case(line_text)
                    
                    pattern = (font_size, bold, italic, case_type)
                    current_page = page_num + 1
                    
                    # Skip body text unless requested
                    if pattern == baseline_key and not show_body_text:
                        continue
                    
                    # Add to items list
                    pattern_items.append({
                        'page': current_page,
                        'pattern': pattern,
                        'text': line_text
                    })
    
    # Merge consecutive items with same pattern (ignoring case)
    merged_items = []
    i = 0
    
    while i < len(pattern_items):
        current = pattern_items[i]
        combined_text = current['text']
        start_page = current['page']
        end_page = current['page']
        
        # Get pattern without case for comparison
        curr_size, curr_bold, curr_italic, curr_case = current['pattern']
        curr_pattern_no_case = (curr_size, curr_bold, curr_italic)
        
        # Look ahead for consecutive items with same pattern (ignoring case)
        j = i + 1
        while j < len(pattern_items):
            next_item = pattern_items[j]
            next_size, next_bold, next_italic, next_case = next_item['pattern']
            next_pattern_no_case = (next_size, next_bold, next_italic)
            
            # Same pattern (ignoring case) and close pages (within 1 page)?
            if (next_pattern_no_case == curr_pattern_no_case and 
                next_item['page'] - end_page <= 1):
                combined_text += " " + next_item['text']
                end_page = next_item['page']
                j += 1
            else:
                break
        
        # Add merged item
        merged_items.append({
            'page': start_page,
            'page_range': f"{start_page}-{end_page}" if end_page != start_page else str(start_page),
            'pattern': current['pattern'],
            'text': combined_text,
            'line_count': j - i
        })
        
        i = j  # Jump to next unprocessed item
    
    # Print merged items
    for item in merged_items:
        size, bold, italic, case = item['pattern']
        bold_str = "BOLD" if bold else "norm"
        italic_str = "+italic" if italic else ""
        style = f"{bold_str}{italic_str}".ljust(12)
        
        # Mark if it's baseline
        baseline_marker = " [BODY]" if item['pattern'] == baseline_key else ""
        
        # Show line count if merged
        line_info = f" ({item['line_count']} lines)" if item['line_count'] > 1 else ""
        
        # Truncate text for display
        text_display = item['text'][:100] + "..." if len(item['text']) > 100 else item['text']
        
        print(f"[p{item['page_range']:>7}] {size:4.1f}pt {style} {case:10s}{baseline_marker}{line_info}")
        print(f"          → {text_display}")
        print()
    
    doc.close()
    
    # Print summary
    print("=" * 80)
    print("UNIQUE PATTERNS SUMMARY")
    print("=" * 80)
    print()
    
    # Sort patterns by size (descending)
    sorted_patterns = sorted(pattern_char_counts.items(), key=lambda x: -x[0][0])
    
    for pattern, char_count in sorted_patterns:
        if pattern == baseline_key:
            continue  # Skip baseline in summary
        
        size, bold, italic, case = pattern
        bold_str = "BOLD" if bold else "norm"
        italic_str = "+italic" if italic else ""
        style = f"{bold_str}{italic_str}".ljust(12)
        
        lines = char_count // 50  # Rough estimate
        
        print(f"{size:4.1f}pt {style} {case:10s} → {char_count:6,} chars (~{lines} lines)")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python list_font_patterns.py <pdf_file> [start_page] [end_page] [--show-body]")
        print("\nLists font patterns in document order for schema creation")
        print("\nExamples:")
        print("  python list_font_patterns.py book_2.pdf")
        print("  python list_font_patterns.py book_2.pdf 10 50")
        print("  python list_font_patterns.py book_2.pdf 1 100 --show-body")
        print("\nUse this to:")
        print("  - See font patterns in context")
        print("  - Identify heading vs body text visually")
        print("  - Create hierarchy detection rules")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    start_page = 1
    end_page = None
    show_body = False
    
    # Parse arguments
    args = sys.argv[2:]
    if '--show-body' in args:
        show_body = True
        args.remove('--show-body')
    
    if len(args) >= 1:
        start_page = int(args[0])
    if len(args) >= 2:
        end_page = int(args[1])
    
    list_font_patterns(pdf_path, start_page, end_page, show_body)


if __name__ == '__main__':
    main()

