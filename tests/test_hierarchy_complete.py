#!/Users/benjamin/Desktop/CODE/book_chunker/venv/bin/python3
"""
Complete Hierarchy Extraction Test

Demonstrates the full two-phase algorithm:
Phase 1: Identify heading candidates (font size + line filtering)
Phase 2: Build hierarchy tree (full pattern + document order)

Shows the algorithm actually works on real PDFs.
"""

import sys
import fitz  # PyMuPDF
from collections import Counter
import re
import json


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


def extract_with_hierarchy(pdf_path, start_page=1, end_page=None, output_json=None):
    """
    Complete extraction with hierarchy detection.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    if end_page is None:
        end_page = total_pages
    
    start_idx = start_page - 1
    end_idx = min(end_page, total_pages)
    
    print(f"\n{'='*80}")
    print(f"COMPLETE HIERARCHY EXTRACTION")
    print(f"PDF: {pdf_path}")
    print(f"Pages: {start_page}-{end_idx} of {total_pages}")
    print(f"{'='*80}\n")
    
    # ===== PHASE 1: IDENTIFY HEADING CANDIDATES =====
    print("PHASE 1: Identifying Heading Candidates (Font Size Filtering)")
    print("-" * 80)
    
    all_lines = []
    
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
                    case = analyze_text_case(line_text)
                    
                    all_lines.append({
                        'page': page_num + 1,
                        'size': font_size,
                        'bold': bold,
                        'italic': italic,
                        'case': case,
                        'text': line_text
                    })
    
    # Group by font SIZE only
    size_groups = []
    i = 0
    
    while i < len(all_lines):
        current = all_lines[i]
        combined_text = current['text']
        start_page = current['page']
        end_page = current['page']
        line_count = 1
        first_line_idx = i
        
        j = i + 1
        while j < len(all_lines):
            next_line = all_lines[j]
            if (next_line['size'] == current['size'] and 
                next_line['page'] - end_page <= 1):
                combined_text += " " + next_line['text']
                end_page = next_line['page']
                line_count += 1
                j += 1
            else:
                break
        
        size_groups.append({
            'page': start_page,
            'size': current['size'],
            'text': combined_text,
            'line_count': line_count,
            'first_line_idx': first_line_idx
        })
        
        i = j
    
    # Identify body text
    max_group_per_size = {}
    for group in size_groups:
        size = group['size']
        if size not in max_group_per_size:
            max_group_per_size[size] = 0
        max_group_per_size[size] = max(max_group_per_size[size], group['line_count'])
    
    body_text_sizes = {size for size, max_lines in max_group_per_size.items() if max_lines >= 7}
    
    print(f"Body text sizes (max group >= 7): {sorted(body_text_sizes, reverse=True)}")
    
    # Filter heading candidates
    # Only keep sizes LARGER than all body text sizes
    max_body_size = max(body_text_sizes) if body_text_sizes else 0
    
    heading_candidates = []
    for group in size_groups:
        # Only keep sizes larger than body text
        if group['size'] <= max_body_size:
            continue
        # Exclude multi-line groups
        if group['line_count'] > 2:
            continue
        
        # Get full pattern from first line
        first_line = all_lines[group['first_line_idx']]
        
        heading_candidates.append({
            'page': group['page'],
            'size': first_line['size'],
            'bold': first_line['bold'],
            'italic': first_line['italic'],
            'case': first_line['case'],
            'text': group['text'],
            'line_count': group['line_count'],
            'full_pattern': (first_line['size'], first_line['bold'], first_line['case'])
        })
    
    print(f"Heading candidates: {len(heading_candidates)} items (not body text, <= 2 lines)\n")
    
    # ===== PHASE 2: BUILD HIERARCHY =====
    print("PHASE 2: Building Hierarchy (Full Pattern + Document Order)")
    print("-" * 80)
    print()
    
    pattern_to_level = {}
    pattern_stack = []
    tree = []
    last_node_at_level = {}  # Track last node at each level
    
    debug = False  # Set to True to see algorithm working
    
    for heading in heading_candidates:
        pattern = heading['full_pattern']
        
        if debug:
            print(f"\nProcessing: {heading['text'][:40]}")
            print(f"  Pattern: {pattern}")
            print(f"  Stack before: {[str(p) for p in pattern_stack]}")
        
        if pattern in pattern_to_level:
            # Pattern seen before → sibling at that level
            target_level = pattern_to_level[pattern]
            
            if debug:
                print(f"  → Pattern SEEN before at level {target_level}")
            
            # Pop stack back to this level (keep pattern at target_level)
            while len(pattern_stack) > target_level + 1:
                pattern_stack.pop()
            
            current_level = target_level
        else:
            # New pattern → child (next level down)
            current_level = len(pattern_stack)
            pattern_to_level[pattern] = current_level
            pattern_stack.append(pattern)
            
            if debug:
                print(f"  → Pattern NEW, assigned level {current_level}")
        
        if debug:
            print(f"  Current level: {current_level}")
            print(f"  Stack after: {[str(p) for p in pattern_stack]}")
        
        # Create node
        node = {
            'title': heading['text'],
            'page': heading['page'],
            'level': current_level + 1,
            'pattern': {
                'size': heading['size'],
                'bold': heading['bold'],
                'case': heading['case']
            },
            'children': []
        }
        
        # Add node to correct parent
        if current_level == 0:
            # Top level - add to tree
            tree.append(node)
        else:
            # Child - add to parent's children array
            parent_level = current_level - 1
            if parent_level in last_node_at_level:
                parent_node = last_node_at_level[parent_level]
                parent_node['children'].append(node)
            else:
                # Fallback: add to tree if no parent found
                tree.append(node)
        
        # Track this node for future children
        last_node_at_level[current_level] = node
        
        # Clear nodes at deeper levels (they're not parents anymore)
        levels_to_clear = [l for l in last_node_at_level.keys() if l > current_level]
        for l in levels_to_clear:
            del last_node_at_level[l]
    
    # ===== OUTPUT RESULTS =====
    print("RESULTING HIERARCHY TREE:")
    print("=" * 80)
    print()
    
    print_tree(tree)
    
    print("\n" + "=" * 80)
    print("PATTERN-TO-LEVEL MAPPING")
    print("=" * 80)
    print()
    
    # Show which patterns map to which levels
    for pattern, level in sorted(pattern_to_level.items(), key=lambda x: x[1]):
        size, bold, case = pattern
        bold_str = "BOLD" if bold else "norm"
        count = sum(1 for h in heading_candidates if h['full_pattern'] == pattern)
        
        # Get examples
        examples = [h['text'][:60] for h in heading_candidates if h['full_pattern'] == pattern][:2]
        
        print(f"Level {level + 1}: {size}pt {bold_str} {case} ({count} occurrences)")
        for ex in examples:
            print(f"  → \"{ex}\"")
        print()
    
    # Save to JSON if requested
    if output_json:
        output = {
            'pdf': pdf_path,
            'pages': f"{start_page}-{end_idx}",
            'hierarchy': tree,
            'pattern_mapping': {
                f"level_{level+1}": {
                    'size': pattern[0],
                    'bold': pattern[1],
                    'case': pattern[2],
                    'count': sum(1 for h in heading_candidates if h['full_pattern'] == pattern)
                }
                for pattern, level in pattern_to_level.items()
            }
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved hierarchy to: {output_json}")
    
    doc.close()
    
    print("\n" + "=" * 80)
    print("SUCCESS! Hierarchy extraction complete.")
    print("=" * 80)
    print()
    print(f"✓ Detected {len(pattern_to_level)} hierarchy levels")
    print(f"✓ Extracted {len(heading_candidates)} headings")
    print(f"✓ Built nested tree structure")
    print(f"✓ Ready to integrate into book_extractor.py")
    print()
    
    return tree


def print_tree(nodes, indent=0):
    """Print tree structure with indentation."""
    for node in nodes:
        prefix = "  " * indent
        size = node['pattern']['size']
        bold = "BOLD" if node['pattern']['bold'] else "norm"
        case = node['pattern']['case']
        
        text_display = node['title'][:60] + "..." if len(node['title']) > 60 else node['title']
        
        print(f"{prefix}[L{node['level']}] p{node['page']:3d} | {size}pt {bold:5} {case:8} | {text_display}")
        
        if node['children']:
            print_tree(node['children'], indent + 1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_hierarchy_complete.py <pdf_file> [start_page] [end_page] [--output file.json]")
        print("\nComplete hierarchy extraction test")
        print("\nPhase 1: Font size filtering (7+ lines = body text, <= 2 lines = headings)")
        print("Phase 2: Pattern-based hierarchy (size + bold + case, document order)")
        print("\nExamples:")
        print("  python test_hierarchy_complete.py book_2.pdf")
        print("  python test_hierarchy_complete.py book_2.pdf 10 100")
        print("  python test_hierarchy_complete.py book_2.pdf 1 200 --output hierarchy.json")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    start_page = 1
    end_page = None
    output_json = None
    
    args = sys.argv[2:]
    if '--output' in args:
        idx = args.index('--output')
        if idx + 1 < len(args):
            output_json = args[idx + 1]
            args.pop(idx)
            args.pop(idx)
    
    if len(args) >= 1:
        start_page = int(args[0])
    if len(args) >= 2:
        end_page = int(args[1])
    
    extract_with_hierarchy(pdf_path, start_page, end_page, output_json)


if __name__ == '__main__':
    main()

