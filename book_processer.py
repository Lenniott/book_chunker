import PyPDF2
import sys
import logging
import json
import re
import fitz  # PyMuPDF
from PyPDF2.generic import IndirectObject
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_outline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add a dedicated handler for the tree log (no timestamp/level)
tree_log_handler = logging.FileHandler('outline_tree.log', mode='w')
tree_log_handler.setFormatter(logging.Formatter('%(message)s'))
tree_logger = logging.getLogger('tree_logger')
tree_logger.setLevel(logging.INFO)
tree_logger.addHandler(tree_log_handler)
tree_logger.propagate = False

def get_page_number(dest, reader):
    try:
        if hasattr(dest, 'page') and dest.page is not None:
            for i, page in enumerate(reader.pages):
                if page == dest.page or getattr(page, 'indirect_ref', None) == getattr(dest.page, 'indirect_ref', None):
                    return i + 1
        if isinstance(dest, dict) and '/Page' in dest:
            page_ref = dest['/Page']
            if isinstance(page_ref, IndirectObject):
                page_ref = page_ref.get_object()
            for i, page in enumerate(reader.pages):
                if page.get_object() == page_ref:
                    return i + 1
    except Exception as e:
        logger.debug(f"Error getting page number: {str(e)}")
    return None

def print_outline_tree(outline, reader, level=0, parent_is_last_list=None):
    if parent_is_last_list is None:
        parent_is_last_list = []
    i = 0
    n = len(outline)
    while i < n:
        item = outline[i]
        is_last = (i == n - 1)
        if hasattr(item, 'title') or (isinstance(item, dict) and '/Title' in item):
            title = getattr(item, 'title', None) or item.get('/Title', None)
            page_num = get_page_number(item, reader)
            page_str = f" [Page {page_num}]" if page_num else " [Page ?]"
            tree_prefix = ''.join(['    ' if last else '│   ' for last in parent_is_last_list])
            branch = '└── ' if is_last else '├── '
            output = f"{tree_prefix}{branch}{title}{page_str}"
            print(output)
            logger.info(output)
            tree_logger.info(output)
            if i + 1 < n and isinstance(outline[i + 1], list):
                print_outline_tree(outline[i + 1], reader, level + 1, parent_is_last_list + [is_last])
                i += 1
        i += 1

def clean_text(text):
    """Clean text while preserving legitimate spacing and content."""
    if not text:
        return ''
    # Remove multiple spaces but preserve paragraph structure
    return re.sub(r'\s+', ' ', text).strip()

def extract_paragraphs_from_pages(reader, start_page, end_page, fitz_doc=None):
    """Extract paragraphs from a range of pages using fitz (PyMuPDF) for block extraction, with smart merging."""
    if not fitz_doc:
        return []

    paragraphs = []
    current_paragraph = []
    last_block_bbox = None
    
    for i in range(start_page - 1, end_page):
        page = fitz_doc.load_page(i)
        blocks = page.get_text("blocks")
        
        # Sort blocks by vertical position then horizontal
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        for block in blocks:
            text = block[4].strip()
            if not text:
                continue
                
            bbox = block[:4]  # x0, y0, x1, y1
            
            # Start new paragraph if:
            # 1. Significant vertical gap
            # 2. Different indentation
            # 3. Previous paragraph ended with clear terminator
            if current_paragraph and last_block_bbox:
                vertical_gap = bbox[1] - last_block_bbox[3]  # y0 of current - y1 of last
                indent_difference = abs(bbox[0] - last_block_bbox[0])  # x0 difference
                last_text = current_paragraph[-1]
                
                new_paragraph = (
                    vertical_gap > 1.5 * (bbox[3] - bbox[1]) or  # Gap > 1.5 times line height
                    indent_difference > 20 or  # Significant indent change
                    re.search(r'[.!?]\s*$', last_text) or  # Clear sentence ending
                    (len(last_text) > 2 and last_text[-1] == '"' and last_text[-2] in '.!?') or  # Quote after sentence
                    text[0].isupper()  # New sentence starts with capital
                )
                
                if new_paragraph:
                    merged = ' '.join(current_paragraph)
                    if merged.strip():
                        paragraphs.append(clean_text(merged))
                    current_paragraph = []
            
            current_paragraph.append(text)
            last_block_bbox = bbox
        
        # End paragraph at page boundary if it has content
        if current_paragraph:
            merged = ' '.join(current_paragraph)
            if merged.strip():
                paragraphs.append(clean_text(merged))
            current_paragraph = []
            last_block_bbox = None
    
    # Handle any remaining text
    if current_paragraph:
        merged = ' '.join(current_paragraph)
        if merged.strip():
            paragraphs.append(clean_text(merged))
    
    return paragraphs

def outline_to_json(outline, reader, page_range=None, parent_is_last_list=None, fitz_doc=None, parent_end_page=None):
    if parent_is_last_list is None:
        parent_is_last_list = []
    result = []
    i = 0
    n = len(outline)
    while i < n:
        item = outline[i]
        node = {}
        if hasattr(item, 'title') or (isinstance(item, dict) and '/Title' in item):
            title = getattr(item, 'title', None) or item.get('/Title', None)
            # Skip nodes with no title or only whitespace in the title
            if not title or not title.strip():
                i += 1
                continue
            
            page_num = get_page_number(item, reader)
            if not page_num:
                i += 1
                continue

            node['title'] = title
            node['page'] = page_num

            # Find the next section's page number to determine this section's end
            next_page = None
            for j in range(i + 1, n):
                next_item = outline[j]
                if hasattr(next_item, 'title') or (isinstance(next_item, dict) and '/Title' in next_item):
                    next_page = get_page_number(next_item, reader)
                    break
            section_end = next_page - 1 if next_page else (parent_end_page if parent_end_page else len(reader.pages))
            if page_range:
                start, end = page_range
                section_start = max(page_num, start)
                section_end = min(section_end, end)
                if section_end < section_start:
                    i += 1
                    continue
            else:
                section_start = page_num
            node['title'] = title
            node['page'] = page_num
            
            # First get any paragraphs at the start of this node
            node_paragraphs = []
            if section_start <= section_end:
                node_paragraphs = extract_paragraphs_from_pages(reader, section_start, section_end, fitz_doc=fitz_doc)
            
            if i + 1 < n and isinstance(outline[i + 1], list):
                children = outline[i + 1]
                child_ranges = []
                child_json = []
                first_child_page = None
                
                # Find the first valid child page
                for child in children:
                    child_page = get_page_number(child, reader)
                    if child_page is not None:
                        first_child_page = child_page
                        break
                
                # If we found a first child page, only keep paragraphs before it
                if first_child_page is not None and node_paragraphs:
                    # Extract text up to the first child
                    node_paragraphs = extract_paragraphs_from_pages(reader, section_start, first_child_page - 1, fitz_doc=fitz_doc)
                
                # Process children
                for ci, child in enumerate(children):
                    child_page = get_page_number(child, reader)
                    if child_page is None:  # Skip children without valid page numbers
                        continue
                    next_child_page = None
                    for cj in range(ci + 1, len(children)):
                        next_child = children[cj]
                        if hasattr(next_child, 'title') or (isinstance(next_child, dict) and '/Title' in next_child):
                            next_child_page = get_page_number(next_child, reader)
                            break
                    child_end = next_child_page - 1 if next_child_page else section_end
                    if page_range:
                        child_start = max(child_page, section_start)
                        child_end = min(child_end, section_end)
                    else:
                        child_start = child_page
                    child_json.append(outline_to_json([child] + (children[ci+1:ci+2] if ci+1 < len(children) and isinstance(children[ci+1], list) else []), reader, (child_start, child_end), parent_is_last_list + [i == n - 1], fitz_doc=fitz_doc, parent_end_page=child_end))
                    child_ranges.append((child_start, child_end))
                
                if child_ranges:  # Only process if we have valid ranges
                    node['children'] = [item for sublist in child_json for item in (sublist if isinstance(sublist, list) else [sublist])]
                    # Only add gaps strictly between children
                    gaps = []
                    sorted_ranges = sorted(child_ranges)
                    for idx in range(len(sorted_ranges) - 1):
                        prev_end = sorted_ranges[idx][1]
                        next_start = sorted_ranges[idx + 1][0]
                        # Only add gap if strictly between children
                        if prev_end + 1 <= next_start - 1:
                            gap_start = prev_end + 1
                            gap_end = next_start - 1
                            # Enforce page_range boundaries
                            if page_range:
                                gap_start = max(gap_start, section_start)
                                gap_end = min(gap_end, section_end)
                            if gap_start <= gap_end:
                                gaps.append((gap_start, gap_end))
                    gap_paragraphs = []
                    for gap_start, gap_end in gaps:
                        if gap_start <= gap_end:
                            gap_paragraphs.extend(extract_paragraphs_from_pages(reader, gap_start, gap_end, fitz_doc=fitz_doc))
                        if gap_paragraphs:
                            # Add gap paragraphs after the node's own paragraphs
                            node_paragraphs.extend(gap_paragraphs)
                    i += 1  # skip children list
            else:
                if section_start <= section_end:
                    node_paragraphs = extract_paragraphs_from_pages(reader, section_start, section_end, fitz_doc=fitz_doc)
            
            # Add all collected paragraphs to the node
            if node_paragraphs:
                # Check for "Part II" in the introduction
                if title == "Introduction" and node_paragraphs and "PART II" in node_paragraphs[0]:
                    node['title'] = "Part II"
                    # The rest of the first paragraph is just the title, so skip it
                    if len(node_paragraphs) > 1:
                        node['paragraphs'] = node_paragraphs[1:]
                    else:
                        node['paragraphs'] = []
                else:
                    node['paragraphs'] = node_paragraphs
            
            result.append(node)
        i += 1
    return result

def process_pdf(pdf_path, mode='json', page_range=None):
    try:
        reader = PyPDF2.PdfReader(open(pdf_path, 'rb'))
        outline = reader.outline
        info = reader.metadata
        title = info.title if info and info.title else "Untitled"
        author = info.author if info and info.author else "Unknown"
        total_pages = len(reader.pages)
        fitz_doc = fitz.open(pdf_path)
        if mode == 'toc':
            print("\nDocument Information:")
            print("=" * 80)
            print(f"Title: {title}")
            print(f"Author: {author}")
            print(f"Total Pages: {total_pages}")
            print("\nPDF Outline Tree (matches PDF viewer):")
            print("=" * 80)
            print("-" * 80)
            tree_logger.info("PDF Outline Tree:")
            tree_logger.info("=" * 80)
            tree_logger.info("-" * 80)
            print_outline_tree(outline, reader)
            print("\n" + "=" * 80)
            print(f"\nOutline tree has been logged to pdf_outline.log and outline_tree.log")
            tree_logger.info("\n" + "=" * 80)
            fitz_doc.close()
            return
        if page_range:
            start, end = page_range
            json_outline = outline_to_json(outline, reader, page_range=(start, end), fitz_doc=fitz_doc)
        else:
            json_outline = outline_to_json(outline, reader, fitz_doc=fitz_doc)
        output = {
            'title': title,
            'author': author,
            'total_pages': total_pages,
            'outline': json_outline
        }
        out_file = 'book_structure.json' if not page_range else f'book_structure_{page_range[0]}_{page_range[1]}.json'
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nJSON structure written to {out_file}")
        fitz_doc.close()
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        return None

def parse_args():
    if len(sys.argv) < 2:
        print('Usage: python book_processer.py <path_to_pdf> [toc|[start,end]]')
        sys.exit(1)
    pdf_path = sys.argv[1]
    if len(sys.argv) == 3:
        arg = sys.argv[2]
        if arg == 'toc':
            return pdf_path, 'toc', None
        elif arg.startswith('[') and arg.endswith(']'):
            try:
                start, end = map(int, arg[1:-1].split(','))
                return pdf_path, 'json', (start, end)
            except Exception:
                print('Invalid page range format. Use [start,end]')
                sys.exit(1)
    return pdf_path, 'json', None

if __name__ == '__main__':
    pdf_path, mode, page_range = parse_args()
    process_pdf(pdf_path, mode, page_range)
