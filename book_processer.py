import PyPDF2
import sys
import logging
import json
import re
import fitz  # PyMuPDF
from PyPDF2.generic import IndirectObject

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
    if not text:
        return ''
    text = re.sub(r'(?:[A-Za-z]\s){2,}[A-Za-z]', lambda m: m.group().replace(' ', ''), text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_paragraphs_from_pages(reader, start_page, end_page, fitz_doc=None):
    """Extract paragraphs from a range of pages using fitz (PyMuPDF) for block extraction."""
    paragraphs = []
    if fitz_doc:
        for i in range(start_page - 1, end_page):
            page = fitz_doc.load_page(i)
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip()
                if text:
                    cleaned = clean_text(text)
                    if cleaned:
                        paragraphs.append(cleaned)
        return paragraphs
    # fallback: just in case, but should not be used
    return []

def outline_to_json(outline, reader, page_range=None, parent_is_last_list=None, fitz_doc=None):
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
            page_num = get_page_number(item, reader)
            if not page_num:
                i += 1
                continue
            next_page = None
            for j in range(i + 1, n):
                next_item = outline[j]
                if hasattr(next_item, 'title') or (isinstance(next_item, dict) and '/Title' in next_item):
                    next_page = get_page_number(next_item, reader)
                    break
            if page_range:
                start, end = page_range
                section_end = next_page - 1 if next_page else len(reader.pages)
                if section_end < start:
                    i += 1
                    continue
                if page_num > end:
                    i += 1
                    continue
                node['title'] = title
                node['page'] = page_num
                if i + 1 < n and isinstance(outline[i + 1], list):
                    node['children'] = outline_to_json(outline[i + 1], reader, page_range, parent_is_last_list + [i == n - 1], fitz_doc=fitz_doc)
                    i += 1
                else:
                    para_start = max(page_num, start)
                    para_end = min(section_end, end)
                    if para_start <= para_end:
                        node['paragraphs'] = extract_paragraphs_from_pages(reader, para_start, para_end, fitz_doc=fitz_doc)
            else:
                node['title'] = title
                node['page'] = page_num
                if i + 1 < n and isinstance(outline[i + 1], list):
                    node['children'] = outline_to_json(outline[i + 1], reader, page_range, parent_is_last_list + [i == n - 1], fitz_doc=fitz_doc)
                    i += 1
                else:
                    para_start = page_num
                    para_end = next_page - 1 if next_page else len(reader.pages)
                    if para_start <= para_end:
                        node['paragraphs'] = extract_paragraphs_from_pages(reader, para_start, para_end, fitz_doc=fitz_doc)
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
