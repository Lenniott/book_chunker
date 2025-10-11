#!/Users/benjamin/Desktop/CODE/book_chunker/venv/bin/python3
"""
Book Extractor V2 - Hybrid Bookmark + Font Analysis Extraction

Uses proven two-phase algorithm:
1. Bookmarks provide reliable chapter structure
2. Font analysis detects sections/subsections within each chapter
3. Outputs properly nested hierarchy with paragraphs

Based on validated test_hybrid_extraction.py
"""

import PyPDF2
import sys
import logging
import json
import argparse
import fitz  # PyMuPDF
import re
from typing import Optional, Tuple, List, Dict, Any
from collections import Counter

# Import utilities
from utils.paragraph_extraction import extract_paragraphs_from_pages
from utils.outline_processing import get_page_number, print_outline_tree
from utils.image_extraction import extract_images_from_page

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('book_extraction_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def clean_watermarks(text):
    """Remove common PDF watermarks from text."""
    if not text:
        return None
    
    watermarks = [
        r'OceanofPDF\.com',
        r'z-lib\.org', 
        r'libgen\.is',
        r'PDF Drive',
        r'Free PDF Books',
        r'Download PDF',
        r'www\.[a-zA-Z0-9.-]+\.com',
    ]
    
    cleaned = text
    for watermark in watermarks:
        cleaned = re.sub(watermark, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Return None if empty after cleaning
    return cleaned if cleaned else None


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


def extract_sections_from_page_range(fitz_doc, start_page, end_page, extract_images=False):
    """
    Extract hierarchical sections from a page range using font analysis.
    
    Returns nested structure with sections, subsections, and paragraphs.
    """
    # Collect all lines in this range
    all_lines = []
    
    for page_num in range(start_page - 1, end_page):
        page = fitz_doc.load_page(page_num)
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
    
    if not all_lines:
        return []
    
    # Phase 1: Group by font SIZE (merge consecutive same-size lines)
    size_groups = []
    i = 0
    
    while i < len(all_lines):
        current = all_lines[i]
        combined_text = current['text']
        line_count = 1
        first_line_idx = i
        pg = current['page']
        
        j = i + 1
        while j < len(all_lines):
            next_line = all_lines[j]
            if (next_line['size'] == current['size'] and 
                next_line['page'] - pg <= 1):
                combined_text += " " + next_line['text']
                pg = next_line['page']
                line_count += 1
                j += 1
            else:
                break
        
        size_groups.append({
            'page': current['page'],
            'size': current['size'],
            'text': combined_text,
            'line_count': line_count,
            'first_line_idx': first_line_idx
        })
        
        i = j
    
    # Identify body text (sizes where max group >= 7 lines)
    max_group_per_size = {}
    for group in size_groups:
        size = group['size']
        if size not in max_group_per_size:
            max_group_per_size[size] = 0
        max_group_per_size[size] = max(max_group_per_size[size], group['line_count'])
    
    body_text_sizes = {size for size, max_lines in max_group_per_size.items() if max_lines >= 7}
    max_body_size = max(body_text_sizes) if body_text_sizes else 0
    
    # Filter heading candidates (> body text size, <= 2 lines)
    heading_candidates = []
    for group in size_groups:
        if group['size'] <= max_body_size:
            continue
        if group['line_count'] > 2:
            continue
        
        first_line = all_lines[group['first_line_idx']]
        
        # Clean watermarks from heading text
        cleaned_text = clean_watermarks(group['text'])
        if cleaned_text:  # Only add if not None/empty after cleaning
            heading_candidates.append({
                'page': group['page'],
                'size': first_line['size'],
                'bold': first_line['bold'],
                'case': first_line['case'],
                'text': cleaned_text,
                'full_pattern': (first_line['size'], first_line['bold'], first_line['case'])
            })
    
    if not heading_candidates:
        # No subsections found, return all text as paragraphs in content array
        paragraphs = extract_paragraphs_from_pages(None, start_page, end_page, fitz_doc)
        return [{
            'content': paragraphs
        }]
    
    # Phase 2: Build hierarchy using pattern stack
    pattern_to_level = {}
    pattern_stack = []
    sections_flat = []
    
    for heading in heading_candidates:
        pattern = heading['full_pattern']
        
        if pattern in pattern_to_level:
            target_level = pattern_to_level[pattern]
            while len(pattern_stack) > target_level + 1:
                pattern_stack.pop()
            current_level = target_level
        else:
            current_level = len(pattern_stack)
            pattern_to_level[pattern] = current_level
            pattern_stack.append(pattern)
        
        sections_flat.append({
            'title': heading['text'],
            'page': heading['page'],
            'level': current_level,
            'pattern': heading['full_pattern']
        })
    
    # Phase 3: Build nested structure with unified content array
    root_sections = []
    stack = []  # (section_dict, level)
    all_sections_flat_for_paras = []  # Track all sections for paragraph extraction
    
    for section_data in sections_flat:
        level = section_data['level']
        section = {
            'title': section_data['title'],
            'page': section_data['page'],
            'content': []
        }
        
        # Pop stack until we find the parent level
        while stack and stack[-1][1] >= level:
            stack.pop()
        
        # Add to parent or root
        if stack:
            parent, parent_level = stack[-1]
            parent['content'].append(section)
        else:
            root_sections.append(section)
        
        # Push current section onto stack
        stack.append((section, level))
        
        # Track for paragraph extraction
        all_sections_flat_for_paras.append(section)
    
    # Phase 4: Build content arrays with proper ordering and flatten redundant structures
    # For each section, we need to insert paragraphs before the first child subsection
    for i, section in enumerate(all_sections_flat_for_paras):
        section_start = section['page']
        
        # Find section end (next sibling or parent's end)
        section_end = end_page
        
        # Look for next section at same or higher level
        for j in range(i + 1, len(all_sections_flat_for_paras)):
            next_section = all_sections_flat_for_paras[j]
            # If next section exists, this section ends before it
            section_end = next_section['page'] - 1
            break
        
        # Find the first child subsection to determine where paragraphs should go
        first_child_page = None
        for child in section.get('content', []):
            if isinstance(child, dict) and 'page' in child:
                if first_child_page is None or child['page'] < first_child_page:
                    first_child_page = child['page']
        
        # Determine paragraph extraction range
        if first_child_page:
            para_end = first_child_page - 1
        else:
            para_end = section_end
        
        # Extract paragraphs for this section (before first child)
        if section_start <= para_end:
            paragraphs = extract_paragraphs_from_pages(
                None,  # reader not needed
                section_start,
                para_end,
                fitz_doc
            )
            
            # Insert paragraphs at the beginning of content array
            # They come before any subsections
            for paragraph in reversed(paragraphs):
                section['content'].insert(0, paragraph)
        
        # Extract images if requested
        if extract_images:
            section_images = []
            for page_num in range(section_start, section_end + 1):
                page_images = extract_images_from_page(fitz_doc, page_num)
                if page_images:
                    section_images.extend(page_images)
            
            if section_images:
                section['images'] = section_images
    
    # Phase 5: Clean up redundant content and merge duplicate structures
    def clean_content(content_list, parent_title="", parent_page=None):
        """Clean content arrays to remove redundant title strings and merge duplicate structures."""
        if not content_list:
            return content_list
            
        cleaned = []
        for item in content_list:
            if isinstance(item, str):
                # Clean string content - remove redundant title strings
                parent_title_clean = re.sub(r'^\d+\s*', '', parent_title.lower()).strip()
                item_clean = item.lower().strip()
                
                # Skip if this string is redundant with parent title
                if (item_clean == parent_title_clean or 
                    item_clean == parent_title.lower() or
                    (len(item_clean) < 10 and item_clean in parent_title.lower())):
                    continue
                    
                cleaned.append(item)
                
            elif isinstance(item, dict) and 'content' in item:
                item_title = item.get('title', '')
                
                # Check if subsection title is redundant with parent title
                parent_title_clean = re.sub(r'^\d+\s*', '', parent_title.lower()).strip()
                item_title_clean = re.sub(r'^\d+\s*', '', item_title.lower()).strip()
                
                # Check if this is a redundant subsection (same page, similar title)
                is_redundant = (
                    item.get('page') == parent_page and
                    parent_title_clean and item_title_clean and 
                    (item_title_clean == parent_title_clean or 
                     item_title_clean in parent_title_clean or
                     parent_title_clean in item_title_clean)
                )
                
                if is_redundant:
                    # Merge child's content directly into parent, skipping the redundant subsection
                    child_content = clean_content(item.get('content', []), item_title, item.get('page'))
                    cleaned.extend(child_content)
                else:
                    # Recursively clean children
                    item['content'] = clean_content(item.get('content', []), item_title, item.get('page'))
                    cleaned.append(item)
            else:
                cleaned.append(item)
        
        return cleaned
    
    # Apply cleaning to all root sections
    for section in root_sections:
        section['content'] = clean_content(section.get('content', []), section.get('title', ''), section.get('page'))
    
    return root_sections


def process_pdf_hybrid(pdf_path, output_file=None, extract_images=False, use_hierarchy=True):
    """
    Process PDF using hybrid bookmark + font analysis approach.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(open(pdf_path, 'rb'))
        fitz_doc = fitz.open(pdf_path)
        
        outline = pdf_reader.outline
        info = pdf_reader.metadata
        
        title = info.title if info and info.title else "Untitled"
        author = info.author if info and info.author else "Unknown"
        total_pages = len(pdf_reader.pages)
        
        logger.info(f"Processing: {title} by {author}")
        logger.info(f"Total pages: {total_pages}")
        logger.info(f"Hierarchy detection: {'ENABLED' if use_hierarchy else 'DISABLED'}")
        
        # Extract top-level chapters from bookmarks
        chapters = []
        i = 0
        while i < len(outline):
            item = outline[i]
            if hasattr(item, 'title') or (isinstance(item, dict) and '/Title' in item):
                chapter_title = getattr(item, 'title', None) or item.get('/Title', None)
                page_num = get_page_number(item, pdf_reader)
                
                if page_num and chapter_title and chapter_title.strip():
                    # Clean watermarks from chapter title
                    chapter_title = clean_watermarks(chapter_title)
                    if not chapter_title:  # Skip if title becomes empty after cleaning
                        continue
                    # Find next chapter to determine page range
                    next_page = None
                    for j in range(i + 1, len(outline)):
                        next_item = outline[j]
                        if hasattr(next_item, 'title') or (isinstance(next_item, dict) and '/Title' in next_item):
                            next_page = get_page_number(next_item, pdf_reader)
                            if next_page:
                                break
                    
                    end_page = next_page - 1 if next_page else total_pages
                    
                    chapter = {
                        'title': chapter_title,
                        'page': page_num,
                        'start_page': page_num,
                        'end_page': end_page
                    }
                    
                    # Extract content if hierarchy detection enabled
                    if use_hierarchy:
                        content = extract_sections_from_page_range(
                            fitz_doc, 
                            page_num, 
                            end_page,
                            extract_images
                        )
                        if content:
                            chapter['content'] = content
                    else:
                        # Simple paragraph extraction
                        paragraphs = extract_paragraphs_from_pages(
                            pdf_reader, 
                            page_num, 
                            end_page, 
                            fitz_doc
                        )
                        if paragraphs:
                            chapter['content'] = paragraphs
                    
                    chapters.append(chapter)
            i += 1
        
        output = {
            'title': title,
            'author': author,
            'total_pages': total_pages,
            'outline': chapters,
            'extraction_info': {
                'source': pdf_path,
                'hierarchy_detection': use_hierarchy,
                'images_extracted': extract_images,
                'extractor_version': '2.0-hybrid'
            }
        }
        
        # Determine output filename
        if not output_file:
            output_file = 'book_structure_v2.json'
        
        # Save JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ Extraction complete!")
        logger.info(f"📄 Output: {output_file}")
        logger.info(f"📊 Chapters: {len(chapters)}")
        
        fitz_doc.close()
        return output
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return None


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Extract structured content from PDF files with hierarchy detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction with hierarchy
  python book_extractor_v2.py book.pdf
  
  # Without hierarchy detection (simple mode)
  python book_extractor_v2.py book.pdf --no-hierarchy
  
  # With images
  python book_extractor_v2.py book.pdf --extract-images
  
  # Custom output file
  python book_extractor_v2.py book.pdf --output my_book.json

Features:
  • Hybrid approach: Bookmarks + Font analysis
  • Automatic hierarchy detection
  • Proper nesting of sections/subsections
  • Works across different book formats
        """
    )
    
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--output', '-o', help='Output JSON filename')
    parser.add_argument('--extract-images', '-i', action='store_true', 
                       help='Extract and embed images (increases file size)')
    parser.add_argument('--no-hierarchy', action='store_true',
                       help='Disable hierarchy detection (simple extraction)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process PDF
    process_pdf_hybrid(
        args.pdf_path, 
        output_file=args.output, 
        extract_images=args.extract_images,
        use_hierarchy=not args.no_hierarchy
    )


if __name__ == '__main__':
    main()

