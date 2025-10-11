#!/Users/benjamin/Desktop/CODE/book_chunker/venv/bin/python3
"""
Book Extractor - Main PDF to JSON extraction tool

Extracts structured content from PDFs using:
1. PDF bookmarks (TOC) for reliable top-level structure
2. Optional font analysis for deeper hierarchy detection
3. Image extraction and embedding
4. Smart paragraph grouping

Always outputs JSON. Use vectorize.py to load into a vector database.
"""

import PyPDF2
import sys
import logging
import json
import argparse
import fitz  # PyMuPDF
from typing import Optional, Tuple, List, Dict, Any

# Import utilities
from utils.paragraph_extraction import extract_paragraphs_from_pages, clean_text
from utils.outline_processing import get_page_number, print_outline_tree
from utils.image_extraction import extract_images_from_page

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('book_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def outline_to_json(outline, reader, page_range=None, parent_is_last_list=None, 
                    fitz_doc=None, parent_end_page=None, extract_images=False):
    """
    Convert PDF outline to hierarchical JSON structure.
    
    Args:
        outline: PDF outline/bookmark structure
        reader: PyPDF2 reader object
        page_range: Optional tuple of (start_page, end_page) to limit extraction
        parent_is_last_list: Internal tracking for tree structure
        fitz_doc: PyMuPDF document for text and image extraction
        parent_end_page: End page of parent section
        extract_images: Whether to extract and embed images
        
    Returns:
        List of section dictionaries with hierarchical structure
    """
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
            
            # Skip nodes with no title or only whitespace
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
            
            # Extract paragraphs for this node
            node_paragraphs = []
            if section_start <= section_end:
                node_paragraphs = extract_paragraphs_from_pages(reader, section_start, section_end, fitz_doc=fitz_doc)
            
            # Handle children (nested sections)
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
                    node_paragraphs = extract_paragraphs_from_pages(reader, section_start, first_child_page - 1, fitz_doc=fitz_doc)
                
                # Process children recursively
                for ci, child in enumerate(children):
                    child_page = get_page_number(child, reader)
                    if child_page is None:
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
                    
                    child_json.append(outline_to_json(
                        [child] + (children[ci+1:ci+2] if ci+1 < len(children) and isinstance(children[ci+1], list) else []), 
                        reader, 
                        (child_start, child_end), 
                        parent_is_last_list + [i == n - 1], 
                        fitz_doc=fitz_doc, 
                        parent_end_page=child_end,
                        extract_images=extract_images
                    ))
                    child_ranges.append((child_start, child_end))
                
                if child_ranges:
                    node['children'] = [item for sublist in child_json for item in (sublist if isinstance(sublist, list) else [sublist])]
                    
                    # Handle gaps between children
                    gaps = []
                    sorted_ranges = sorted(child_ranges)
                    for idx in range(len(sorted_ranges) - 1):
                        prev_end = sorted_ranges[idx][1]
                        next_start = sorted_ranges[idx + 1][0]
                        
                        if prev_end + 1 <= next_start - 1:
                            gap_start = prev_end + 1
                            gap_end = next_start - 1
                            
                            if page_range:
                                gap_start = max(gap_start, section_start)
                                gap_end = min(gap_end, section_end)
                            
                            if gap_start <= gap_end:
                                gaps.append((gap_start, gap_end))
                    
                    # Extract paragraphs from gaps
                    gap_paragraphs = []
                    for gap_start, gap_end in gaps:
                        if gap_start <= gap_end:
                            gap_paragraphs.extend(extract_paragraphs_from_pages(reader, gap_start, gap_end, fitz_doc=fitz_doc))
                    
                    if gap_paragraphs:
                        node_paragraphs.extend(gap_paragraphs)
                
                i += 1  # Skip children list
            else:
                if section_start <= section_end:
                    node_paragraphs = extract_paragraphs_from_pages(reader, section_start, section_end, fitz_doc=fitz_doc)
            
            # Add paragraphs to the node
            if node_paragraphs:
                node['paragraphs'] = node_paragraphs
            
            # Extract images if requested
            if extract_images and fitz_doc:
                section_images = []
                for page_num in range(section_start, section_end + 1):
                    page_images = extract_images_from_page(fitz_doc, page_num)
                    if page_images:
                        section_images.extend(page_images)
                
                if section_images:
                    node['images'] = section_images
                    logger.info(f"Section '{title}': Extracted {len(section_images)} images")
            
            result.append(node)
        i += 1
    
    return result


def process_pdf(pdf_path, mode='json', page_range=None, output_file=None, extract_images=False):
    """
    Process PDF and extract structured content.
    
    Args:
        pdf_path: Path to PDF file
        mode: 'json' for extraction, 'toc' to just show table of contents
        page_range: Optional tuple of (start, end) pages
        output_file: Optional output filename
        extract_images: Whether to extract and embed images
        
    Returns:
        Dictionary with extracted content or None on error
    """
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
            print("\nPDF Outline Tree (Table of Contents):")
            print("=" * 80)
            print_outline_tree(outline, reader)
            print("\n" + "=" * 80)
            fitz_doc.close()
            return None
        
        # Extract structure
        logger.info(f"Processing: {title} by {author}")
        logger.info(f"Total pages: {total_pages}")
        if page_range:
            logger.info(f"Page range: {page_range[0]}-{page_range[1]}")
        if extract_images:
            logger.info("Image extraction: ENABLED")
        
        if page_range:
            start, end = page_range
            json_outline = outline_to_json(outline, reader, page_range=(start, end), fitz_doc=fitz_doc, extract_images=extract_images)
        else:
            json_outline = outline_to_json(outline, reader, fitz_doc=fitz_doc, extract_images=extract_images)
        
        output = {
            'title': title,
            'author': author,
            'total_pages': total_pages,
            'outline': json_outline,
            'extraction_info': {
                'source': pdf_path,
                'page_range': f"{page_range[0]}-{page_range[1]}" if page_range else f"1-{total_pages}",
                'images_extracted': extract_images,
                'extractor_version': '2.0-hybrid'
            }
        }
        
        # Determine output filename
        if not output_file:
            if page_range:
                output_file = f'book_structure_{page_range[0]}_{page_range[1]}.json'
            else:
                output_file = 'book_structure.json'
        
        # Save JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ Extraction complete!")
        logger.info(f"📄 Output: {output_file}")
        logger.info(f"📊 Sections: {len(json_outline)}")
        
        # Count total paragraphs
        def count_paragraphs(nodes):
            count = 0
            for node in nodes:
                count += len(node.get('paragraphs', []))
                if 'children' in node:
                    count += count_paragraphs(node['children'])
            return count
        
        total_paragraphs = count_paragraphs(json_outline)
        logger.info(f"📝 Paragraphs: {total_paragraphs}")
        
        if extract_images:
            def count_images(nodes):
                count = 0
                for node in nodes:
                    count += len(node.get('images', []))
                    if 'children' in node:
                        count += count_images(node['children'])
                return count
            
            total_images = count_images(json_outline)
            logger.info(f"🖼️  Images: {total_images}")
        
        fitz_doc.close()
        return output
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return None


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Extract structured content from PDF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  python book_extractor.py book.pdf
  
  # Extract with images
  python book_extractor.py book.pdf --extract-images
  
  # Extract specific page range
  python book_extractor.py book.pdf --pages 10-50
  
  # Custom output file
  python book_extractor.py book.pdf --output my_book.json
  
  # Show table of contents only
  python book_extractor.py book.pdf --toc

Output:
  Creates a JSON file with hierarchical structure matching the book's TOC.
  Use vectorize.py to load the JSON into a vector database.
        """
    )
    
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--pages', '-p', help='Page range to extract (e.g., "10-50")')
    parser.add_argument('--output', '-o', help='Output JSON filename')
    parser.add_argument('--extract-images', '-i', action='store_true', 
                       help='Extract and embed images (increases file size)')
    parser.add_argument('--toc', action='store_true', 
                       help='Show table of contents and exit')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse page range
    page_range = None
    if args.pages:
        try:
            parts = args.pages.split('-')
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
                page_range = (start, end)
            else:
                logger.error("Invalid page range format. Use: START-END (e.g., 10-50)")
                sys.exit(1)
        except ValueError:
            logger.error("Invalid page range format. Use: START-END (e.g., 10-50)")
            sys.exit(1)
    
    # Process PDF
    mode = 'toc' if args.toc else 'json'
    process_pdf(args.pdf_path, mode=mode, page_range=page_range, 
                output_file=args.output, extract_images=args.extract_images)


if __name__ == '__main__':
    main()

