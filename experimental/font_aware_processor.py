import PyPDF2
import sys
import logging
import json
import re
import fitz  # PyMuPDF
import base64
from PyPDF2.generic import IndirectObject
from typing import Optional, List, Dict, Tuple
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('font_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_page_number(dest, reader):
    """Extract page number from PDF bookmark destination."""
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

def clean_text(text):
    """Clean text while preserving legitimate spacing and content."""
    if not text:
        return ''
    # Remove multiple spaces but preserve paragraph structure
    return re.sub(r'\s+', ' ', text).strip()

def analyze_text_case(text: str) -> str:
    """Analyze the case pattern of text."""
    if not text or not text.strip():
        return "empty"
    
    text = text.strip()
    
    # Remove common punctuation for analysis
    clean_text_for_analysis = re.sub(r'[^\w\s]', '', text)
    if not clean_text_for_analysis:
        return "punctuation_only"
    
    words = clean_text_for_analysis.split()
    if not words:
        return "empty"
    
    # Check if all words are uppercase
    if all(word.isupper() for word in words):
        return "all_caps"
    
    # Check if all words are lowercase
    if all(word.islower() for word in words):
        return "all_lower"
    
    # Check title case (first letter of each word capitalized)
    title_case_words = [word for word in words if len(word) > 0 and word[0].isupper() and (len(word) == 1 or word[1:].islower())]
    if len(title_case_words) == len(words):
        return "title_case"
    
    # Check sentence case (first word capitalized, rest lowercase unless proper nouns)
    if len(words) > 0 and words[0][0].isupper():
        lowercase_words = [word for word in words[1:] if word.islower()]
        if len(lowercase_words) >= len(words[1:]) * 0.7:  # 70% threshold for sentence case
            return "sentence_case"
    
    # Mixed case
    return "mixed_case"

def detect_list_item(text: str) -> Dict[str, any]:
    """Detect if text is a list item and what type."""
    text = text.strip()
    
    list_info = {
        "is_list_item": False,
        "list_type": None,
        "list_marker": None
    }
    
    if not text:
        return list_info
    
    # Numbered lists (1., 2., etc.)
    numbered_match = re.match(r'^(\d+)\.?\s+(.+)', text)
    if numbered_match:
        list_info.update({
            "is_list_item": True,
            "list_type": "numbered",
            "list_marker": numbered_match.group(1)
        })
        return list_info
    
    # Lettered lists (a., b., A., B., etc.)
    lettered_match = re.match(r'^([a-zA-Z])\.?\s+(.+)', text)
    if lettered_match:
        list_info.update({
            "is_list_item": True,
            "list_type": "lettered",
            "list_marker": lettered_match.group(1)
        })
        return list_info
    
    # Bullet points (•, -, *, etc.)
    bullet_match = re.match(r'^([•\-\*\+▪▫◦‣⁃])\s+(.+)', text)
    if bullet_match:
        list_info.update({
            "is_list_item": True,
            "list_type": "bullet",
            "list_marker": bullet_match.group(1)
        })
        return list_info
    
    # Roman numerals (i., ii., I., II., etc.)
    roman_match = re.match(r'^([ivxlcdmIVXLCDM]+)\.?\s+(.+)', text)
    if roman_match:
        list_info.update({
            "is_list_item": True,
            "list_type": "roman",
            "list_marker": roman_match.group(1)
        })
        return list_info
    
    return list_info

def detect_figure_reference(text: str) -> Dict[str, any]:
    """Detect if text contains figure references."""
    text = text.strip()
    
    figure_info = {
        "is_figure": False,
        "figure_type": None,
        "figure_number": None,
        "figure_title": None
    }
    
    # Common figure patterns
    figure_patterns = [
        r'^(FIGURE|Figure|Fig\.?)\s+(\d+(?:\.\d+)?)',  # FIGURE 7.6, Figure 1, Fig. 2.1
        r'^(TABLE|Table)\s+(\d+(?:\.\d+)?)',          # TABLE 1, Table 2.1
        r'^(CHART|Chart)\s+(\d+(?:\.\d+)?)',          # CHART 1
        r'^(DIAGRAM|Diagram)\s+(\d+(?:\.\d+)?)',      # DIAGRAM 1
        r'^(IMAGE|Image)\s+(\d+(?:\.\d+)?)',          # IMAGE 1
    ]
    
    for pattern in figure_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            figure_type = match.group(1).lower()
            figure_number = match.group(2)
            
            # Extract title if present (text after the figure reference)
            remaining_text = text[match.end():].strip()
            figure_title = remaining_text if remaining_text else None
            
            figure_info.update({
                "is_figure": True,
                "figure_type": figure_type,
                "figure_number": figure_number,
                "figure_title": figure_title
            })
            break
    
    return figure_info

def extract_images_from_page(fitz_doc, page_num: int) -> List[Dict]:
    """Extract all images from a specific page and convert to base64."""
    try:
        page = fitz_doc.load_page(page_num - 1)  # Convert to 0-indexed
        images = []
        image_list = page.get_images()
        
        logger.debug(f"Page {page_num}: Found {len(image_list)} images")
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image reference
                xref = img[0]
                logger.debug(f"Page {page_num}, Image {img_index}: Processing xref {xref}")
                
                # Try multiple methods to extract image
                img_data = None
                img_ext = "png"  # Default format
                width = None
                height = None
                
                # Method 1: extract_image (preferred)
                try:
                    pix_dict = fitz_doc.extract_image(xref)
                    img_data = pix_dict["image"]
                    img_ext = pix_dict["ext"]
                    logger.debug(f"Page {page_num}, Image {img_index}: Extracted via extract_image, format: {img_ext}, size: {len(img_data)} bytes")
                except Exception as e1:
                    logger.debug(f"Page {page_num}, Image {img_index}: extract_image failed: {str(e1)}")
                    
                    # Method 2: Create pixmap directly from xref
                    try:
                        pix = fitz.Pixmap(fitz_doc, xref)
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_ext = "png"
                            width = pix.width
                            height = pix.height
                            logger.debug(f"Page {page_num}, Image {img_index}: Extracted via Pixmap, size: {len(img_data)} bytes")
                        pix = None  # Free memory
                    except Exception as e2:
                        logger.debug(f"Page {page_num}, Image {img_index}: Pixmap method failed: {str(e2)}")
                        
                        # Method 3: Get image through page rendering (fallback)
                        try:
                            # Get image bbox from page
                            img_bbox = img[1:5] if len(img) > 4 else None  # x0, y0, x1, y1
                            if img_bbox:
                                # Render the specific area as image
                                clip = fitz.Rect(img_bbox)
                                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                                pix = page.get_pixmap(matrix=mat, clip=clip)
                                img_data = pix.tobytes("png")
                                img_ext = "png"
                                width = pix.width
                                height = pix.height
                                logger.debug(f"Page {page_num}, Image {img_index}: Extracted via page rendering, size: {len(img_data)} bytes")
                                pix = None
                        except Exception as e3:
                            logger.debug(f"Page {page_num}, Image {img_index}: Page rendering failed: {str(e3)}")
                
                if img_data and len(img_data) > 0:
                    # Convert to base64
                    img_base64 = base64.b64encode(img_data).decode()
                    
                    # Get dimensions if not already set
                    if width is None or height is None:
                        try:
                            temp_pix = fitz.Pixmap(img_data)
                            width = temp_pix.width
                            height = temp_pix.height
                            temp_pix = None
                        except:
                            width = width or 0
                            height = height or 0
                    
                    images.append({
                        "image_index": img_index,
                        "format": img_ext,
                        "base64": img_base64,
                        "width": width,
                        "height": height,
                        "size_bytes": len(img_data),
                        "extraction_method": "multiple_fallback"
                    })
                    
                    logger.info(f"Page {page_num}: Successfully extracted image {img_index} ({img_ext}, {len(img_data)} bytes)")
                else:
                    logger.warning(f"Page {page_num}, Image {img_index}: Failed to extract image data with all methods")
                
            except Exception as e:
                logger.warning(f"Page {page_num}, Image {img_index}: Unexpected error: {str(e)}")
                continue
        
        logger.info(f"Page {page_num}: Successfully extracted {len(images)} out of {len(image_list)} images")
        return images
        
    except Exception as e:
        logger.error(f"Error processing page {page_num} for images: {str(e)}")
        return []

def extract_paragraphs_from_pages(reader, start_page, end_page, fitz_doc=None):
    """Extract paragraphs using the EXACT proven logic from book_processer.py, then add font metadata."""
    if not fitz_doc:
        return []

    # First, use the EXACT same logic as book_processer.py to get clean paragraphs
    paragraphs_text = []
    current_paragraph = []
    last_block_bbox = None
    
    # Store all blocks with metadata for later font analysis
    all_blocks_with_metadata = []
    
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
            
            # Store block metadata for font analysis
            block_metadata = {
                'text': text,
                'bbox': bbox,
                'page': i + 1,
                'paragraph_index': len(paragraphs_text),  # Current paragraph being built
                'block_in_paragraph': len(current_paragraph)  # Position in current paragraph
            }
            all_blocks_with_metadata.append(block_metadata)
            
            # Start new paragraph if:
            # 1. Significant vertical gap
            # 2. Different indentation
            # 3. Previous paragraph ended with clear terminator
            if current_paragraph and last_block_bbox:
                vertical_gap = bbox[1] - last_block_bbox[3]  # y0 of current - y1 of last
                indent_difference = abs(bbox[0] - last_block_bbox[0])  # x0 difference
                last_text = current_paragraph[-1]  # EXACT same as book_processer.py
                
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
                        paragraphs_text.append(clean_text(merged))
                    current_paragraph = []
            
            current_paragraph.append(text)  # EXACT same as book_processer.py
            last_block_bbox = bbox
        
        # Smart page boundary handling: only continue paragraph if it makes sense
        if current_paragraph:
            # Check if we should end the paragraph at page boundary
            last_text = current_paragraph[-1].strip()
            should_end_paragraph = True
            
            # Look ahead to next page's first block to decide
            if i + 1 < end_page:  # There is a next page
                next_page = fitz_doc.load_page(i + 1)
                next_blocks = next_page.get_text("blocks")
                next_blocks.sort(key=lambda b: (b[1], b[0]))
                
                # Find first non-empty block on next page
                next_text = None
                for block in next_blocks:
                    text = block[4].strip()
                    if text:
                        next_text = text
                        break
                
                if next_text:
                    # Enhanced cross-page logic:
                    # Page breaks should NOT create paragraph breaks unless there's a logical reason
                    # A paragraph can contain multiple sentences across pages
                    
                    # Check if previous text ends with sentence punctuation
                    ends_with_sentence = re.search(r'[.!?]\s*$', last_text)
                    ends_with_colon = last_text.endswith(':')
                    ends_with_comma = last_text.endswith(',')
                    
                    # Check if next text starts a new logical unit
                    starts_with_lowercase = next_text[0].islower()
                    starts_with_list_marker = re.match(r'^(\d+\.|[a-zA-Z]\.|[•\-\*\+▪▫◦‣⁃]|[ivxlcdmIVXLCDM]+\.)\s+', next_text)
                    
                    # Check for figure patterns that should be combined
                    last_is_figure = re.match(r'^(FIGURE|Figure|Fig\.?|TABLE|Table|CHART|Chart|DIAGRAM|Diagram|IMAGE|Image)\s+\d+(?:\.\d+)?$', last_text.strip())
                    next_is_caption = not re.match(r'^(FIGURE|Figure|Fig\.?|TABLE|Table|CHART|Chart|DIAGRAM|Diagram|IMAGE|Image)\s+\d+', next_text.strip())
                    
                    # Decision logic for page boundaries:
                    if starts_with_list_marker:
                        # New list item = always new paragraph (clear logical break)
                        should_end_paragraph = True
                    elif last_is_figure and next_is_caption:
                        # FIGURE reference + caption = combine them
                        should_end_paragraph = False
                    elif not ends_with_sentence and starts_with_lowercase:
                        # No sentence ending + lowercase = definitely continuing sentence
                        should_end_paragraph = False
                    elif ends_with_comma and starts_with_lowercase:
                        # Comma + lowercase = definitely continuing sentence
                        should_end_paragraph = False
                    elif ends_with_colon:
                        # Colon = likely continuing with explanation/list/example
                        should_end_paragraph = False
                    else:
                        # Default: continue paragraph across page breaks
                        # Page breaks alone don't create paragraph breaks
                        should_end_paragraph = False
            
            if should_end_paragraph:
                merged = ' '.join(current_paragraph)
                if merged.strip():
                    paragraphs_text.append(clean_text(merged))
                current_paragraph = []
    
    # Handle any remaining text
    if current_paragraph:
        merged = ' '.join(current_paragraph)
        if merged.strip():
            paragraphs_text.append(clean_text(merged))
    
    # Now create paragraph objects with metadata
    # Map blocks to paragraphs by reconstructing the paragraph building process
    enhanced_paragraphs = []
    block_index = 0
    
    for para_text in paragraphs_text:
        # Count how many blocks contributed to this paragraph by reconstructing
        # We need to match the text content to figure out which blocks belong here
        para_blocks = []
        
        # Find blocks that could have contributed to this paragraph
        # This is approximate but should work for font analysis
        current_para_chars = 0
        target_chars = len(para_text.replace(' ', ''))  # Remove spaces for comparison
        
        while block_index < len(all_blocks_with_metadata) and current_para_chars < target_chars * 0.8:
            block = all_blocks_with_metadata[block_index]
            para_blocks.append(block)
            current_para_chars += len(block['text'].replace(' ', ''))
            block_index += 1
            
            # If we've matched the content reasonably well, stop
            if current_para_chars >= target_chars * 0.8:
                break
        
        if para_blocks:
            # Get page information
            pages = list(set([block['page'] for block in para_blocks]))
            
            # Detect list information
            list_info = detect_list_item(para_text)
            
            paragraph_info = {
                "text": para_text,
                "case_type": analyze_text_case(para_text),
                "pages": sorted(pages),
                "word_count": len(para_text.split()),
                "character_count": len(para_text),
                "block_count": len(para_blocks),
                "is_list_item": list_info["is_list_item"],
                "list_type": list_info["list_type"],
                "list_marker": list_info["list_marker"],
            }
            enhanced_paragraphs.append(paragraph_info)
        else:
            # Fallback for paragraphs without blocks
            list_info = detect_list_item(para_text)
            
            paragraph_info = {
                "text": para_text,
                "case_type": analyze_text_case(para_text),
                "pages": [start_page],  # Default to start page
                "word_count": len(para_text.split()),
                "character_count": len(para_text),
                "block_count": 0,
                "is_list_item": list_info["is_list_item"],
                "list_type": list_info["list_type"],
                "list_marker": list_info["list_marker"],
            }
            enhanced_paragraphs.append(paragraph_info)
    
    return enhanced_paragraphs

def get_font_info_for_text_range(fitz_doc, page_num, text_content, bbox_hint=None):
    """Get accurate font information for a specific text content within a page."""
    try:
        page = fitz_doc.load_page(page_num - 1)  # Convert to 0-indexed
        text_dict = page.get_text("dict")
        
        # Find text blocks that contain our target text
        target_words = text_content.lower().split()[:5]  # First 5 words for matching
        matching_spans = []
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = ""
                line_spans = []
                
                for span in line["spans"]:
                    span_text = span["text"]
                    line_text += span_text
                    line_spans.append(span)
                
                # Check if this line contains our target text
                line_words = line_text.lower().split()
                if len(line_words) >= 2:  # Need at least 2 words to match
                    # Check if any of our target words appear in this line
                    matches = sum(1 for word in target_words if word in line_words)
                    if matches >= min(2, len(target_words)):  # At least 2 matches or all target words
                        matching_spans.extend(line_spans)
        
        if not matching_spans:
            # Fallback: use all text spans from the page
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            matching_spans.append(span)
        
        if not matching_spans:
            return "unknown", 0.0, 0, False, False
        
        # Analyze the matching spans
        font_info = []
        total_chars = 0
        
        for span in matching_spans:
            span_text = span["text"].strip()
            if span_text:
                char_count = len(span_text)
                font_info.append({
                    'font': span["font"],
                    'size': span["size"],
                    'flags': span["flags"],
                    'chars': char_count
                })
                total_chars += char_count
        
        if not font_info:
            return "unknown", 0.0, 0, False, False
        
        # Calculate weighted averages
        font_counter = Counter()
        size_sum = 0
        flags_counter = Counter()
        
        for info in font_info:
            font_counter[info['font']] += info['chars']
            size_sum += info['size'] * info['chars']
            flags_counter[info['flags']] += info['chars']
        
        # Get dominant font and average size
        dominant_font = font_counter.most_common(1)[0][0]
        avg_size = size_sum / total_chars if total_chars > 0 else 0.0
        dominant_flags = flags_counter.most_common(1)[0][0]
        
        # Determine bold and italic from flags
        is_bold = bool(dominant_flags & 2**4)  # Bit 4 for bold
        is_italic = bool(dominant_flags & 2**1)  # Bit 1 for italic
        
        return dominant_font, avg_size, dominant_flags, is_bold, is_italic
        
    except Exception as e:
        logger.debug(f"Error getting font info for page {page_num}: {str(e)}")
        return "unknown", 0.0, 0, False, False

def add_font_metadata_to_paragraphs(paragraphs, fitz_doc):
    """Add font metadata to already-formed paragraphs by analyzing their content."""
    logger.info(f"Adding font metadata and figure detection to {len(paragraphs)} paragraphs...")
    
    for i, paragraph in enumerate(paragraphs):
        text = paragraph.get('text', '')
        pages = paragraph.get('pages', [])
        
        if not text.strip():
            continue
            
        # Get font metadata
        primary_page = pages[0] if pages else 1
        font_name, font_size, font_flags, is_bold, is_italic = get_font_info_for_text_range(fitz_doc, primary_page, text)
        
        # Detect lists
        list_info = detect_list_item(text)
        
        # Detect figures
        figure_info = detect_figure_reference(text)
        
        # Extract images if this relates to a figure and we have page info
        images = []
        if pages and (figure_info["is_figure"] or "FIGURE" in text.upper() or "TABLE" in text.upper() or "CHART" in text.upper()):
            # For figures, check current page AND surrounding pages (±2 pages)
            pages_to_check = set()
            for page_num in pages:
                # Add current page and surrounding pages
                for offset in range(-2, 3):  # -2, -1, 0, +1, +2
                    check_page = page_num + offset
                    if check_page > 0:  # Ensure valid page number
                        pages_to_check.add(check_page)
            
            # Extract images from all candidate pages
            for page_num in sorted(pages_to_check):
                page_images = extract_images_from_page(fitz_doc, page_num)
                if page_images:
                    # Add page info to each image for tracking
                    for img in page_images:
                        img["source_page"] = page_num
                    images.extend(page_images)
        
        # Update paragraph with all metadata
        paragraph.update({
            'font_name': font_name,
            'font_size': font_size,
            'font_flags': font_flags,
            'is_bold': is_bold,
            'is_italic': is_italic,
            'is_list_item': list_info['is_list_item'],
            'list_type': list_info['list_type'],
            'list_marker': list_info['list_marker'],
            'is_figure': figure_info['is_figure'],
            'figure_type': figure_info['figure_type'],
            'figure_number': figure_info['figure_number'],
            'figure_title': figure_info['figure_title'],
            'images': images
        })
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(paragraphs)} paragraphs...")
    
    # Log summary
    figure_count = sum(1 for p in paragraphs if p.get('is_figure', False))
    image_count = sum(len(p.get('images', [])) for p in paragraphs)
    logger.info(f"Font metadata complete. Found {figure_count} figures with {image_count} total images.")
    
    return paragraphs

def outline_to_json_with_fonts(outline, reader, page_range=None, parent_is_last_list=None, fitz_doc=None, parent_end_page=None):
    """Convert PDF outline to JSON with detailed font analysis for paragraphs."""
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
            
            # Extract paragraphs first using proven logic
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
                    
                    child_json.append(outline_to_json_with_fonts(
                        [child] + (children[ci+1:ci+2] if ci+1 < len(children) and isinstance(children[ci+1], list) else []), 
                        reader, 
                        (child_start, child_end), 
                        parent_is_last_list + [i == n - 1], 
                        fitz_doc=fitz_doc, 
                        parent_end_page=child_end
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
            
            # Add font metadata to the complete paragraphs
            if node_paragraphs:
                node_paragraphs = add_font_metadata_to_paragraphs(node_paragraphs, fitz_doc)
                node['paragraphs'] = node_paragraphs
                
                # Add summary statistics
                node['paragraph_count'] = len(node_paragraphs)
                node['total_words'] = sum(p['word_count'] for p in node_paragraphs)
                node['total_characters'] = sum(p['character_count'] for p in node_paragraphs)
                
                # Font analysis summary
                fonts_used = [p['font_name'] for p in node_paragraphs]
                sizes_used = [p['font_size'] for p in node_paragraphs]
                cases_used = [p['case_type'] for p in node_paragraphs]
                
                node['font_analysis'] = {
                    'unique_fonts': list(set(fonts_used)),
                    'font_distribution': dict(Counter(fonts_used)),
                    'size_range': [min(sizes_used), max(sizes_used)] if sizes_used else [0, 0],
                    'case_distribution': dict(Counter(cases_used)),
                    'avg_font_size': round(sum(sizes_used) / len(sizes_used), 2) if sizes_used else 0
                }
            
            result.append(node)
        i += 1
    
    return result

def add_node_paths(nodes: List[Dict], parent_path: Optional[str] = None):
    """Add hierarchical node paths to each section."""
    for i, node in enumerate(nodes):
        current_path = f"{parent_path}.{i + 1}" if parent_path else str(i + 1)
        node['node_path'] = current_path

        if 'children' in node and node['children']:
            add_node_paths(node['children'], current_path)

def process_pdf_with_fonts(pdf_path: str, page_range: Optional[Tuple[int, int]] = None) -> Dict:
    """Process PDF and extract structure with detailed font analysis."""
    try:
        reader = PyPDF2.PdfReader(open(pdf_path, 'rb'))
        outline = reader.outline
        info = reader.metadata
        
        title = info.title if info and info.title else "Untitled"
        author = info.author if info and info.author else "Unknown"
        total_pages = len(reader.pages)
        
        fitz_doc = fitz.open(pdf_path)
        
        logger.info(f"Processing PDF: {title} by {author} ({total_pages} pages)")
        
        if page_range:
            start, end = page_range
            json_outline = outline_to_json_with_fonts(outline, reader, page_range=(start, end), fitz_doc=fitz_doc)
            logger.info(f"Processing pages {start}-{end}")
        else:
            json_outline = outline_to_json_with_fonts(outline, reader, fitz_doc=fitz_doc)
        
        # Add node paths
        add_node_paths(json_outline)
        
        output = {
            'title': title,
            'author': author,
            'total_pages': total_pages,
            'processed_pages': f"{page_range[0]}-{page_range[1]}" if page_range else f"1-{total_pages}",
            'outline': json_outline,
            'processing_info': {
                'script_version': '2.1',
                'features': ['intelligent_paragraph_grouping', 'accurate_font_analysis', 'case_detection', 'list_detection', 'hierarchical_paths'],
                'approach': 'paragraph_first_then_accurate_font_metadata',
                'improvements': ['text_matching_font_detection', 'proper_bold_italic_detection', 'list_item_identification']
            }
        }
        
        # Generate filename
        if page_range:
            out_file = f'font_structure_{page_range[0]}_{page_range[1]}.json'
        else:
            out_file = 'font_structure.json'
        
        # Save output
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Font-aware structure written to {out_file}")
        fitz_doc.close()
        
        return output
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise

def parse_args():
    """Parse command line arguments."""
    if len(sys.argv) < 2:
        print('Usage: python font_aware_processor.py <path_to_pdf> [[start,end]]')
        print('Examples:')
        print('  python font_aware_processor.py book.pdf')
        print('  python font_aware_processor.py book.pdf [1,50]')
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    page_range = None
    
    if len(sys.argv) == 3:
        arg = sys.argv[2]
        if arg.startswith('[') and arg.endswith(']'):
            try:
                start, end = map(int, arg[1:-1].split(','))
                page_range = (start, end)
            except Exception:
                print('Invalid page range format. Use [start,end]')
                sys.exit(1)
    
    return pdf_path, page_range

if __name__ == '__main__':
    pdf_path, page_range = parse_args()
    result = process_pdf_with_fonts(pdf_path, page_range)
    
    # Print summary
    total_sections = len(result['outline'])
    total_paragraphs = sum(section.get('paragraph_count', 0) for section in result['outline'])
    
    print(f"\nProcessing Complete!")
    print(f"Sections extracted: {total_sections}")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Output saved to: {'font_structure.json' if not page_range else f'font_structure_{page_range[0]}_{page_range[1]}.json'}") 