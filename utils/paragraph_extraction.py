"""
Paragraph Extraction Utilities

Extracts and cleans text paragraphs from PDF pages using PyMuPDF.
Smart merging logic to avoid awkward splits at page boundaries.
"""

import re
import logging
from typing import List
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def clean_text(text):
    """Clean text while preserving legitimate spacing and content."""
    if not text:
        return ''
    
    # Remove common PDF watermarks/sources
    watermarks = [
        r'OceanofPDF\.com',
        r'z-lib\.org',
        r'libgen\.is',
        r'PDF Drive',
        r'Free PDF Books',
        r'Download PDF',
        r'www\.[a-zA-Z0-9.-]+\.com',
    ]
    
    for watermark in watermarks:
        text = re.sub(watermark, '', text, flags=re.IGNORECASE)
    
    # Remove multiple spaces but preserve paragraph structure
    cleaned = re.sub(r'\s+', ' ', text).strip()
    
    # Return None if text is empty after cleaning (so it gets filtered out)
    return cleaned if cleaned else None


def extract_paragraphs_from_pages(reader, start_page, end_page, fitz_doc=None):
    """
    Extract paragraphs from a range of pages using PyMuPDF.
    
    Uses robust paragraph detection based on sentence terminators and capitalization:
    - Merges lines that are clearly continuations (lowercase start + no sentence terminator before)
    - Splits on actual paragraph boundaries (sentence terminator + capital, or significant gaps)
    - Each paragraph is a complete, standalone unit
    
    Args:
        reader: PyPDF2 reader object (not used currently, kept for compatibility)
        start_page: Starting page number (1-indexed)
        end_page: Ending page number (1-indexed, inclusive)
        fitz_doc: PyMuPDF document object
        
    Returns:
        List of paragraph strings, each representing a complete paragraph
    """
    if not fitz_doc:
        return []

    # First, collect all text lines with their positions
    text_lines = []
    
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
            text_lines.append({
                'text': text,
                'bbox': bbox,
                'page': i + 1
            })
    
    if not text_lines:
        return []
    
    # Sentence terminators that indicate potential paragraph breaks
    SENTENCE_TERMINATORS = r'[.!?:;]'
    
    # Merge lines and detect paragraph boundaries
    paragraphs = []
    current_paragraph_lines = []
    
    for i, line_data in enumerate(text_lines):
        text = line_data['text']
        bbox = line_data['bbox']
        
        # Check if this should be merged with previous line or start new paragraph
        is_continuation = False
        
        if current_paragraph_lines:
            last_line_data = current_paragraph_lines[-1]
            last_text = last_line_data['text']
            last_bbox = last_line_data['bbox']
            
            # Check vertical gap (significant gap = new paragraph)
            vertical_gap = bbox[1] - last_bbox[3]  # y0 of current - y1 of last
            line_height = bbox[3] - bbox[1]
            significant_gap = vertical_gap > 1.5 * line_height
            
            # Check indentation change (significant indent = new paragraph)
            indent_difference = abs(bbox[0] - last_bbox[0])
            significant_indent = indent_difference > 20
            
            # Check if last line ends with sentence terminator
            last_ends_with_terminator = bool(re.search(SENTENCE_TERMINATORS + r'\s*$', last_text))
            
            # Check if current line starts with lowercase (after trimming)
            text_stripped = text.lstrip()
            starts_lowercase = text_stripped and text_stripped[0].islower()
            
            # Check for quote after sentence terminator
            last_ends_with_quote_after_term = (
                len(last_text) > 2 and 
                last_text[-1] == '"' and 
                last_text[-2] in '.!?:;'
            )
            
            # It's a continuation if:
            # - Starts with lowercase AND previous doesn't end with terminator
            # - AND no significant gap or indent change
            if (starts_lowercase and 
                not last_ends_with_terminator and 
                not last_ends_with_quote_after_term and
                not significant_gap and 
                not significant_indent):
                is_continuation = True
        
        if is_continuation:
            # Merge with previous line
            current_paragraph_lines.append(line_data)
        else:
            # Start new paragraph (save previous if exists)
            if current_paragraph_lines:
                # Merge all lines in current paragraph
                merged_text = ' '.join(line['text'] for line in current_paragraph_lines)
                cleaned = clean_text(merged_text)
                if cleaned:  # Only add if not None/empty
                    paragraphs.append(cleaned)
            
            # Start new paragraph
            current_paragraph_lines = [line_data]
    
    # Handle final paragraph
    if current_paragraph_lines:
        merged_text = ' '.join(line['text'] for line in current_paragraph_lines)
        cleaned = clean_text(merged_text)
        if cleaned:  # Only add if not None/empty
            paragraphs.append(cleaned)
    
    return paragraphs

