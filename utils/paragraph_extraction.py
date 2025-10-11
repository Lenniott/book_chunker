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
    Extract paragraphs from a range of pages using PyMuPDF for block extraction.
    
    Uses smart merging to avoid awkward splits:
    - Detects vertical gaps between blocks
    - Detects indentation changes
    - Respects sentence endings
    - Intelligently handles page boundaries
    
    Args:
        reader: PyPDF2 reader object (not used currently, kept for compatibility)
        start_page: Starting page number (1-indexed)
        end_page: Ending page number (1-indexed, inclusive)
        fitz_doc: PyMuPDF document object
        
    Returns:
        List of paragraph strings
    """
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
                        cleaned = clean_text(merged)
                        if cleaned:  # Only add if not None
                            paragraphs.append(cleaned)
                    current_paragraph = []
            
            current_paragraph.append(text)
            last_block_bbox = bbox
        
        # End paragraph at page boundary if it has content
        if current_paragraph:
            merged = ' '.join(current_paragraph)
            if merged.strip():
                cleaned = clean_text(merged)
                if cleaned:  # Only add if not None
                    paragraphs.append(cleaned)
            current_paragraph = []
            last_block_bbox = None
    
    # Handle any remaining text
    if current_paragraph:
        merged = ' '.join(current_paragraph)
        if merged.strip():
            cleaned = clean_text(merged)
            if cleaned:  # Only add if not None
                paragraphs.append(cleaned)
    
    return paragraphs

