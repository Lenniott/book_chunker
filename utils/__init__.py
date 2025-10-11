"""
Book Chunker Utilities

This package provides reusable components for PDF extraction:
- image_extraction: Extract and encode images from PDFs
- paragraph_extraction: Extract and clean text paragraphs
- outline_processing: Process PDF bookmarks and TOC
"""

from .image_extraction import extract_images_from_page
from .paragraph_extraction import extract_paragraphs_from_pages, clean_text
from .outline_processing import get_page_number, print_outline_tree

__all__ = [
    'extract_images_from_page',
    'extract_paragraphs_from_pages',
    'clean_text',
    'get_page_number',
    'print_outline_tree',
]

