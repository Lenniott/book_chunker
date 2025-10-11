"""
Outline/TOC Processing Utilities

Utilities for working with PDF bookmarks and table of contents.
"""

import logging
from PyPDF2.generic import IndirectObject

logger = logging.getLogger(__name__)


def get_page_number(dest, reader):
    """
    Extract page number from PDF bookmark destination.
    
    Args:
        dest: Bookmark destination object
        reader: PyPDF2 reader object
        
    Returns:
        Page number (1-indexed) or None if not found
    """
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
    """
    Print a tree view of the PDF outline/bookmarks.
    
    Args:
        outline: PDF outline structure
        reader: PyPDF2 reader object
        level: Current depth level (for indentation)
        parent_is_last_list: List tracking if parent nodes are last in their level
    """
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
            if i + 1 < n and isinstance(outline[i + 1], list):
                print_outline_tree(outline[i + 1], reader, level + 1, parent_is_last_list + [is_last])
                i += 1
        i += 1

