"""
Image Extraction Utilities

Extracts images from PDF pages and converts them to base64 encoding.
"""

import base64
import logging
from typing import List, Dict
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_images_from_page(fitz_doc, page_num: int) -> List[Dict]:
    """
    Extract all images from a specific page and convert to base64.
    
    Args:
        fitz_doc: PyMuPDF document object
        page_num: Page number (1-indexed)
        
    Returns:
        List of image dictionaries with base64 data, format, dimensions, etc.
    """
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

