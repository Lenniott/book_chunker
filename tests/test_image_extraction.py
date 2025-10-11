#!/usr/bin/env python3
"""
Test script to debug image extraction on specific pages.
"""

import fitz
import logging
from font_aware_processor import extract_images_from_page

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_image_extraction_on_page(pdf_path: str, page_num: int):
    """Test image extraction on a specific page."""
    print(f"🔍 Testing image extraction on page {page_num}")
    
    try:
        # Open PDF
        fitz_doc = fitz.open(pdf_path)
        
        # Load the specific page
        page = fitz_doc.load_page(page_num - 1)  # 0-indexed
        
        # Get basic page info
        print(f"📄 Page {page_num} info:")
        print(f"   Size: {page.rect}")
        
        # Check for images using different methods
        image_list = page.get_images()
        print(f"   Images found by get_images(): {len(image_list)}")
        
        # Show image details
        for i, img in enumerate(image_list):
            print(f"   Image {i}: xref={img[0]}, bbox={img[1:5] if len(img) > 4 else 'N/A'}")
        
        # Test our extraction function
        print(f"\n🎯 Testing our extraction function:")
        extracted_images = extract_images_from_page(fitz_doc, page_num)
        
        print(f"✅ Successfully extracted {len(extracted_images)} images")
        for i, img in enumerate(extracted_images):
            print(f"   Image {i}: {img['format']}, {img['size_bytes']} bytes, {img['width']}x{img['height']}")
            print(f"              Base64 preview: {img['base64'][:50]}...")
        
        # Check for text containing FIGURE
        text_dict = page.get_text("dict")
        figure_found = False
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                if "FIGURE" in line_text:
                    print(f"📊 Found FIGURE text: '{line_text.strip()}'")
                    figure_found = True
        
        if not figure_found:
            print("❌ No FIGURE text found on this page")
        
        fitz_doc.close()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    pdf_path = "book_2.pdf"  # Adjust if needed
    
    # Test pages that we know have figures
    test_pages = [323, 260, 214, 106, 93]  # From your examples
    
    for page_num in test_pages:
        print(f"\n{'='*60}")
        test_image_extraction_on_page(pdf_path, page_num)
        print()

if __name__ == "__main__":
    main() 