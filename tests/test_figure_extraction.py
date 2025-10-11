#!/usr/bin/env python3
"""
Test script for figure extraction functionality.
"""

import json
import sys
from font_aware_processor import detect_figure_reference, extract_images_from_page
import fitz

def test_figure_detection():
    """Test figure reference detection."""
    test_cases = [
        "FIGURE 7.6",
        "Figure 1.2 Sample diagram",
        "TABLE 3.1 Results summary",
        "This is regular text",
        "Fig. 2 shows the process",
        "CHART 5.3 Performance metrics"
    ]
    
    print("🔍 Testing Figure Detection:")
    print("=" * 50)
    
    for text in test_cases:
        result = detect_figure_reference(text)
        status = "✅ FIGURE" if result["is_figure"] else "❌ TEXT"
        print(f"{status}: '{text}'")
        if result["is_figure"]:
            print(f"   Type: {result['figure_type']}")
            print(f"   Number: {result['figure_number']}")
            if result['figure_title']:
                print(f"   Title: {result['figure_title']}")
        print()

def analyze_existing_json(json_file):
    """Analyze existing JSON for figure references."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"📊 Analyzing {json_file} for figures:")
        print("=" * 50)
        
        figure_count = 0
        image_count = 0
        
        def analyze_container(container, path=""):
            nonlocal figure_count, image_count
            
            # Check paragraphs
            for i, para in enumerate(container.get('paragraphs', [])):
                if para.get('is_figure', False):
                    figure_count += 1
                    images = para.get('images', [])
                    image_count += len(images)
                    
                    print(f"📸 {path}paragraph[{i}]:")
                    print(f"   Text: {para['text'][:100]}...")
                    print(f"   Type: {para.get('figure_type', 'unknown')}")
                    print(f"   Number: {para.get('figure_number', 'unknown')}")
                    print(f"   Images: {len(images)}")
                    if images:
                        for j, img in enumerate(images):
                            print(f"     Image {j}: {img['format']} ({img.get('width', '?')}x{img.get('height', '?')}) - {img.get('size_bytes', 0)} bytes")
                    print()
            
            # Check sections recursively
            for i, section in enumerate(container.get('sections', [])):
                analyze_container(section, f"{path}section[{i}]/")
        
        # Analyze chapters
        for i, chapter in enumerate(data.get('chapters', [])):
            analyze_container(chapter, f"chapter[{i}]/")
        
        print(f"📈 Summary: {figure_count} figures found with {image_count} total images")
        
    except Exception as e:
        print(f"❌ Error analyzing {json_file}: {str(e)}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_figure_extraction.py test          # Test figure detection")
        print("  python test_figure_extraction.py analyze <json_file>  # Analyze existing JSON")
        return
    
    command = sys.argv[1]
    
    if command == "test":
        test_figure_detection()
    elif command == "analyze" and len(sys.argv) > 2:
        analyze_existing_json(sys.argv[2])
    else:
        print("Invalid command or missing arguments.")

if __name__ == "__main__":
    main() 