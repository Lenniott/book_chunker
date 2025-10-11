import json
import re
import sys
import os
from typing import Dict, List, Union

def clean_text(text: str) -> str:
    """Clean text by replacing Unicode characters and fixing spacing issues.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text with standard ASCII characters and proper spacing
    """
    # Replace Unicode apostrophe with standard single quote
    text = text.replace('\u2019', "'")
    
    # Fix words that are stuck together without proper spacing
    # Example: "becauseIfailed" -> "because I failed"
    text = re.sub(r'([a-z])I([a-z])', r'\1 I \2', text)
    
    return text

def split_camel_case(text: str) -> str:
    """Split camel case text into space-separated words.
    
    Examples:
        >>> split_camel_case("thisIsATest")
        'this Is A Test'
        >>> split_camel_case("AnotherExample")
        'Another Example'
        >>> split_camel_case("ABC")
        'ABC'
        >>> split_camel_case("simpleword")
        'simpleword'
        >>> split_camel_case("becauseIfailed")
        'because I failed'
    """
    # First clean the text
    text = clean_text(text)
    
    # Handle special cases
    if not text or len(text) <= 1:
        return text
        
    # Don't split if all uppercase (likely an acronym)
    if text.isupper():
        return text
        
    # Don't split if all lowercase
    if text.islower():
        return text
        
    # Pattern to find camel case boundaries
    pattern = r'(?<!^)(?<![\W_])(?=[A-Z][a-z])|(?<!^)(?<=[a-z])(?=[A-Z])'
    return ' '.join(re.split(pattern, text))

def process_node(node: Dict) -> Dict:
    """Recursively process a node and its children, splitting camel case text."""
    # Process title
    if 'title' in node and isinstance(node['title'], str):
        node['title'] = split_camel_case(node['title'])
    
    # Process paragraphs
    if 'paragraphs' in node and node['paragraphs']:
        node['paragraphs'] = [
            ' '.join(split_camel_case(word) for word in paragraph.split())
            for paragraph in node['paragraphs']
        ]
    
    # Process children recursively
    if 'children' in node:
        node['children'] = [process_node(child) for child in node['children']]
    
    return node

def process_book_structure(input_file: str, output_file: str) -> None:
    """Process a book structure JSON file, splitting camel case text.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to save processed JSON file
    """
    # Read input file
    with open(input_file, 'r') as f:
        book_data = json.load(f)
    
    # Process book title and author
    if 'title' in book_data:
        book_data['title'] = split_camel_case(book_data['title'])
    if 'author' in book_data:
        book_data['author'] = split_camel_case(book_data['author'])
    
    # Process outline
    if 'outline' in book_data:
        book_data['outline'] = [process_node(node) for node in book_data['outline']]
    
    # Save processed data
    with open(output_file, 'w') as f:
        json.dump(book_data, f, indent=2)

def main():
    if len(sys.argv) != 2:
        print("Usage: python split_camel_case.py <input_json_file>")
        print("Example: python split_camel_case.py book_structure.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Generate output filename
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_processed{ext}"
    
    try:
        process_book_structure(input_file, output_file)
        print(f"Successfully processed {input_file}")
        print(f"Saved results to {output_file}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 