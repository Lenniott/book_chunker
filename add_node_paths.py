import json
import sys
from typing import List, Dict, Optional

def add_paths_to_nodes(nodes: List[Dict], parent_path: Optional[str] = None):
    """
    Recursively adds a 'node_path' to each node in the book's outline.

    Args:
        nodes: A list of node dictionaries, each representing a section.
        parent_path: The path of the parent node.
    """
    for i, node in enumerate(nodes):
        # Create the current node's path
        current_path = f"{parent_path}.{i + 1}" if parent_path else str(i + 1)
        node['node_path'] = current_path

        # If the node has children, recurse
        if 'children' in node and node['children']:
            add_paths_to_nodes(node['children'], current_path)

def main(file_path: str):
    """
    Reads a JSON file, adds node paths, and writes it back.

    Args:
        file_path: The path to the book_structure.json file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Start the recursive path adding process on the outline
        if 'outline' in data and data['outline']:
            add_paths_to_nodes(data['outline'])

        # Write the updated data back to the same file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Successfully added node paths to {file_path}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Default to 'book_structure.json' if no file is provided
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = 'book_structure.json'
    
    main(json_file) 