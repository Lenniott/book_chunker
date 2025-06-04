from dotenv import load_dotenv
load_dotenv()
import os
import json
import psycopg2
from psycopg2.extras import Json, register_uuid
from typing import Dict, List, Optional, Tuple
import uuid
from datetime import datetime
import hashlib

# Register UUID type with psycopg2
register_uuid()

class BookDBLoader:
    def __init__(self, db_connection_params: Dict[str, str]):
        """Initialize the database loader with connection parameters."""
        self.conn_params = db_connection_params
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish database connection."""
        self.conn = psycopg2.connect(**self.conn_params)
        self.cursor = self.conn.cursor()

    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_or_create_book(self, title: str, author: str, total_pages: int) -> uuid.UUID:
        """Get existing book ID or create new book entry."""
        # Check if book exists
        self.cursor.execute("""
            SELECT id FROM books 
            WHERE title = %s AND author = %s
        """, (title, author))
        result = self.cursor.fetchone()
        
        if result:
            book_id = result[0]
            # Update existing book
            self.cursor.execute("""
                UPDATE books 
                SET total_pages = %s, updated_at = NOW()
                WHERE id = %s
            """, (total_pages, book_id))
        else:
            # Create new book
            book_id = uuid.uuid4()
            self.cursor.execute("""
                INSERT INTO books (id, title, author, total_pages)
                VALUES (%s, %s, %s, %s)
            """, (book_id, title, author, total_pages))
        
        return book_id

    def _calculate_node_path(self, parent_path: Optional[str], node_order: int) -> str:
        """Calculate the materialized path for a node."""
        if parent_path is None:
            return str(node_order)
        return f"{parent_path}.{node_order}"

    def _get_content_hash(self, paragraphs: List[str]) -> str:
        """Calculate a hash of the content for change detection."""
        content = "".join(paragraphs)
        return hashlib.sha256(content.encode()).hexdigest()

    def _process_node(self, 
                     node: Dict, 
                     book_id: uuid.UUID, 
                     parent_id: Optional[uuid.UUID], 
                     parent_path: Optional[str],
                     node_order: int,
                     depth: int = 0) -> uuid.UUID:
        """Process a single content node and its children recursively."""
        # Check if node exists - Fixed the ambiguous column reference
        self.cursor.execute("""
            SELECT cn.id, vm.content_hash 
            FROM content_nodes cn
            LEFT JOIN vector_metadata vm ON cn.id = vm.content_node_id
            WHERE cn.book_id = %s AND cn.title = %s AND cn.page_number = %s
            AND (cn.parent_id = %s OR (cn.parent_id IS NULL AND %s IS NULL))
        """, (book_id, node['title'], node['page'], parent_id, parent_id))
        
        result = self.cursor.fetchone()
        node_path = self._calculate_node_path(parent_path, node_order)
        
        if result:
            node_id, existing_hash = result
            # Update existing node
            self.cursor.execute("""
                UPDATE content_nodes 
                SET node_path = %s, depth = %s, node_order = %s,
                    has_paragraphs = %s, updated_at = NOW()
                WHERE id = %s
            """, (node_path, depth, node_order, bool(node.get('paragraphs')), node_id))
        else:
            # Create new node
            node_id = uuid.uuid4()
            self.cursor.execute("""
                INSERT INTO content_nodes 
                (id, book_id, parent_id, title, page_number, node_path, 
                 depth, node_order, has_paragraphs)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (node_id, book_id, parent_id, node['title'], node['page'],
                  node_path, depth, node_order, bool(node.get('paragraphs'))))

        # Process paragraphs if they exist
        if 'paragraphs' in node and node['paragraphs']:
            content_hash = self._get_content_hash(node['paragraphs'])
            
            # Delete existing paragraphs
            self.cursor.execute("""
                DELETE FROM paragraphs 
                WHERE content_node_id = %s
            """, (node_id,))
            
            # Insert new paragraphs
            for i, paragraph in enumerate(node['paragraphs']):
                word_count = len(paragraph.split())
                char_count = len(paragraph)
                self.cursor.execute("""
                    INSERT INTO paragraphs 
                    (content_node_id, content, paragraph_order, word_count, char_count)
                    VALUES (%s, %s, %s, %s, %s)
                """, (node_id, paragraph, i, word_count, char_count))
            
            # Update vector metadata if content changed
            if not result or existing_hash != content_hash:
                self.cursor.execute("""
                    DELETE FROM vector_metadata 
                    WHERE content_node_id = %s
                """, (node_id,))

        # Process children recursively
        if 'children' in node:
            for i, child in enumerate(node['children']):
                self._process_node(child, book_id, node_id, node_path, i + 1, depth + 1)

        return node_id

    def load_book_from_json(self, json_file_path: str) -> uuid.UUID:
        """Load or update a book from a JSON file."""
        with open(json_file_path, 'r') as f:
            book_data = json.load(f)

        try:
            # Get or create book
            book_id = self._get_or_create_book(
                book_data['title'],
                book_data['author'],
                book_data['total_pages']
            )

            # Process outline recursively
            for i, node in enumerate(book_data['outline']):
                self._process_node(node, book_id, None, None, i + 1)

            self.conn.commit()
            return book_id

        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Error loading book: {str(e)}")

def load_book_to_db(json_file_path: str, db_connection_params: Dict[str, str]) -> uuid.UUID:
    """Convenience function to load a book from JSON to database."""
    with BookDBLoader(db_connection_params) as loader:
        return loader.load_book_from_json(json_file_path)

# Example usage:
if __name__ == "__main__":
    import sys
    import os
    
    # Check if JSON file path is provided
    if len(sys.argv) != 2:
        print("Usage: python db_loader.py <path_to_json_file>")
        print("Example: python db_loader.py book_structure.json")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File '{json_file_path}' not found.")
        sys.exit(1)
    
    # Database connection parameters from environment
    db_params = {
        "dbname": os.getenv("PG_DBNAME"),
        "user": os.getenv("PG_USER"),
        "password": os.getenv("PG_PASSWORD"),
        "host": os.getenv("PG_HOST"),
        "port": os.getenv("PG_PORT"),
    }
    
    # Load the book
    try:
        print(f"Loading book from: {json_file_path}")
        book_id = load_book_to_db(json_file_path, db_params)
        print(f"Successfully loaded/updated book with ID: {book_id}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)