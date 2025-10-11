from dotenv import load_dotenv
load_dotenv()
import os
import json
import psycopg2
from psycopg2.extras import Json, register_uuid
from typing import Dict, List, Optional, Tuple, Any
import uuid
from datetime import datetime
import hashlib
import base64
import re

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
    
    def _process_images(self, paragraph_data: Dict[str, Any], content_node_id: uuid.UUID, paragraph_id: Optional[uuid.UUID] = None):
        """Process images from paragraph data and store them in the database."""
        if 'images' not in paragraph_data or not paragraph_data['images']:
            return
        
        # Delete existing images for this content node/paragraph
        if paragraph_id:
            self.cursor.execute("""
                DELETE FROM images 
                WHERE paragraph_id = %s
            """, (paragraph_id,))
        else:
            self.cursor.execute("""
                DELETE FROM images 
                WHERE content_node_id = %s AND paragraph_id IS NULL
            """, (content_node_id,))
        
        # Insert new images
        for i, image_data in enumerate(paragraph_data['images']):
            # Extract image metadata
            base64_data = image_data.get('base64', '')
            if not base64_data:
                continue
                
            # Calculate image size in bytes (approximate from base64)
            image_size_bytes = len(base64_data.encode('utf-8'))
            
            # Try to determine image format from base64 header
            image_format = 'unknown'
            if base64_data.startswith('/9j/'):
                image_format = 'jpg'
            elif base64_data.startswith('iVBORw0KGgo'):
                image_format = 'png'
            elif base64_data.startswith('R0lGODlh'):
                image_format = 'gif'
            
            # Extract other metadata if available
            width = image_data.get('width')
            height = image_data.get('height')
            metadata = {k: v for k, v in image_data.items() if k not in ['base64', 'width', 'height']}
            
            self.cursor.execute("""
                INSERT INTO images 
                (content_node_id, paragraph_id, image_data, image_format, 
                 image_size_bytes, width, height, image_order, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (content_node_id, paragraph_id, base64_data, image_format, 
                  image_size_bytes, width, height, i, Json(metadata)))
    
    def _extract_paragraph_text(self, paragraph_data: Any) -> str:
        """Extract text content from paragraph data (handles both string and dict formats)."""
        if isinstance(paragraph_data, str):
            return paragraph_data
        elif isinstance(paragraph_data, dict):
            return paragraph_data.get('text', '')
        return str(paragraph_data)

    def _process_node(self, 
                     node: Dict, 
                     book_id: uuid.UUID, 
                     parent_id: Optional[uuid.UUID], 
                     parent_path: Optional[str],
                     node_order: int,
                     depth: int = 0) -> uuid.UUID:
        """Process a single content node and its sections recursively."""
        # Use node_path from JSON if it exists, otherwise calculate it
        node_path = node.get('node_path') or self._calculate_node_path(parent_path, node_order)
        
        # Check if node exists using node_path
        self.cursor.execute("""
            SELECT cn.id, vm.content_hash 
            FROM content_nodes cn
            LEFT JOIN vector_metadata vm ON cn.id = vm.content_node_id
            WHERE cn.book_id = %s AND cn.node_path = %s
        """, (book_id, node_path))
        
        result = self.cursor.fetchone()
        
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

        # Process header_paragraph if it exists
        if 'header_paragraph' in node:
            header = node['header_paragraph']
            text_content = self._extract_paragraph_text(header)
            
            # Create a special paragraph for the header
            paragraph_id = uuid.uuid4()
            word_count = len(text_content.split()) if text_content else 0
            char_count = len(text_content) if text_content else 0
            
            self.cursor.execute("""
                INSERT INTO paragraphs 
                (content_node_id, content, paragraph_order, word_count, char_count)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (node_id, text_content, -1, word_count, char_count))  # -1 indicates header
            paragraph_id = self.cursor.fetchone()[0]
            
            # Process images in header paragraph
            if isinstance(header, dict):
                self._process_images(header, node_id, paragraph_id)

        # Process paragraphs if they exist
        if 'paragraphs' in node and node['paragraphs']:
            # Calculate content hash for change detection
            paragraph_texts = [self._extract_paragraph_text(p) for p in node['paragraphs']]
            content_hash = self._get_content_hash(paragraph_texts)
            
            # Delete existing paragraphs (except header)
            self.cursor.execute("""
                DELETE FROM paragraphs 
                WHERE content_node_id = %s AND paragraph_order >= 0
            """, (node_id,))
            
            # Insert new paragraphs
            for i, paragraph_data in enumerate(node['paragraphs']):
                text_content = self._extract_paragraph_text(paragraph_data)
                word_count = len(text_content.split()) if text_content else 0
                char_count = len(text_content) if text_content else 0
                
                self.cursor.execute("""
                    INSERT INTO paragraphs 
                    (content_node_id, content, paragraph_order, word_count, char_count)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (node_id, text_content, i, word_count, char_count))
                paragraph_id = self.cursor.fetchone()[0]
                
                # Process images in this paragraph if it's a dict with images
                if isinstance(paragraph_data, dict):
                    self._process_images(paragraph_data, node_id, paragraph_id)
            
            # Update vector metadata if content changed
            if not result or existing_hash != content_hash:
                self.cursor.execute("""
                    DELETE FROM vector_metadata 
                    WHERE content_node_id = %s
                """, (node_id,))

        # Process sections recursively (replaces the old 'children' logic)
        if 'sections' in node and node['sections']:
            for i, section in enumerate(node['sections']):
                self._process_node(section, book_id, node_id, node_path, i + 1, depth + 1)

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

    def delete_book(self, book_id: uuid.UUID) -> bool:
        """Delete a book and all its associated data from the database.
        
        Args:
            book_id: UUID of the book to delete
            
        Returns:
            bool: True if book was found and deleted, False if book was not found
        """
        try:
            # First check if book exists
            self.cursor.execute("""
                SELECT 1 FROM books WHERE id = %s
            """, (book_id,))
            
            if not self.cursor.fetchone():
                return False
                
            # Delete vector metadata for all content nodes of this book
            self.cursor.execute("""
                DELETE FROM vector_metadata
                WHERE content_node_id IN (
                    SELECT id FROM content_nodes WHERE book_id = %s
                )
            """, (book_id,))
            
            # Delete images for all content nodes of this book
            self.cursor.execute("""
                DELETE FROM images
                WHERE content_node_id IN (
                    SELECT id FROM content_nodes WHERE book_id = %s
                )
            """, (book_id,))
            
            # Delete paragraphs for all content nodes of this book
            self.cursor.execute("""
                DELETE FROM paragraphs
                WHERE content_node_id IN (
                    SELECT id FROM content_nodes WHERE book_id = %s
                )
            """, (book_id,))
            
            # Delete all content nodes for this book
            self.cursor.execute("""
                DELETE FROM content_nodes WHERE book_id = %s
            """, (book_id,))
            
            # Finally delete the book itself
            self.cursor.execute("""
                DELETE FROM books WHERE id = %s
            """, (book_id,))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Error deleting book: {str(e)}")

def load_book_to_db(json_file_path: str, db_connection_params: Dict[str, str]) -> uuid.UUID:
    """Convenience function to load a book from JSON to database."""
    with BookDBLoader(db_connection_params) as loader:
        return loader.load_book_from_json(json_file_path)

# Example usage:
if __name__ == "__main__":
    import sys
    import os
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  To load book:   python db_loader.py <path_to_json_file>")
        print("  To delete book: python db_loader.py --delete <book_uuid>")
        print("Example: python db_loader.py book_structure.json")
        print("Example: python db_loader.py --delete 123e4567-e89b-12d3-a456-426614174000")
        sys.exit(1)
    
    # Database connection parameters from environment
    db_params = {
        "dbname": os.getenv("PG_DBNAME"),
        "user": os.getenv("PG_USER"),
        "password": os.getenv("PG_PASSWORD"),
        "host": os.getenv("PG_HOST"),
        "port": os.getenv("PG_PORT"),
    }
    
    # Handle delete command
    if sys.argv[1] == "--delete":
        if len(sys.argv) != 3:
            print("Error: Book UUID required for delete operation")
            sys.exit(1)
            
        try:
            book_id = uuid.UUID(sys.argv[2])
            with BookDBLoader(db_params) as loader:
                if loader.delete_book(book_id):
                    print(f"Successfully deleted book with ID: {book_id}")
                else:
                    print(f"Book with ID {book_id} not found")
        except ValueError:
            print("Error: Invalid UUID format")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)
    
    # Handle load command
    else:
        json_file_path = sys.argv[1]
        if not os.path.exists(json_file_path):
            print(f"Error: File '{json_file_path}' not found.")
            sys.exit(1)
        
    try:
        print(f"Loading book from: {json_file_path}")
        book_id = load_book_to_db(json_file_path, db_params)
        print(f"Successfully loaded/updated book with ID: {book_id}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)