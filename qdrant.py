import os
# Clear any existing environment variables first
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']

from dotenv import load_dotenv
# Force reload of .env file
load_dotenv(override=True)

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import openai
import uuid
import logging
import sys
import argparse
import json
from typing import Dict, List, Tuple, Any

# Temporary debug - remove after testing
print("\nEnvironment Debug Info:")
print(f".env file exists: {os.path.exists('.env')}")
print(f"Current working directory: {os.getcwd()}")
print(f"All env vars starting with 'OPENAI': {[k for k in os.environ.keys() if k.startswith('OPENAI')]}")
api_key = os.getenv('OPENAI_API_KEY', '')
print("\nAPI Key Debug Info:")
print(f"Raw length: {len(api_key)}")
print(f"First 20 chars: {api_key[:20]}")
print(f"Contains 'sk-proj-': {'sk-proj-' in api_key}")
print(f"Key format check: {api_key.startswith('sk-')}")
print(f"All chars: {[ord(c) for c in api_key[:10]]}")  # Show ASCII values of first 10 chars

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vectorization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PG_PARAMS = {
    "dbname": os.getenv("PG_DBNAME"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
}

# --- SETUP ---
openai.api_key = OPENAI_API_KEY
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def test_connections():
    """Test all required connections."""
    tests = {
        "PostgreSQL": test_postgres_connection,
        "Qdrant": test_qdrant_connection,
        "OpenAI": test_openai_connection
    }
    
    results = {}
    for name, test_func in tests.items():
        try:
            results[name] = test_func()
            logger.info(f"✓ {name} connection test passed")
        except Exception as e:
            logger.error(f"✗ {name} connection test failed: {str(e)}")
            results[name] = False
    
    return all(results.values())

def test_postgres_connection():
    """Test PostgreSQL connection and required tables."""
    try:
        conn = psycopg2.connect(**PG_PARAMS)
        cur = conn.cursor()
        
        # Test required tables exist
        required_tables = ['content_nodes', 'paragraphs', 'vector_metadata']
        for table in required_tables:
            cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)", (table,))
            exists = cur.fetchone()[0]
            if not exists:
                logger.error(f"Required table '{table}' does not exist!")
                return False
            logger.info(f"✓ Table '{table}' exists")
        
        # Test we can query content_nodes
        cur.execute("SELECT COUNT(*) FROM content_nodes WHERE depth = 1")
        chapter_count = cur.fetchone()[0]
        logger.info(f"Found {chapter_count} chapters in content_nodes")
        
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"PostgreSQL connection test failed: {str(e)}")
        return False

def test_qdrant_connection():
    """Test Qdrant connection."""
    try:
        collections = qdrant.get_collections()
        logger.info(f"Found {len(collections.collections)} collections in Qdrant")
        return True
    except Exception as e:
        logger.error(f"Qdrant connection test failed: {str(e)}")
        return False

def test_openai_connection():
    """Test OpenAI API connection with a small test."""
    try:
        test_embedding = vectorize_text("test")
        return True
    except Exception as e:
        logger.error(f"OpenAI API connection test failed: {str(e)}")
        return False

def get_book_id(book_title: str) -> uuid.UUID:
    """Get the book ID from the database."""
    conn = psycopg2.connect(**PG_PARAMS)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
    cur.execute("""
            SELECT id, title, author 
            FROM books 
            WHERE title ILIKE %s
        """, (f"%{book_title}%",))
        books = cur.fetchall()
        
        if not books:
            logger.error(f"No book found matching title: {book_title}")
            return None
        
        if len(books) > 1:
            logger.info("\nMultiple books found matching your search:")
            for i, book in enumerate(books, 1):
                logger.info(f"{i}. {book['title']} by {book['author']}")
            logger.error("Please be more specific with the book title")
            return None
            
        return books[0]['id']
    finally:
        cur.close()
        conn.close()

def get_sections_with_content(book_id: uuid.UUID) -> List[Dict]:
    """Get sections with their full content and context for a specific book."""
    conn = psycopg2.connect(**PG_PARAMS)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        # Get all sections with their paragraphs and context
        cur.execute("""
            WITH RECURSIVE node_tree AS (
                -- Get all content nodes
                SELECT 
                    cn.id,
                    cn.title,
                    cn.node_path,
                    cn.depth,
                    cn.parent_id,
                    cn.book_id,
                    cn.page_number
                FROM content_nodes cn
                WHERE cn.book_id = %s
            )
            SELECT 
                cn.id as section_id,
                cn.title as section_title,
                cn.node_path,
                cn.depth,
                cn.page_number,
                parent.title as parent_title,
                parent.node_path as parent_path,
                parent.id as parent_id,
                -- Aggregate paragraphs into a single text with metadata
                json_agg(
                    json_build_object(
                        'paragraph_id', p.id,
                        'content', p.content,
                        'paragraph_order', p.paragraph_order,
                        'word_count', p.word_count,
                        'char_count', p.char_count
                    ) ORDER BY p.paragraph_order
                ) as paragraphs,
                -- Calculate section statistics
                COUNT(p.id) as paragraph_count,
                SUM(p.word_count) as total_words,
                SUM(p.char_count) as total_chars
            FROM node_tree cn
            LEFT JOIN node_tree parent ON cn.parent_id = parent.id
            LEFT JOIN paragraphs p ON p.content_node_id = cn.id
            WHERE cn.book_id = %s
            GROUP BY 
                cn.id, cn.title, cn.node_path, cn.depth, 
                cn.page_number, parent.title, parent.node_path, parent.id
            ORDER BY cn.node_path
        """, (book_id, book_id))
        
        sections = cur.fetchall()
        return [dict(s) for s in sections]
    finally:
    cur.close()
    conn.close()

def vectorize_text(text: str) -> Tuple[List[float], int]:
    """Vectorize text and return both the embedding and its size in bytes."""
    response = openai.embeddings.create(
        input=text,
        model=OPENAI_MODEL
    )
    embedding = response.data[0].embedding
    # Calculate size of the embedding (1536 floats * 4 bytes per float)
    size_bytes = len(embedding) * 4
    return embedding, size_bytes

def ensure_collection_exists(collection_name: str):
    """Ensure the specified Qdrant collection exists."""
    if collection_name not in [c.name for c in qdrant.get_collections().collections]:
        logger.info(f"Creating Qdrant collection: {collection_name}")
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # OpenAI text-embedding-3-small size
                distance=Distance.COSINE
            )
        )
    else:
        logger.info(f"Using existing collection: {collection_name}")

def store_section_vectors(sections: List[Dict], collection: str, pg_conn) -> Dict[str, int]:
    """Store vectors for sections and return size information."""
    total_size = 0
    vector_count = 0
    points = []
    
    for section in sections:
        # Filter out None values and empty paragraphs
        valid_paragraphs = [p for p in section['paragraphs'] if p and p.get('content')]
        
        if not valid_paragraphs:
            logger.warning(f"Skipping section '{section['section_title']}' - no valid paragraphs")
            continue
            
        # Combine paragraphs into a single text with clear separation
        section_text = "\n\n".join(p['content'] for p in valid_paragraphs)
        
        # Skip if section text is empty
        if not section_text.strip():
            logger.warning(f"Skipping section '{section['section_title']}' - empty content after filtering")
            continue
        
        # Vectorize the section
        embedding, size_bytes = vectorize_text(section_text)
        
        # Create unique ID for the section vector
        vector_id = str(uuid.uuid4())
        
        # Prepare metadata
        metadata = {
            "section_id": str(section['section_id']),
            "section_title": section['section_title'],
            "node_path": section['node_path'],
            "depth": section['depth'],
            "page_number": section['page_number'],
            "parent_title": section['parent_title'],
            "parent_path": section['parent_path'],
            "parent_id": str(section['parent_id']) if section['parent_id'] else None,
            # Section statistics
            "paragraph_count": len(valid_paragraphs),  # Updated to use filtered count
            "total_words": sum(p.get('word_count', 0) for p in valid_paragraphs),
            "total_chars": sum(p.get('char_count', 0) for p in valid_paragraphs),
            # Keep paragraph details for reference
            "paragraphs": valid_paragraphs  # Store only valid paragraphs
        }
        
        # Add to points batch
        points.append(PointStruct(
            id=vector_id,
            vector=embedding,
            payload=metadata
        ))
        
        # Update size tracking
        total_size += size_bytes
        vector_count += 1
        
        # Save metadata to PostgreSQL
        cur = pg_conn.cursor()
        cur.execute("""
            INSERT INTO vector_metadata (
                id, content_node_id, qdrant_vector_id, vector_collection,
                vectorized_at, vector_model, content_hash, vector_size_bytes,
                search_level, context_metadata
            ) VALUES (
                %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s
            )
            ON CONFLICT (content_node_id, vector_collection) DO UPDATE SET
                qdrant_vector_id = EXCLUDED.qdrant_vector_id,
                vectorized_at = NOW(),
                vector_model = EXCLUDED.vector_model,
                content_hash = EXCLUDED.content_hash,
                vector_size_bytes = EXCLUDED.vector_size_bytes,
                search_level = EXCLUDED.search_level,
                context_metadata = EXCLUDED.context_metadata
        """, (
            str(uuid.uuid4()),
            section['section_id'],
            vector_id,
            collection,
            OPENAI_MODEL,
            str(hash(section_text)),
            size_bytes,
            'section',
            Json(metadata)
        ))
        pg_conn.commit()
        cur.close()
        
        # Log progress
        if vector_count % 10 == 0:
            logger.info(f"Processed {vector_count} sections...")
            logger.info(f"Last section: {section['section_title']} ({len(valid_paragraphs)} paragraphs)")
    
    # Batch upsert to Qdrant
    if points:
        logger.info(f"Storing {len(points)} vectors in Qdrant...")
        qdrant.upsert(
            collection_name=collection,
            points=points
        )
    
    return {
        "total_size_bytes": total_size,
        "vector_count": vector_count,
        "avg_vector_size": total_size / vector_count if vector_count > 0 else 0
    }

def main():
    parser = argparse.ArgumentParser(description='Vectorize book sections and store in Qdrant')
    parser.add_argument('--book', '-b', required=True, help='Book title to vectorize (can be partial match)')
    parser.add_argument('--collection', '-c', required=True, help='Qdrant collection name to use')
    parser.add_argument('--show-stats', '-s', action='store_true', help='Show vector storage statistics')
    args = parser.parse_args()

    # Test all connections first
    logger.info("Testing connections...")
    if not test_connections():
        logger.error("Connection tests failed. Exiting.")
        return

    # Get book ID
    book_id = get_book_id(args.book)
    if not book_id:
        return

    # Ensure collection exists
    ensure_collection_exists(args.collection)
    
    # Get sections with content
    logger.info("Fetching sections with content...")
    sections = get_sections_with_content(book_id)
    logger.info(f"Found {len(sections)} sections to process")
    
    # Connect to PostgreSQL for metadata storage
    pg_conn = psycopg2.connect(**PG_PARAMS)
    
    try:
        # Store vectors and get size information
        size_info = store_section_vectors(sections, args.collection, pg_conn)
        
        # Log summary
        logger.info("\nVectorization Summary:")
        logger.info(f"Book ID: {book_id}")
        logger.info(f"Collection: {args.collection}")
        logger.info(f"Total sections processed: {size_info['vector_count']}")
        logger.info(f"Total vector storage size: {size_info['total_size_bytes'] / (1024*1024):.2f} MB")
        logger.info(f"Average vector size: {size_info['avg_vector_size'] / 1024:.2f} KB")
        
        if args.show_stats:
            # Get and display collection statistics from Qdrant
            collection_info = qdrant.get_collection(args.collection)
            logger.info("\nQdrant Collection Statistics:")
            logger.info(f"Vectors count: {collection_info.vectors_count}")
            logger.info(f"Points count: {collection_info.points_count}")
            logger.info(f"Status: {collection_info.status}")
            
    finally:
    pg_conn.close()

if __name__ == "__main__":
    main()