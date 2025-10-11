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
from datetime import datetime

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
        required_tables = ['content_nodes', 'paragraphs', 'vector_metadata', 'images']
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
    """Get sections with their full content, images, and context for a specific book."""
    conn = psycopg2.connect(**PG_PARAMS)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        # Get all sections with their paragraphs, images, and context
        # Fixed: Use separate subqueries to avoid cartesian product between paragraphs and images
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
            ),
            section_paragraphs AS (
                -- Get paragraphs for each section
                SELECT 
                    cn.id as section_id,
                    json_agg(
                        jsonb_build_object(
                            'paragraph_id', p.id,
                            'content', p.content,
                            'paragraph_order', p.paragraph_order,
                            'word_count', p.word_count,
                            'char_count', p.char_count
                        ) ORDER BY p.paragraph_order
                    ) FILTER (WHERE p.id IS NOT NULL) as paragraphs,
                    COUNT(p.id) as paragraph_count,
                    SUM(p.word_count) as total_words,
                    SUM(p.char_count) as total_chars
                FROM node_tree cn
                LEFT JOIN paragraphs p ON p.content_node_id = cn.id
                GROUP BY cn.id
            ),
            section_images AS (
                -- Get images for each section
                SELECT 
                    cn.id as section_id,
                    json_agg(
                        jsonb_build_object(
                            'image_id', i.id,
                            'paragraph_id', i.paragraph_id,
                            'image_format', i.image_format,
                            'image_size_bytes', i.image_size_bytes,
                            'width', i.width,
                            'height', i.height,
                            'image_order', i.image_order,
                            'metadata', i.metadata
                        ) ORDER BY i.image_order
                    ) FILTER (WHERE i.id IS NOT NULL) as images,
                    COUNT(i.id) as image_count
                FROM node_tree cn
                LEFT JOIN images i ON i.content_node_id = cn.id
                GROUP BY cn.id
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
                -- Get paragraphs and images from separate subqueries
                COALESCE(sp.paragraphs, '[]'::json) as paragraphs,
                COALESCE(si.images, '[]'::json) as images,
                COALESCE(sp.paragraph_count, 0) as paragraph_count,
                COALESCE(si.image_count, 0) as image_count,
                COALESCE(sp.total_words, 0) as total_words,
                COALESCE(sp.total_chars, 0) as total_chars
            FROM node_tree cn
            LEFT JOIN node_tree parent ON cn.parent_id = parent.id
            LEFT JOIN section_paragraphs sp ON sp.section_id = cn.id
            LEFT JOIN section_images si ON si.section_id = cn.id
            WHERE cn.book_id = %s
            ORDER BY cn.node_path
        """, (book_id, book_id))
        
        sections = cur.fetchall()
        return [dict(s) for s in sections]
    finally:
    cur.close()
    conn.close()

def vectorize_text(text: str) -> Tuple[List[float], int]:
    """Vectorize text and return both the embedding and its size in bytes."""
    # Rough estimate: 1 token ≈ 4 characters for English text
    # OpenAI limit is 8192 tokens, so let's use ~32,000 characters as a safe limit
    max_chars = 30000
    
    if len(text) > max_chars:
        # If text is too long, truncate it and log a warning
        logger.warning(f"Text too long ({len(text)} chars, ~{len(text)//4} tokens). Truncating to {max_chars} chars.")
        text = text[:max_chars] + "... [TRUNCATED]"
    
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

def chunk_paragraphs_by_groups(paragraphs: List[Dict], chunk_size: int = 3) -> List[List[Dict]]:
    """Split paragraphs into chunks of specified size while preserving paragraph boundaries."""
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        chunk = paragraphs[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def store_section_vectors_chunked(sections: List[Dict], collection: str, pg_conn, paragraphs_per_chunk: int = 3) -> Dict[str, int]:
    """
    Store section vectors in Qdrant with paragraph-based chunking.
    
    Args:
        sections: List of section dictionaries with content and images
        collection: Qdrant collection name
        pg_conn: PostgreSQL connection
        paragraphs_per_chunk: Number of paragraphs per chunk (default: 3)
    
    Returns:
        Dictionary with storage statistics
        
    METADATA INCLUDED IN EACH CHUNK VECTOR:
    - section_id: Original section UUID
    - section_title: Title of the parent section
    - node_path: Hierarchical path (e.g., "10.2.3")
    - depth: How deep in the hierarchy (0=chapter, 1=section, etc.)
    - page_number: Where this content appears in the book
    - parent_title/parent_path/parent_id: Parent section info
    - chunk_index: Which chunk this is (0, 1, 2...)
    - total_chunks_in_section: How many chunks the section was split into
    - is_first_chunk/is_last_chunk: Boolean flags
    - paragraph_count: Number of paragraphs in this chunk
    - image_count: Number of images in this chunk
    - total_words/total_chars: Text statistics for this chunk
    - paragraphs: Full paragraph objects with metadata
    - images: Image objects that belong to paragraphs in this chunk
    - has_images: Boolean flag
    - content_types: ["text"] or ["text", "images"]
    - chunking_method: "paragraph_groups"
    - paragraphs_per_chunk: The chunk size used
    - chunk_text_preview: First 200 chars of the combined text
    """
    logger.info(f"Processing {len(sections)} sections with {paragraphs_per_chunk} paragraphs per chunk...")
    
    points = []
    total_size = 0
    vector_count = 0
    chunk_count = 0
    
    for section in sections:
        # Filter out None values and empty paragraphs
        valid_paragraphs = [p for p in section['paragraphs'] if p and p.get('content')] if section['paragraphs'] else []
        
        if not valid_paragraphs:
            continue
        
        # Process image metadata once for the section
        section_image_metadata = []
        if section['images']:
            for img in section['images']:
                if img:  # Filter out None values
                    section_image_metadata.append({
                        'image_id': img['image_id'],
                        'paragraph_id': img['paragraph_id'],
                        'image_format': img['image_format'],
                        'image_size_bytes': img['image_size_bytes'],
                        'width': img['width'],
                        'height': img['height'],
                        'image_order': img['image_order'],
                        'metadata': img['metadata']
                    })
        
        # Split paragraphs into chunks
        paragraph_chunks = chunk_paragraphs_by_groups(valid_paragraphs, paragraphs_per_chunk)
        
        for chunk_idx, paragraph_chunk in enumerate(paragraph_chunks):
            # Combine paragraphs in this chunk
            chunk_text = "\n\n".join(p['content'] for p in paragraph_chunk)
            
            # Skip if chunk is too small (less than 50 characters)
            if len(chunk_text) < 50:
                continue
            
            # Truncate if too long (keep paragraph boundaries when possible)
            if len(chunk_text) > 30000:
                # Try to fit complete paragraphs within limit
                truncated_text = ""
                for p in paragraph_chunk:
                    test_text = truncated_text + ("\n\n" if truncated_text else "") + p['content']
                    if len(test_text) <= 30000:
                        truncated_text = test_text
                    else:
                        break
                chunk_text = truncated_text if truncated_text else chunk_text[:30000]
            
            # Get embeddings
            try:
                embedding, size_bytes = vectorize_text(chunk_text)
            except Exception as e:
                logger.error(f"Failed to vectorize chunk {chunk_idx} of section {section['section_title']}: {e}")
                continue
            
            # Create unique vector ID for this chunk (Qdrant requires UUID or int)
            vector_id = str(uuid.uuid4())
            
            # Filter images that belong to paragraphs in this chunk
            chunk_paragraph_ids = {p['paragraph_id'] for p in paragraph_chunk}
            chunk_images = [img for img in section_image_metadata 
                           if img['paragraph_id'] in chunk_paragraph_ids]
            
            # Create chunk metadata
            metadata = {
                "section_id": str(section['section_id']),
                "section_title": section['section_title'],
                "node_path": section['node_path'],
                "depth": section['depth'],
                "page_number": section['page_number'],
                "parent_title": section['parent_title'],
                "parent_path": section['parent_path'],
                "parent_id": str(section['parent_id']) if section['parent_id'] else None,
                
                # Chunk-specific metadata
                "chunk_index": chunk_idx,
                "total_chunks_in_section": len(paragraph_chunks),
                "is_first_chunk": chunk_idx == 0,
                "is_last_chunk": chunk_idx == len(paragraph_chunks) - 1,
                
                # Chunk statistics
                "paragraph_count": len(paragraph_chunk),
                "image_count": len(chunk_images),
                "total_words": sum(p.get('word_count', 0) for p in paragraph_chunk),
                "total_chars": len(chunk_text),
                
                # Content details
                "paragraphs": paragraph_chunk,
                "images": chunk_images,
                "has_images": len(chunk_images) > 0,
                "content_types": ["text"] + (["images"] if chunk_images else []),
                
                # Chunking info
                "chunking_method": "paragraph_groups",
                "paragraphs_per_chunk": paragraphs_per_chunk,
                "chunk_text_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                "readable_chunk_id": f"{section['section_id']}_chunk_{chunk_idx}"
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
            chunk_count += 1
            
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
                vector_id,  # This will be unique per chunk
                collection,
                OPENAI_MODEL,
                str(hash(chunk_text)),
                size_bytes,
                'chunk',  # Changed from 'section' to 'chunk'
                Json(metadata)
            ))
            pg_conn.commit()
            cur.close()
            
            # Log progress with chunk info
            if chunk_count % 10 == 0:
                logger.info(f"Processed {chunk_count} chunks ({vector_count} vectors)...")
                logger.info(f"Last chunk: {section['section_title']} chunk {chunk_idx}/{len(paragraph_chunks)-1} ({len(paragraph_chunk)} paragraphs, {len(chunk_images)} images)")
    
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
        "chunk_count": chunk_count,
        "avg_vector_size": total_size / vector_count if vector_count > 0 else 0
    }

def store_section_vectors(sections: List[Dict], collection: str, pg_conn) -> Dict[str, int]:
    """Store vectors for sections with image metadata and return size information (original method)."""
    logger.info(f"Processing {len(sections)} sections (whole sections, no chunking)...")
    
    points = []
    total_size = 0
    vector_count = 0
    
    for section in sections:
        # Filter out None values and empty paragraphs
        valid_paragraphs = [p for p in section['paragraphs'] if p and p.get('content')] if section['paragraphs'] else []
        
        if not valid_paragraphs:
            logger.warning(f"Skipping section '{section['section_title']}' - no valid paragraphs")
            continue
            
        # Combine paragraphs into a single text with clear separation
        section_text = "\n\n".join(p['content'] for p in valid_paragraphs)
        
        # Skip if section text is empty
        if not section_text.strip():
            logger.warning(f"Skipping section '{section['section_title']}' - empty content after filtering")
            continue
        
        # Truncate if too long
        if len(section_text) > 30000:
            section_text = section_text[:30000]
        
        # Vectorize the section
        embedding, size_bytes = vectorize_text(section_text)
        
        # Create unique ID for the section vector
        vector_id = str(uuid.uuid4())
        
        # Process image metadata
        image_metadata = []
        if section['images']:
            for img in section['images']:
                if img:  # Filter out None values
                    image_metadata.append({
                        'image_id': img['image_id'],
                        'paragraph_id': img['paragraph_id'],
                        'image_format': img['image_format'],
                        'image_size_bytes': img['image_size_bytes'],
                        'width': img['width'],
                        'height': img['height'],
                        'image_order': img['image_order'],
                        'metadata': img['metadata']
                    })
        
        # Prepare metadata with image information
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
            "paragraph_count": len(valid_paragraphs),
            "image_count": len(image_metadata),
            "total_words": sum(p.get('word_count', 0) for p in valid_paragraphs),
            "total_chars": sum(p.get('char_count', 0) for p in valid_paragraphs),
            # Keep paragraph details for reference
            "paragraphs": valid_paragraphs,
            # NEW: Include image metadata for retrieval
            "images": image_metadata,
            # Summary of content types
            "has_images": len(image_metadata) > 0,
            "content_types": ["text"] + (["images"] if image_metadata else [])
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
        
        # Log progress with image info
        if vector_count % 10 == 0:
            logger.info(f"Processed {vector_count} sections...")
            logger.info(f"Last section: {section['section_title']} ({len(valid_paragraphs)} paragraphs, {len(image_metadata)} images)")
    
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

def analyze_content_structure(sections: List[Dict], output_file: str = None) -> Dict[str, Any]:
    """Analyze how content will be combined without actually vectorizing."""
    analysis = {
        "analysis_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_sections": len(sections),
            "analysis_mode": "content_structure_preview"
        },
        "section_analysis": [],
        "statistics": {
            "sections_with_content": 0,
            "sections_with_images": 0,
            "total_paragraphs": 0,
            "total_images": 0,
            "total_characters": 0,
            "total_words": 0,
            "sections_by_depth": {},
            "image_formats": {},
            "large_sections": []
        }
    }
    
    for section in sections:
        # Filter out None values and empty paragraphs
        valid_paragraphs = [p for p in section['paragraphs'] if p and p.get('content')] if section['paragraphs'] else []
        
        # Process image metadata
        image_metadata = []
        if section['images']:
            for img in section['images']:
                if img:  # Filter out None values
                    image_metadata.append({
                        'image_id': img['image_id'],
                        'paragraph_id': img['paragraph_id'],
                        'image_format': img['image_format'],
                        'image_size_bytes': img['image_size_bytes'],
                        'width': img['width'],
                        'height': img['height'],
                        'image_order': img['image_order']
                    })
        
        if not valid_paragraphs:
            continue
            
        # Combine paragraphs into a single text (this is what would be vectorized)
        section_text = "\n\n".join(p['content'] for p in valid_paragraphs)
        
        # Calculate text statistics
        char_count = len(section_text)
        word_count = len(section_text.split())
        estimated_tokens = char_count // 4  # Rough estimate
        
        # Determine if section would be truncated
        would_truncate = char_count > 30000
        
        section_analysis = {
            "section_info": {
                "section_id": str(section['section_id']),
                "section_title": section['section_title'],
                "node_path": section['node_path'],
                "depth": section['depth'],
                "page_number": section['page_number'],
                "parent_title": section['parent_title'],
                "parent_path": section['parent_path']
            },
            "content_structure": {
                "paragraph_count": len(valid_paragraphs),
                "image_count": len(image_metadata),
                "has_images": len(image_metadata) > 0,
                "content_types": ["text"] + (["images"] if image_metadata else [])
            },
            "text_analysis": {
                "total_characters": char_count,
                "total_words": word_count,
                "estimated_tokens": estimated_tokens,
                "would_truncate": would_truncate,
                "truncated_length": 30000 if would_truncate else char_count,
                "text_preview": section_text[:200] + "..." if len(section_text) > 200 else section_text,
                "text_ending": "..." + section_text[-200:] if len(section_text) > 200 else ""
            },
            "paragraphs": [
                {
                    "paragraph_id": p['paragraph_id'],
                    "order": p['paragraph_order'],
                    "word_count": p['word_count'],
                    "char_count": p['char_count'],
                    "content_preview": p['content'][:100] + "..." if len(p['content']) > 100 else p['content']
                }
                for p in valid_paragraphs[:5]  # Show first 5 paragraphs
            ] + ([{"note": f"... and {len(valid_paragraphs) - 5} more paragraphs"}] if len(valid_paragraphs) > 5 else []),
            "images": image_metadata,
            "combined_text_sample": {
                "method": "paragraphs joined with double newlines",
                "example": "\n\n".join(p['content'][:50] + "..." for p in valid_paragraphs[:3])
            }
        }
        
        analysis["section_analysis"].append(section_analysis)
        
        # Update statistics
        stats = analysis["statistics"]
        stats["sections_with_content"] += 1
        if image_metadata:
            stats["sections_with_images"] += 1
        stats["total_paragraphs"] += len(valid_paragraphs)
        stats["total_images"] += len(image_metadata)
        stats["total_characters"] += char_count
        stats["total_words"] += word_count
        
        # Track depth distribution
        depth = section['depth']
        stats["sections_by_depth"][depth] = stats["sections_by_depth"].get(depth, 0) + 1
        
        # Track image formats
        for img in image_metadata:
            fmt = img['image_format']
            stats["image_formats"][fmt] = stats["image_formats"].get(fmt, 0) + 1
        
        # Track large sections
        if would_truncate:
            stats["large_sections"].append({
                "section_title": section['section_title'],
                "node_path": section['node_path'],
                "characters": char_count,
                "estimated_tokens": estimated_tokens
            })
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"content_analysis_{timestamp}.json"
    
    # Save analysis to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    return analysis, output_file

def analyze_content_structure_chunked(sections: List[Dict], output_file: str = None, paragraphs_per_chunk: int = 3) -> Dict[str, Any]:
    """Analyze how content will be combined with chunking without actually vectorizing."""
    analysis = {
        "analysis_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_sections": len(sections),
            "analysis_mode": "chunked_content_structure_preview",
            "paragraphs_per_chunk": paragraphs_per_chunk
        },
        "chunk_analysis": [],
        "statistics": {
            "sections_with_content": 0,
            "sections_with_images": 0,
            "total_paragraphs": 0,
            "total_images": 0,
            "total_characters": 0,
            "total_words": 0,
            "total_chunks": 0,
            "sections_by_depth": {},
            "image_formats": {},
            "large_chunks": [],
            "chunks_per_section_distribution": {}
        }
    }
    
    for section in sections:
        # Filter out None values and empty paragraphs
        valid_paragraphs = [p for p in section['paragraphs'] if p and p.get('content')] if section['paragraphs'] else []
        
        if not valid_paragraphs:
            continue
        
        # Process image metadata once for the section
        section_image_metadata = []
        if section['images']:
            for img in section['images']:
                if img:  # Filter out None values
                    section_image_metadata.append({
                        'image_id': img['image_id'],
                        'paragraph_id': img['paragraph_id'],
                        'image_format': img['image_format'],
                        'image_size_bytes': img['image_size_bytes'],
                        'width': img['width'],
                        'height': img['height'],
                        'image_order': img['image_order']
                    })
        
        # Split paragraphs into chunks
        paragraph_chunks = chunk_paragraphs_by_groups(valid_paragraphs, paragraphs_per_chunk)
        
        # Track chunks per section
        chunk_count = len(paragraph_chunks)
        stats = analysis["statistics"]
        stats["chunks_per_section_distribution"][chunk_count] = stats["chunks_per_section_distribution"].get(chunk_count, 0) + 1
        
        for chunk_idx, paragraph_chunk in enumerate(paragraph_chunks):
            # Combine paragraphs in this chunk
            chunk_text = "\n\n".join(p['content'] for p in paragraph_chunk)
            
            # Skip if chunk is too small
            if len(chunk_text) < 50:
                continue
            
            # Calculate statistics
            char_count = len(chunk_text)
            word_count = len(chunk_text.split())
            estimated_tokens = char_count // 4
            
            # Determine if chunk would be truncated
            would_truncate = char_count > 30000
            
            # Filter images that belong to paragraphs in this chunk
            chunk_paragraph_ids = {p['paragraph_id'] for p in paragraph_chunk}
            chunk_images = [img for img in section_image_metadata 
                           if img['paragraph_id'] in chunk_paragraph_ids]
            
            chunk_analysis = {
                "section_info": {
                    "section_id": str(section['section_id']),
                    "section_title": section['section_title'],
                    "node_path": section['node_path'],
                    "depth": section['depth'],
                    "page_number": section['page_number'],
                    "parent_title": section['parent_title'],
                    "parent_path": section['parent_path']
                },
                "chunk_info": {
                    "chunk_index": chunk_idx,
                    "total_chunks_in_section": len(paragraph_chunks),
                    "is_first_chunk": chunk_idx == 0,
                    "is_last_chunk": chunk_idx == len(paragraph_chunks) - 1,
                    "readable_chunk_id": f"{section['section_id']}_chunk_{chunk_idx}"
                },
                "content_structure": {
                    "paragraph_count": len(paragraph_chunk),
                    "image_count": len(chunk_images),
                    "has_images": len(chunk_images) > 0,
                    "content_types": ["text"] + (["images"] if chunk_images else [])
                },
                "text_analysis": {
                    "total_characters": char_count,
                    "total_words": word_count,
                    "estimated_tokens": estimated_tokens,
                    "would_truncate": would_truncate,
                    "truncated_length": 30000 if would_truncate else char_count,
                    "text_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    "text_ending": "..." + chunk_text[-200:] if len(chunk_text) > 200 else ""
                },
                "paragraphs": [
                    {
                        "paragraph_id": p['paragraph_id'],
                        "order": p['paragraph_order'],
                        "word_count": p['word_count'],
                        "char_count": p['char_count'],
                        "content_preview": p['content'][:100] + "..." if len(p['content']) > 100 else p['content']
                    }
                    for p in paragraph_chunk
                ],
                "images": chunk_images,
                "combined_text_sample": {
                    "method": "paragraphs joined with double newlines",
                    "example": "\n\n".join(p['content'][:50] + "..." for p in paragraph_chunk[:2])
                }
            }
            
            analysis["chunk_analysis"].append(chunk_analysis)
            
            # Update statistics
            stats["total_chunks"] += 1
            if chunk_images:
                stats["sections_with_images"] += 1 if chunk_idx == 0 else 0  # Count section once
            stats["total_paragraphs"] += len(paragraph_chunk)
            stats["total_images"] += len(chunk_images)
            stats["total_characters"] += char_count
            stats["total_words"] += word_count
            
            # Track large chunks
            if would_truncate:
                stats["large_chunks"].append({
                    "section_title": section['section_title'],
                    "node_path": section['node_path'],
                    "chunk_index": chunk_idx,
                    "characters": char_count,
                    "estimated_tokens": estimated_tokens
                })
        
        # Update section-level statistics
        if valid_paragraphs:
            stats["sections_with_content"] += 1
            
        # Track depth distribution
        depth = section['depth']
        stats["sections_by_depth"][depth] = stats["sections_by_depth"].get(depth, 0) + 1
        
        # Track image formats
        for img in section_image_metadata:
            fmt = img['image_format']
            stats["image_formats"][fmt] = stats["image_formats"].get(fmt, 0) + 1
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"chunked_analysis_{paragraphs_per_chunk}para_{timestamp}.json"
    
    # Save analysis to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    return analysis, output_file

def main():
    parser = argparse.ArgumentParser(description='Vectorize book sections with image metadata and store in Qdrant')
    parser.add_argument('--book', '-b', required=True, help='Book title to vectorize (can be partial match)')
    parser.add_argument('--collection', '-c', help='Qdrant collection name to use (not needed for analysis)')
    parser.add_argument('--show-stats', '-s', action='store_true', help='Show vector storage statistics')
    parser.add_argument('--analyze-only', '-a', action='store_true', help='Only analyze content structure without vectorizing')
    parser.add_argument('--output', '-o', help='Output file for analysis (only used with --analyze-only)')
    parser.add_argument('--chunk-size', '-cs', type=int, default=0, help='Chunk size for paragraph-based chunking (0 = no chunking, use whole sections)')
    args = parser.parse_args()

    # Validate arguments
    if not args.analyze_only and not args.collection:
        parser.error("--collection is required unless using --analyze-only")

    # Test all connections first (skip OpenAI for analysis-only mode)
    logger.info("Testing connections...")
    if args.analyze_only:
        # Only test PostgreSQL for analysis mode
        if not test_postgres_connection():
            logger.error("PostgreSQL connection test failed. Exiting.")
            return
        logger.info("✓ PostgreSQL connection test passed")
    else:
        if not test_connections():
            logger.error("Connection tests failed. Exiting.")
            return

    # Get book ID
    book_id = get_book_id(args.book)
    if not book_id:
        return

    # Get sections with content and images
    logger.info("Fetching sections with content and images...")
    sections = get_sections_with_content(book_id)
    
    # Count sections with images - fix the counting logic
    sections_with_images = 0
    total_images = 0
    
    for section in sections:
        if section.get('images') and section['images'] != [None]:
            # Filter out None values and count actual images
            actual_images = [img for img in section['images'] if img is not None]
            if actual_images:
                sections_with_images += 1
                total_images += len(actual_images)
    
    logger.info(f"Found {len(sections)} sections to process")
    logger.info(f"Sections with images: {sections_with_images}")
    logger.info(f"Total images: {total_images}")

    if args.analyze_only:
        # Analysis mode - no vectorization
        if args.chunk_size > 0:
            logger.info(f"Running content structure analysis with chunking ({args.chunk_size} paragraphs per chunk)...")
            analysis, output_file = analyze_content_structure_chunked(sections, args.output, args.chunk_size)
        else:
            logger.info("Running content structure analysis (whole sections)...")
            analysis, output_file = analyze_content_structure(sections, args.output)
        
        # Log analysis summary
        stats = analysis["statistics"]
        logger.info("\nContent Structure Analysis Summary:")
        logger.info(f"Book ID: {book_id}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Sections with content: {stats['sections_with_content']}")
        logger.info(f"Sections with images: {stats['sections_with_images']}")
        logger.info(f"Total paragraphs: {stats['total_paragraphs']}")
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Total characters: {stats['total_characters']:,}")
        logger.info(f"Total words: {stats['total_words']:,}")
        
        # Handle different analysis types
        if 'total_chunks' in stats:
            # Chunked analysis
            logger.info(f"Total chunks: {stats['total_chunks']}")
            logger.info(f"Large chunks (>30k chars): {len(stats['large_chunks'])}")
            if stats.get('chunks_per_section_distribution'):
                logger.info(f"Chunks per section: {dict(stats['chunks_per_section_distribution'])}")
        else:
            # Whole section analysis
            logger.info(f"Large sections (>30k chars): {len(stats['large_sections'])}")
        
        if stats['image_formats']:
            logger.info(f"Image formats: {dict(stats['image_formats'])}")
        
        if stats['sections_by_depth']:
            logger.info(f"Sections by depth: {dict(stats['sections_by_depth'])}")
            
        logger.info(f"\n✅ Analysis complete! Check {output_file} for detailed breakdown.")
        return

    # Normal vectorization mode
    # Ensure collection exists
    ensure_collection_exists(args.collection)
    
    # Connect to PostgreSQL for metadata storage
    pg_conn = psycopg2.connect(**PG_PARAMS)
    
    try:
        # Store vectors and get size information
        if args.chunk_size > 0:
            logger.info(f"Using chunked vectorization with {args.chunk_size} paragraphs per chunk")
            size_info = store_section_vectors_chunked(sections, args.collection, pg_conn, args.chunk_size)
        else:
            logger.info("Using whole-section vectorization (no chunking)")
            size_info = store_section_vectors(sections, args.collection, pg_conn)
        
        # Log summary
        logger.info("\nVectorization Summary:")
        logger.info(f"Book ID: {book_id}")
        logger.info(f"Collection: {args.collection}")
        logger.info(f"Total sections processed: {size_info['vector_count']}")
        logger.info(f"Sections with images: {sections_with_images}")
        logger.info(f"Total images linked: {total_images}")
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