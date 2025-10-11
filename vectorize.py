#!/Users/benjamin/Desktop/CODE/book_chunker/venv/bin/python3
"""
Vectorize - Load structured book JSON into Qdrant vector database

Takes JSON files produced by book_extractor.py and:
1. Generates embeddings using OpenAI
2. Stores vectors in Qdrant with metadata
3. Optional paragraph-based chunking
4. Analysis mode to preview without vectorizing

No PostgreSQL dependency - all metadata stored in Qdrant payload.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import json
import sys
import logging
import argparse
from typing import List, Dict, Any, Tuple
from datetime import datetime
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import openai

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vectorization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup
openai.api_key = OPENAI_API_KEY
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def test_connections():
    """Test Qdrant and OpenAI connections."""
    try:
        # Test Qdrant
        collections = qdrant.get_collections()
        logger.info(f"✓ Qdrant connected ({len(collections.collections)} collections)")
        
        # Test OpenAI
        test_response = openai.embeddings.create(input="test", model=OPENAI_MODEL)
        logger.info(f"✓ OpenAI connected (model: {OPENAI_MODEL})")
        
        return True
    except Exception as e:
        logger.error(f"✗ Connection test failed: {str(e)}")
        return False


def load_book_json(json_path: str) -> Dict[str, Any]:
    """Load and validate book JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate required fields
        required = ['title', 'author', 'outline']
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        logger.info(f"Loaded: {data['title']} by {data['author']}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        sys.exit(1)


def vectorize_text(text: str) -> Tuple[List[float], int]:
    """
    Vectorize text using OpenAI embeddings.
    
    Returns:
        Tuple of (embedding vector, size in bytes)
    """
    # Truncate if too long (OpenAI limit ~8k tokens, ~32k chars)
    max_chars = 30000
    
    if len(text) > max_chars:
        logger.warning(f"Text too long ({len(text)} chars). Truncating to {max_chars}")
        text = text[:max_chars] + "... [TRUNCATED]"
    
    response = openai.embeddings.create(
        input=text,
        model=OPENAI_MODEL
    )
    embedding = response.data[0].embedding
    size_bytes = len(embedding) * 4  # 4 bytes per float
    return embedding, size_bytes


def chunk_paragraphs(paragraphs: List[str], chunk_size: int) -> List[List[str]]:
    """Split paragraphs into chunks of specified size."""
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        chunk = paragraphs[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def extract_sections_from_outline(outline: List[Dict], parent_path: str = "", parent_hierarchy: List[str] = None) -> List[Dict]:
    """
    Flatten hierarchical outline into list of sections with full context.
    
    Args:
        outline: List of chapter/section objects with content arrays
        parent_path: Numeric path (e.g., "1.2.3")
        parent_hierarchy: Full title hierarchy (e.g., ["Chapter 1", "Introduction"])
    
    Returns:
        List of section dictionaries with metadata
    """
    if parent_hierarchy is None:
        parent_hierarchy = []
    
    sections = []
    
    for idx, node in enumerate(outline):
        # Create paths for this node
        node_path = f"{parent_path}.{idx + 1}" if parent_path else str(idx + 1)
        current_hierarchy = parent_hierarchy + [node.get('title', 'Untitled')]
        
        # Collect all text content from this node's content array
        text_content = []
        content_index = 0
        
        if 'content' in node:
            for item in node['content']:
                if isinstance(item, str):
                    # It's a paragraph string
                    text_content.append(item)
                    content_index += 1
                elif isinstance(item, dict):
                    # It's a subsection - recurse into it
                    child_sections = extract_sections_from_outline(
                        [item], 
                        node_path, 
                        current_hierarchy
                    )
                    sections.extend(child_sections)
        
        # Create section entry if we have text content
        if text_content:
            section = {
                'title': node.get('title', 'Untitled'),
                'page': node.get('page', 0),
                'path': node_path,
                'paragraphs': text_content,
                'images': node.get('images', []),
                'parent_path': parent_path if parent_path else None,
                'section_hierarchy': ' > '.join(current_hierarchy),
                'depth_level': len(current_hierarchy) - 1,
                'content_index': content_index
            }
            sections.append(section)
    
    return sections


def vectorize_sections(book_data: Dict, collection_name: str, chunk_size: int = 0) -> Dict[str, int]:
    """
    Vectorize book sections and store in Qdrant.
    
    Args:
        book_data: Book JSON data
        collection_name: Qdrant collection name
        chunk_size: Paragraphs per chunk (0 = whole section)
        
    Returns:
        Statistics dictionary
    """
    logger.info(f"Extracting sections from outline...")
    sections = extract_sections_from_outline(book_data['outline'])
    logger.info(f"Found {len(sections)} sections")
    
    # Ensure collection exists
    ensure_collection_exists(collection_name)
    
    points = []
    total_size = 0
    vector_count = 0
    
    for section in sections:
        paragraphs = section['paragraphs']
        if not paragraphs:
            continue
        
        # Chunking logic
        if chunk_size > 0 and len(paragraphs) > chunk_size:
            # Split into chunks
            para_chunks = chunk_paragraphs(paragraphs, chunk_size)
            
            for chunk_idx, para_chunk in enumerate(para_chunks):
                chunk_text = "\n\n".join(para_chunk)
                
                if len(chunk_text) < 50:  # Skip tiny chunks
                    continue
                
                # Vectorize
                try:
                    embedding, size_bytes = vectorize_text(chunk_text)
                except Exception as e:
                    logger.error(f"Failed to vectorize chunk {chunk_idx} of '{section['title']}': {e}")
                    continue
                
                # Create metadata
                metadata = {
                    "book_title": book_data['title'],
                    "book_author": book_data['author'],
                    "section_title": section['title'],
                    "section_path": section['path'],
                    "section_hierarchy": section.get('section_hierarchy', ''),
                    "depth_level": section.get('depth_level', 0),
                    "content_index": section.get('content_index', 0),
                    "page": section['page'],
                    "parent_path": section['parent_path'],
                    "chunk_index": chunk_idx,
                    "total_chunks": len(para_chunks),
                    "is_first_chunk": chunk_idx == 0,
                    "is_last_chunk": chunk_idx == len(para_chunks) - 1,
                    "paragraph_count": len(para_chunk),
                    "text": chunk_text,
                    "has_images": len(section.get('images', [])) > 0,
                    "image_count": len(section.get('images', [])),
                    "chunking_method": "paragraph_groups",
                    "paragraphs_per_chunk": chunk_size
                }
                
                # Add point
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                ))
                
                total_size += size_bytes
                vector_count += 1
                
                if vector_count % 10 == 0:
                    logger.info(f"Processed {vector_count} chunks...")
        else:
            # Whole section
            section_text = "\n\n".join(paragraphs)
            
            if len(section_text) < 50:
                continue
            
            # Vectorize
            try:
                embedding, size_bytes = vectorize_text(section_text)
            except Exception as e:
                logger.error(f"Failed to vectorize section '{section['title']}': {e}")
                continue
            
            # Create metadata
            metadata = {
                "book_title": book_data['title'],
                "book_author": book_data['author'],
                "section_title": section['title'],
                "section_path": section['path'],
                "section_hierarchy": section.get('section_hierarchy', ''),
                "depth_level": section.get('depth_level', 0),
                "content_index": section.get('content_index', 0),
                "page": section['page'],
                "parent_path": section['parent_path'],
                "paragraph_count": len(paragraphs),
                "text": section_text,
                "has_images": len(section.get('images', [])) > 0,
                "image_count": len(section.get('images', [])),
                "chunking_method": "whole_section"
            }
            
            # Add point
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=metadata
            ))
            
            total_size += size_bytes
            vector_count += 1
            
            if vector_count % 10 == 0:
                logger.info(f"Processed {vector_count} sections...")
    
    # Batch upload to Qdrant
    if points:
        logger.info(f"Uploading {len(points)} vectors to Qdrant...")
        qdrant.upsert(
            collection_name=collection_name,
            points=points
        )
    
    return {
        "vector_count": vector_count,
        "total_size_bytes": total_size,
        "avg_vector_size": total_size / vector_count if vector_count > 0 else 0
    }


def analyze_content(book_data: Dict, chunk_size: int = 0) -> Dict[str, Any]:
    """
    Analyze content structure without vectorizing.
    
    Args:
        book_data: Book JSON data
        chunk_size: Paragraphs per chunk (0 = whole section)
        
    Returns:
        Analysis results
    """
    sections = extract_sections_from_outline(book_data['outline'])
    
    analysis = {
        "book_title": book_data['title'],
        "book_author": book_data['author'],
        "total_sections": len(sections),
        "sections_with_content": 0,
        "total_paragraphs": 0,
        "total_images": 0,
        "estimated_chunks": 0,
        "largest_section": None,
        "largest_section_size": 0
    }
    
    for section in sections:
        paragraphs = section['paragraphs']
        if not paragraphs:
            continue
        
        analysis["sections_with_content"] += 1
        analysis["total_paragraphs"] += len(paragraphs)
        analysis["total_images"] += len(section.get('images', []))
        
        section_text = "\n\n".join(paragraphs)
        section_size = len(section_text)
        
        if section_size > analysis["largest_section_size"]:
            analysis["largest_section_size"] = section_size
            analysis["largest_section"] = section['title']
        
        # Estimate chunks
        if chunk_size > 0 and len(paragraphs) > chunk_size:
            analysis["estimated_chunks"] += (len(paragraphs) + chunk_size - 1) // chunk_size
        else:
            analysis["estimated_chunks"] += 1
    
    return analysis


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


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Vectorize book JSON files and load into Qdrant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze without vectorizing
  python vectorize.py book_structure.json --analyze-only
  
  # Vectorize whole sections
  python vectorize.py book_structure.json --collection my_books
  
  # Vectorize with 3-paragraph chunks
  python vectorize.py book_structure.json --collection my_books --chunk-size 3
  
  # Batch process multiple books
  python vectorize.py book1.json book2.json --collection my_library

Configuration:
  Set environment variables in .env file:
  - QDRANT_URL: Qdrant server URL
  - QDRANT_API_KEY: Qdrant API key
  - OPENAI_API_KEY: OpenAI API key
        """
    )
    
    parser.add_argument('json_files', nargs='+', help='JSON file(s) to vectorize')
    parser.add_argument('--collection', '-c', help='Qdrant collection name')
    parser.add_argument('--chunk-size', '-s', type=int, default=0,
                       help='Paragraphs per chunk (0 = whole sections)')
    parser.add_argument('--analyze-only', '-a', action='store_true',
                       help='Analyze content without vectorizing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.analyze_only and not args.collection:
        parser.error("--collection is required unless using --analyze-only")
    
    # Test connections (skip OpenAI for analysis mode)
    if not args.analyze_only:
        logger.info("Testing connections...")
        if not test_connections():
            logger.error("Connection tests failed. Exiting.")
            sys.exit(1)
    
    # Process each JSON file
    for json_path in args.json_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {json_path}")
        logger.info(f"{'='*60}")
        
        # Load JSON
        book_data = load_book_json(json_path)
        
        if args.analyze_only:
            # Analysis mode
            analysis = analyze_content(book_data, args.chunk_size)
            
            print(f"\n📊 Content Analysis:")
            print(f"  Book: {analysis['book_title']} by {analysis['book_author']}")
            print(f"  Total sections: {analysis['total_sections']}")
            print(f"  Sections with content: {analysis['sections_with_content']}")
            print(f"  Total paragraphs: {analysis['total_paragraphs']}")
            print(f"  Total images: {analysis['total_images']}")
            print(f"  Estimated vectors: {analysis['estimated_chunks']}")
            if args.chunk_size > 0:
                print(f"  Chunking: {args.chunk_size} paragraphs per chunk")
            else:
                print(f"  Chunking: Whole sections (no chunking)")
            print(f"  Largest section: {analysis['largest_section']}")
            print(f"  Largest section size: {analysis['largest_section_size']:,} characters")
        else:
            # Vectorization mode
            stats = vectorize_sections(book_data, args.collection, args.chunk_size)
            
            print(f"\n✅ Vectorization complete!")
            print(f"  Collection: {args.collection}")
            print(f"  Vectors created: {stats['vector_count']}")
            print(f"  Total size: {stats['total_size_bytes'] / (1024*1024):.2f} MB")
            print(f"  Average vector size: {stats['avg_vector_size'] / 1024:.2f} KB")


if __name__ == '__main__':
    main()

