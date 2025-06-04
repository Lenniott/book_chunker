# Query Enhancement System Plan

## Overview
This plan outlines the implementation of an enhanced query system for the book chunker that will handle both specific lesson queries and broader, non-lesson-specific queries. The system will use a query understanding agent to transform user questions into effective search queries and provide context-aware responses.

## Current Limitations
- System primarily handles lesson-specific queries
- Limited ability to find relevant content for broader questions
- No context awareness in search results
- Single-level vector storage (chapter level only)
- No tracking of vector storage size

## Proposed Enhancements

### 1. Query Understanding Agent
- [ ] Create `query_agent.py` for query transformation
- [ ] Implement query expansion to generate multiple semantic search queries
- [ ] Add context awareness to query understanding
- [ ] Support both specific and broad query types
- [ ] Include metadata about search scope and context requirements
- [ ] Add command-line arguments for book and collection selection

### 2. Enhanced Vector Storage
- [ ] Add paragraph-level vector storage
- [ ] Include rich context metadata with vectors
- [ ] Maintain relationships between chapters and paragraphs
- [ ] Update database schema to support new storage requirements
- [ ] Implement efficient vector search at both levels
- [ ] Add vector storage size tracking and logging
- [ ] Calculate and log total vector space usage

### 3. Context-Aware Search
- [ ] Implement enhanced search functionality
- [ ] Support multi-level search (chapter and paragraph)
- [ ] Add result ranking and combination
- [ ] Include context in search results
- [ ] Support related concept search
- [ ] Add collection-specific search parameters

### 4. Database Updates
- [ ] Add search_level column to vector_metadata
- [ ] Add context_metadata JSONB column
- [ ] Create new paragraph_vectors table
- [ ] Update existing vector storage procedures
- [ ] Add indexes for efficient querying
- [ ] Add vector_size tracking columns

## Implementation Priority
1. Query Understanding Agent (High)
   - Most immediate impact on search quality
   - Can be implemented without major database changes
   - Will improve current search capabilities
   - Add book/collection parameters first

2. Enhanced Vector Storage (Medium)
   - Requires database schema updates
   - More complex implementation
   - Needed for paragraph-level search
   - Add size tracking and logging

3. Context-Aware Search (Medium)
   - Builds on Query Understanding Agent
   - Requires both agent and storage enhancements
   - Key for improved search results
   - Collection-specific implementation

4. Database Updates (Low)
   - Can be implemented gradually
   - Required for full system enhancement
   - Can be done in parallel with other work

## Technical Details

### Query Understanding Agent
```python
class QueryUnderstandingAgent:
    def __init__(self, book_title: str, collection_name: str):
        self.book_title = book_title
        self.collection_name = collection_name

    def expand_query(self, user_query: str) -> List[str]
    def get_context_aware_query(self, user_query: str, book_context: Dict) -> Dict
```

### Vector Storage with Size Tracking
```python
def store_paragraph_vectors(chapter_id: str, paragraphs: List[Dict], collection: str) -> Dict[str, int]:
    """
    Store vectors and return size information
    Returns: Dict with 'total_size_bytes', 'vector_count', 'avg_vector_size'
    """
    # Implementation here

def log_vector_storage_info(collection: str) -> Dict[str, Any]:
    """
    Get and log vector storage information from Qdrant
    Returns: Dict with collection stats including total size
    """
    # Implementation here
```

### Search Function
```python
class EnhancedBookSearch:
    def __init__(self, book_title: str, collection_name: str, query_agent: QueryUnderstandingAgent):
        self.book_title = book_title
        self.collection_name = collection_name
        self.query_agent = query_agent

    async def search(self, user_query: str, book_context: Dict) -> List[Dict]
```

### Database Schema Updates
```sql
-- Add size tracking columns
ALTER TABLE vector_metadata
ADD COLUMN search_level VARCHAR(10),
ADD COLUMN context_metadata JSONB,
ADD COLUMN vector_size_bytes BIGINT,
ADD COLUMN storage_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW();

CREATE TABLE paragraph_vectors (
    id UUID PRIMARY KEY,
    content_node_id UUID,
    paragraph_order INTEGER,
    vector_id VARCHAR(255),
    context_metadata JSONB,
    vector_size_bytes BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add size tracking view
CREATE VIEW vector_storage_stats AS
SELECT 
    collection_name,
    COUNT(*) as vector_count,
    SUM(vector_size_bytes) as total_size_bytes,
    AVG(vector_size_bytes) as avg_vector_size,
    MAX(storage_timestamp) as last_updated
FROM vector_metadata
GROUP BY collection_name;
```

## Command Line Interface
```python
def main():
    parser = argparse.ArgumentParser(description='Enhanced Book Query System')
    parser.add_argument('--book', '-b', required=True, help='Book title to use')
    parser.add_argument('--collection', '-c', required=True, help='Qdrant collection name')
    parser.add_argument('--show-stats', '-s', action='store_true', help='Show vector storage statistics')
    args = parser.parse_args()
```

## Vector Size Tracking
- Track size of each vector in bytes
- Log total collection size
- Calculate average vector size
- Monitor storage growth over time
- Provide size estimates before operations

## Future Considerations
- Add support for multiple books
- Implement query caching
- Add user feedback loop for query improvement
- Support for multiple languages
- Integration with external knowledge bases
- Storage optimization strategies
- Automated cleanup of unused vectors

## Success Metrics
- Improved search result relevance
- Higher user satisfaction with responses
- Better handling of broad queries
- Reduced need for query refinement
- Faster response times
- Efficient vector storage usage
- Accurate size tracking and reporting 