-- Add new columns to vector_metadata table
ALTER TABLE vector_metadata
ADD COLUMN IF NOT EXISTS vector_size_bytes INTEGER,
ADD COLUMN IF NOT EXISTS search_level VARCHAR(50),
ADD COLUMN IF NOT EXISTS context_metadata JSONB;

-- Add index on search_level for faster filtering
CREATE INDEX IF NOT EXISTS idx_vector_metadata_search_level 
ON vector_metadata(search_level);

-- Add index on context_metadata for faster JSON queries
CREATE INDEX IF NOT EXISTS idx_vector_metadata_context 
ON vector_metadata USING GIN (context_metadata);

-- Update existing rows to have default values
UPDATE vector_metadata 
SET 
    vector_size_bytes = 0,
    search_level = 'section',
    context_metadata = '{}'::jsonb
WHERE vector_size_bytes IS NULL; 