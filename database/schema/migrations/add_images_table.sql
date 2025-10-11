-- Create images table to store base64 image data
CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_node_id UUID NOT NULL REFERENCES content_nodes(id) ON DELETE CASCADE,
    paragraph_id UUID REFERENCES paragraphs(id) ON DELETE CASCADE,
    image_data TEXT NOT NULL, -- base64 encoded image
    image_format VARCHAR(10), -- jpg, png, etc.
    image_size_bytes INTEGER,
    width INTEGER,
    height INTEGER,
    image_order INTEGER DEFAULT 0, -- order within the paragraph/node
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_images_content_node_id ON images(content_node_id);
CREATE INDEX IF NOT EXISTS idx_images_paragraph_id ON images(paragraph_id);
CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at);

-- Add index on image_order for proper ordering
CREATE INDEX IF NOT EXISTS idx_images_order ON images(content_node_id, image_order); 