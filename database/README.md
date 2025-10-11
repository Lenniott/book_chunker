# Database Module

This directory contains PostgreSQL integration for storing book content in a relational database. This is **optional** - the main workflow stores all metadata directly in Qdrant.

## Why Use PostgreSQL?

The database layer is useful if you:
- Want to store book content separately from vectors
- Need relational queries across books
- Want to track versioning and changes
- Need a traditional database alongside vector search

## Schema

The schema includes:
- `books` - Book metadata (title, author, pages)
- `content_nodes` - Hierarchical sections with node paths
- `paragraphs` - Individual paragraphs with statistics
- `images` - Image metadata and base64 data
- `vector_metadata` - Tracking which sections are vectorized

## Setup

1. Create PostgreSQL database:
```bash
createdb book_library
```

2. Initialize schema:
```bash
psql book_library < database/schema/init_schema.sql
```

3. Apply migrations:
```bash
psql book_library < database/schema/migrations/add_images_table.sql
psql book_library < database/schema/migrations/add_vector_size_columns.sql
```

## Usage

Load a book JSON into PostgreSQL:
```bash
python database/db_loader.py book_structure.json
```

Delete a book:
```bash
python database/db_loader.py --delete <book_uuid>
```

## Configuration

Set environment variables in `.env`:
```
PG_DBNAME=book_library
PG_USER=postgres
PG_PASSWORD=your_password
PG_HOST=localhost
PG_PORT=5432
```

## Integration with Vectorization

The original `qdrant.py` script included PostgreSQL integration. If you need this:
1. Use the database loader to store books
2. Modify `vectorize.py` to read from PostgreSQL instead of JSON
3. Track vector metadata in the `vector_metadata` table

## Note

Most users don't need this complexity. The simplified workflow (JSON → Qdrant) stores all necessary metadata in Qdrant's payload and is easier to manage.

Consider using PostgreSQL if you're building a production system with multiple users, need complex queries, or want separation between storage and vectors.

