from dotenv import load_dotenv
load_dotenv()
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import sys

def get_paragraphs_from_section(db_params, book_title, section_path):
    """
    Get paragraphs from a specific section of a book.
    section_path should be a list of section titles in order, e.g. ['Workbook For Students', 'Part I']
    """
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # First, get the book ID
        cur.execute("""
            SELECT id FROM books WHERE title = %s
        """, (book_title,))
        book = cur.fetchone()
        if not book:
            print(f"Book '{book_title}' not found")
            return
        
        book_id = book['id']
        
        # Build the recursive query to find the section
        query = """
        WITH RECURSIVE section_tree AS (
            -- Base case: get the root node (Workbook For Students)
            SELECT 
                cn.id,
                cn.title,
                cn.node_path,
                cn.depth,
                1 as path_index
            FROM content_nodes cn
            WHERE cn.book_id = %s 
            AND cn.title = %s
            AND cn.parent_id IS NULL
            
            UNION ALL
            
            -- Recursive case: get children
            SELECT 
                child.id,
                child.title,
                child.node_path,
                child.depth,
                st.path_index + 1
            FROM content_nodes child
            JOIN section_tree st ON child.parent_id = st.id
            WHERE child.title = %s
            AND st.path_index < array_length(%s, 1)
        )
        SELECT 
            cn.id,
            cn.title,
            cn.node_path,
            p.content,
            p.paragraph_order
        FROM section_tree st
        JOIN content_nodes cn ON cn.id = st.id
        LEFT JOIN paragraphs p ON p.content_node_id = cn.id
        WHERE st.path_index = array_length(%s, 1)
        ORDER BY p.paragraph_order;
        """
        
        # Execute the query with the section path
        cur.execute(query, (
            book_id,
            section_path[0],  # First section
            section_path[1],  # Second section (used in recursive part)
            section_path,     # Full path for array length
            section_path      # Full path for final filter
        ))
        
        results = cur.fetchall()
        
        if not results:
            print(f"No content found for path: {' > '.join(section_path)}")
            return
        
        # Print the results
        print(f"\nContent from: {' > '.join(section_path)}")
        print("-" * 80)
        current_paragraph = 1
        for row in results:
            if row['content']:
                print(f"\nParagraph {current_paragraph}:")
                print(row['content'])
                print("-" * 40)
                current_paragraph += 1
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    # Database connection parameters from environment
    db_params = {
        "dbname": os.getenv("PG_DBNAME"),
        "user": os.getenv("PG_USER"),
        "password": os.getenv("PG_PASSWORD"),
        "host": os.getenv("PG_HOST"),
        "port": os.getenv("PG_PORT"),
    }
    
    # Get paragraphs from the specified section
    book_title = "A Course in Miracles"
    section_path = ["Workbook For Students", "Part I"]
    
    get_paragraphs_from_section(db_params, book_title, section_path) 