CREATE OR REPLACE FUNCTION get_lesson_paragraphs(
    p_lesson_number INTEGER
)
RETURNS TABLE (
    node_title TEXT,
    node_path TEXT,
    content TEXT,
    paragraph_order INTEGER,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
DECLARE
    v_node_path TEXT;
BEGIN
    -- Determine the node path based on the lesson number
    IF p_lesson_number BETWEEN 1 AND 50 THEN
        -- Regular lessons: 1-50
        v_node_path := '5.1.' || p_lesson_number;
    ELSIF p_lesson_number BETWEEN 51 AND 80 THEN
        -- Review I: Lessons 51-80
        v_node_path := '5.1.52.' || (p_lesson_number - 50);
    ELSIF p_lesson_number BETWEEN 81 AND 110 THEN
        -- Review II: Lessons 81-110
        v_node_path := '5.1.53.' || (p_lesson_number - 80);
    ELSIF p_lesson_number BETWEEN 111 AND 140 THEN
        -- Review III: Lessons 111-140
        v_node_path := '5.1.54.' || (p_lesson_number - 110);
    ELSIF p_lesson_number BETWEEN 141 AND 170 THEN
        -- Review IV: Lessons 141-170
        v_node_path := '5.1.55.' || (p_lesson_number - 140);
    ELSIF p_lesson_number BETWEEN 171 AND 200 THEN
        -- Review V: Lessons 171-200
        v_node_path := '5.1.56.' || (p_lesson_number - 170);
    ELSIF p_lesson_number BETWEEN 201 AND 220 THEN
        -- Review VI: Lessons 201-220
        v_node_path := '5.1.57.' || (p_lesson_number - 200);
    ELSIF p_lesson_number BETWEEN 221 AND 230 THEN
        -- 11. What Is Creation?
        v_node_path := '5.13.10' || (p_lesson_number - 220);
    ELSIF p_lesson_number BETWEEN 231 AND 339 THEN
        -- 12. What Is the Ego?
        v_node_path := '5.14.9' || (p_lesson_number - 230);
    ELSIF p_lesson_number BETWEEN 340 AND 365 THEN
        -- Final Lessons: 361-365
        v_node_path := '5.18.' || (p_lesson_number - 339);
    ELSE
        -- Handle other lesson ranges or raise an exception
        RAISE EXCEPTION 'Lesson number % is not supported in this function', p_lesson_number;
    END IF;

    -- Use the existing function to get paragraphs
    RETURN QUERY
    SELECT * FROM get_node_paragraphs('A Course in Miracles: Original Edition', string_to_array(v_node_path, '.')::INTEGER[]);

END;
$$ LANGUAGE plpgsql;
