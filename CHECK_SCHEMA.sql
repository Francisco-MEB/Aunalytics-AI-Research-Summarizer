-- Check the actual table structure

SELECT 
    column_name, 
    data_type, 
    udt_name
FROM information_schema.columns
WHERE table_name = 'documents'
ORDER BY ordinal_position;
