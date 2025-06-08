import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Database connection parameters
DB_PARAMS = {
    "host": "localhost",
    "database": "brickbrain",
    "user": "brickbrain",
    "password": "brickbrain_password",
    "port": 5432
}

# Path to data files
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                       "data", "rebrickable")

# Mapping of CSV files to table names
CSV_TABLE_MAPPING = {
    "colors.csv": "colors",
    "elements.csv": "elements",
    "inventories.csv": "inventories",
    "inventory_minifigs.csv": "inventory_minifigs",
    "inventory_parts.csv": "inventory_parts",
    "inventory_sets.csv": "inventory_sets",
    "minifigs.csv": "minifigs",
    "part_categories.csv": "part_categories",
    "part_relationships.csv": "part_relationships",
    "parts.csv": "parts",
    "sets.csv": "sets",
    "themes.csv": "themes",
}

def connect_to_db():
    """Connect to the PostgreSQL database"""
    try:
        connection = psycopg2.connect(**DB_PARAMS)
        return connection
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def load_csv_to_db(connection, csv_file, table_name):
    """Load data from CSV file to the specified database table"""
    try:
        # Read the CSV file with proper handling for missing values
        df = pd.read_csv(os.path.join(DATA_DIR, csv_file), na_values=['', 'NA', 'NULL'])
        
        # Convert column names to lowercase for consistency with PostgreSQL
        df.columns = [col.lower() for col in df.columns]
        
        logger.info(f"Loading {len(df)} rows from {csv_file} into {table_name}")
        
        # Handle specific tables and their data issues
        if table_name == 'colors':
            # Convert integer columns to string first to avoid overflow errors
            if 'num_parts' in df.columns:
                df['num_parts'] = df['num_parts'].fillna(0).astype(str)
            if 'num_sets' in df.columns:
                df['num_sets'] = df['num_sets'].fillna(0).astype(str)
            if 'y1' in df.columns:
                df['y1'] = df['y1'].fillna(0)
            if 'y2' in df.columns:
                df['y2'] = df['y2'].fillna(0)
        elif table_name == 'elements':
            # Handle the elements table specifically
            df = df.where(pd.notnull(df), None)
            # Only keep rows where part_num and color_id are not null
            df = df.dropna(subset=['part_num', 'color_id'])
            # Convert color_id to integer
            df['color_id'] = df['color_id'].astype(int)
            # Convert design_id to integer where possible
            df['design_id'] = pd.to_numeric(df['design_id'], errors='ignore')
        
        # Replace NaN values with None for proper SQL NULL values
        df = df.where(pd.notnull(df), None)
        
        # Create cursor
        cursor = connection.cursor()
        
        # Track success and failure counts
        success_count = 0
        failure_count = 0
        
        if not df.empty:
            columns = list(df.columns)
            
            # Create INSERT statements manually for each row to avoid type conversion issues
            for index, row in df.iterrows():
                values = []
                for val in row:
                    if val is None:
                        values.append('NULL')
                    elif isinstance(val, str):
                        escaped_val = val.replace("'", "''")
                        values.append(f"'{escaped_val}'")
                    elif isinstance(val, bool):
                        values.append('TRUE' if val else 'FALSE')
                    elif pd.isna(val):  # Check for NaN values
                        values.append('NULL')
                    else:
                        values.append(str(val))
                
                insert_query = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({', '.join(values)})
                    ON CONFLICT DO NOTHING;
                """
                try:
                    cursor.execute(insert_query)
                    success_count += 1
                    # Print progress every 1000 rows
                    if success_count % 1000 == 0:
                        logger.info(f"Progress: {success_count} rows inserted into {table_name}")
                except Exception as e:
                    # More detailed error message including table name, row index, and specific fields
                    primary_key_val = None
                    if table_name == 'colors' and 'id' in row:
                        primary_key_val = row['id']
                    elif table_name == 'parts' and 'part_num' in row:
                        primary_key_val = row['part_num']
                    elif table_name == 'elements' and 'element_id' in row:
                        primary_key_val = row['element_id']
                    
                    error_msg = f"Failed to insert row into {table_name} (PK: {primary_key_val}, Row #{index}): {str(e)[:100]}"
                    logger.error(error_msg)
                    logger.error(f"Row data: {dict(row)}")
                    
                    # Rollback the transaction and raise a custom exception to stop processing
                    connection.rollback()
                    raise Exception(f"Stopping import due to error: {error_msg}")
        
        # Commit the transaction
        connection.commit()
        cursor.close()
        
        logger.info(f"Completed loading {table_name}: {success_count} rows inserted, {failure_count} rows failed")
        
    except Exception as e:
        logger.error(f"Error loading {csv_file} to {table_name}: {e}")
        connection.rollback()
        raise

def main():
    """Main function to upload all data files to the database"""
    try:
        # Connect to database
        connection = connect_to_db()
        connection.autocommit = False
        
        # First, run the schema creation script if it exists
        schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  "db", "rebrickable_schema.sql")
        
        if os.path.exists(schema_path):
            logger.info("Creating database schema...")
            with open(schema_path, 'r') as schema_file:
                schema_script = schema_file.read()
            
            # Execute the schema script statement by statement
            cursor = connection.cursor()
            
            # Split script into individual statements and execute each one
            current_statement = ""
            for line in schema_script.split('\n'):
                # Skip comment lines
                if line.strip().startswith('--'):
                    continue
                    
                current_statement += line + "\n"
                
                # If we find a statement terminator, execute the statement
                if line.strip().endswith(';'):
                    if current_statement.strip():
                        try:
                            cursor.execute(current_statement)
                            connection.commit()
                        except Exception as e:
                            logger.error(f"Error executing SQL: {e}")
                            connection.rollback()
                    current_statement = ""
            
            cursor.close()
            logger.info("Schema created successfully")
        else:
            logger.warning(f"Schema file not found at {schema_path}")
        
        # Disable foreign key constraints for data loading
        cursor = connection.cursor()
        logger.info("Disabling foreign key constraints...")
        cursor.execute("SET session_replication_role = replica;")
        connection.commit()
        cursor.close()
        
        # Define the order of table loading to respect foreign key constraints
        ordered_tables = [
            "colors.csv",
            "themes.csv",
            "sets.csv",
            "minifigs.csv",
            "part_categories.csv",
            "parts.csv",
            "part_relationships.csv",
            "elements.csv",
            "inventories.csv",
            "inventory_minifigs.csv",
            "inventory_parts.csv",
            "inventory_sets.csv"
        ]
        
        # Process each CSV file in order
        for csv_file in ordered_tables:
            if os.path.exists(os.path.join(DATA_DIR, csv_file)):
                table_name = CSV_TABLE_MAPPING.get(csv_file)
                if table_name:
                    logger.info(f"=== Starting import of {csv_file} into {table_name} ===")
                    load_csv_to_db(connection, csv_file, table_name)
                    logger.info(f"=== Completed import of {csv_file} ===")
                else:
                    logger.warning(f"No table mapping found for {csv_file}")
            else:
                logger.warning(f"CSV file not found: {os.path.join(DATA_DIR, csv_file)}")
        
        # Re-enable foreign key constraints
        cursor = connection.cursor()
        logger.info("Re-enabling foreign key constraints...")
        cursor.execute("SET session_replication_role = DEFAULT;")
        connection.commit()
        cursor.close()
        
        # Verify table counts
        cursor = connection.cursor()
        for table_name in set(CSV_TABLE_MAPPING.values()):
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                logger.info(f"Table {table_name} contains {count:,} rows")
            except Exception as e:
                logger.error(f"Could not count rows in {table_name}: {e}")
        cursor.close()
        
        # Close the connection
        connection.close()
        logger.info("Data upload completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to upload data: {e}")
        # Make sure to re-enable constraints even if there's an error
        try:
            cursor = connection.cursor()
            cursor.execute("SET session_replication_role = DEFAULT;")
            connection.commit()
            cursor.close()
        except:
            pass
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()