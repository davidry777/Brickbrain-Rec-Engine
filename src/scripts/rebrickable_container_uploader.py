#!/usr/bin/env python3
"""
Optimized data uploader specifically for container environments
Handles autovacuum conflicts and container performance issues
"""

import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging
import time
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Database connection parameters
DB_PARAMS = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "database": os.environ.get("DB_NAME", "brickbrain"),
    "user": os.environ.get("DB_USER", "brickbrain"),
    "password": os.environ.get("DB_PASSWORD", "brickbrain_password"),
    "port": int(os.environ.get("DB_PORT", "5432"))
}

# Path to data files (container-optimized)
DATA_DIR = "/app/data/rebrickable"

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

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("Received interrupt signal. Cleaning up...")
    sys.exit(0)

def optimize_postgres_for_bulk_load(connection):
    """Configure PostgreSQL for optimal bulk loading performance"""
    cursor = connection.cursor()
    try:
        # Core session optimizations (only parameters that can be set at session level)
        optimizations = [
            "SET synchronous_commit = OFF",
            "SET maintenance_work_mem = '512MB'",  # Increased for container
            "SET work_mem = '128MB'",              # Increased for container
            "SET temp_buffers = '64MB'",           # Increased for container
            "SET track_counts = OFF",              # Disable stats collection - prevents autovacuum
            # Remove server-level parameters that cause transaction failures
        ]
        
        for opt in optimizations:
            try:
                cursor.execute(opt)
                logger.debug(f"Applied: {opt}")
            except Exception as e:
                logger.warning(f"Could not apply {opt}: {e}")
                # Reset transaction on error
                connection.rollback()
                cursor = connection.cursor()
        
        connection.commit()
        logger.info("Applied container-optimized bulk loading settings")
        
    except Exception as e:
        logger.error(f"Error applying optimizations: {e}")
        connection.rollback()
    finally:
        cursor.close()

def load_data_with_priority(connection):
    """Load data with high priority to minimize autovacuum conflicts"""
    try:
        # Set aggressive timeouts to avoid long waits
        cursor = connection.cursor()
        cursor.execute("SET lock_timeout = '3s';")  # Very short timeout
        cursor.execute("SET statement_timeout = '2min';")  # Reasonable timeout
        connection.commit()
        cursor.close()
        
        # Load critical schema first with improved error handling
        schema_path = "/app/src/db/rebrickable_schema.sql"
        schema_loaded = False
        
        if os.path.exists(schema_path):
            logger.info("Creating database schema...")
            try:
                with open(schema_path, 'r') as schema_file:
                    schema_script = schema_file.read()
                
                # First, try to create the essential tables independently
                # Skip the full schema if it's problematic and create minimal tables
                essential_tables_created = create_essential_tables(connection)
                
                if essential_tables_created:
                    logger.info("Essential tables created successfully")
                    schema_loaded = True
                else:
                    logger.warning("Essential table creation failed, trying full schema...")
                    
                    # Split and clean statements - skip DROP statements to avoid conflicts
                    statements = [s.strip() for s in schema_script.split(';') 
                                 if s.strip() and not s.strip().startswith('--') 
                                 and not s.strip().upper().startswith('DROP')]
                    
                    statements_applied = 0
                    for statement in statements:
                        # Use individual connections for each statement to avoid transaction issues
                        success = execute_statement_safely(connection, statement)
                        if success:
                            statements_applied += 1
                    
                    if statements_applied > 0:
                        schema_loaded = True
                        logger.info(f"Schema setup completed ({statements_applied}/{len(statements)} statements applied)")
                    else:
                        logger.warning("Schema setup failed - will rely on fallback table creation")
                        
            except Exception as e:
                logger.warning(f"Schema file processing failed: {e} - will rely on fallback table creation")
        
        # Verify critical tables exist before proceeding
        critical_tables = ['part_categories', 'colors', 'themes']
        tables_ready = 0
        
        for table in critical_tables:
            # Use fresh cursor for each check to avoid transaction issues
            cursor = connection.cursor()
            try:
                cursor.execute(f"SELECT 1 FROM {table} LIMIT 1;")
                logger.debug(f"✅ Table {table} exists")
                tables_ready += 1
                cursor.close()
            except Exception:
                cursor.close()
                logger.warning(f"❌ Table {table} does not exist. Creating basic table...")
                
                # Use fresh cursor for table creation with retry
                max_retries = 2
                table_created = False
                
                for retry in range(max_retries + 1):
                    cursor = connection.cursor()
                    try:
                        if table == 'part_categories':
                            cursor.execute("""
                                CREATE TABLE IF NOT EXISTS part_categories (
                                    id INTEGER PRIMARY KEY,
                                    name VARCHAR(255) NOT NULL
                                );
                            """)
                        elif table == 'colors':
                            cursor.execute("""
                                CREATE TABLE IF NOT EXISTS colors (
                                    id INTEGER PRIMARY KEY,
                                    name VARCHAR(255) NOT NULL,
                                    rgb VARCHAR(6),
                                    is_trans BOOLEAN,
                                    num_parts BIGINT NOT NULL DEFAULT 0,
                                    num_sets BIGINT NOT NULL DEFAULT 0,
                                    y1 INTEGER NOT NULL DEFAULT 0,
                                    y2 INTEGER NOT NULL DEFAULT 0
                                );
                            """)
                        elif table == 'themes':
                            cursor.execute("""
                                CREATE TABLE IF NOT EXISTS themes (
                                    id INTEGER PRIMARY KEY,
                                    name VARCHAR(255) NOT NULL,
                                    parent_id INTEGER
                                );
                            """)
                        connection.commit()
                        logger.info(f"✅ Created fallback table {table}")
                        tables_ready += 1
                        table_created = True
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        error_msg = str(e).lower()
                        
                        # Always rollback and cleanup on error
                        try:
                            connection.rollback()
                        except Exception:
                            pass
                        try:
                            cursor.close()
                        except Exception:
                            pass
                        
                        # Check if we should retry
                        if retry < max_retries and any(x in error_msg for x in ['lock timeout', 'deadlock', 'could not obtain']):
                            logger.debug(f"Retrying table creation for {table} (attempt {retry + 2}/{max_retries + 1})")
                            time.sleep(0.5)
                            continue
                        
                        if retry == max_retries:
                            logger.error(f"❌ Failed to create fallback table {table} after {max_retries + 1} attempts: {e}")
                        break
                    finally:
                        try:
                            cursor.close()
                        except Exception:
                            pass
                
                if not table_created:
                    logger.warning(f"❌ Could not create table {table}, but continuing...")
        
        logger.info(f"✅ {tables_ready}/{len(critical_tables)} critical tables verified/created")
        
        # Disable constraints for faster loading
        cursor = connection.cursor()
        cursor.execute("SET session_replication_role = replica;")
        connection.commit()
        cursor.close()
        
        # Load tables in optimized order (smallest to largest)
        ordered_tables = [
            ("part_categories.csv", "part_categories"),
            ("colors.csv", "colors"),
            ("themes.csv", "themes"),
            ("minifigs.csv", "minifigs"),
            ("sets.csv", "sets"),
            ("part_relationships.csv", "part_relationships"),
            ("inventories.csv", "inventories"),
            ("inventory_minifigs.csv", "inventory_minifigs"),
            ("inventory_sets.csv", "inventory_sets"),
            ("parts.csv", "parts"),
            ("elements.csv", "elements"),
            ("inventory_parts.csv", "inventory_parts"),  # Largest last
        ]
        
        total_start = time.time()
        
        for csv_file, table_name in ordered_tables:
            if os.path.exists(os.path.join(DATA_DIR, csv_file)):
                logger.info(f"🔄 Loading {csv_file} into {table_name}")
                load_table_fast(connection, csv_file, table_name)
            else:
                logger.warning(f"File not found: {csv_file}")
        
        total_time = time.time() - total_start
        logger.info(f"✅ All data loaded in {total_time:.1f} seconds")
        
        # Re-enable constraints
        cursor = connection.cursor()
        cursor.execute("SET session_replication_role = DEFAULT;")
        connection.commit()
        cursor.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return False

def load_table_fast(connection, csv_file, table_name):
    """Fast table loading with minimal overhead"""
    start_time = time.time()
    
    try:
        # Read CSV with optimized settings
        df = pd.read_csv(
            os.path.join(DATA_DIR, csv_file), 
            na_values=['', 'NA', 'NULL'],
            dtype=str,  # Read as strings initially
            engine='c'  # Use C engine for speed
        )
        
        # Lowercase columns
        df.columns = [col.lower() for col in df.columns]
        
        # Basic data type optimization
        df = optimize_dataframe(df, table_name)
        
        if not df.empty:
            # Fast batch insert
            cursor = connection.cursor()
            columns = list(df.columns)
            
            # Convert to tuples efficiently  
            data_tuples = [tuple(None if pd.isna(val) else val for val in row) 
                          for row in df.itertuples(index=False, name=None)]
            
            # Single batch insert for smaller tables, chunked for large ones
            if len(data_tuples) <= 50000:
                # Single insert for smaller tables
                insert_query = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES %s ON CONFLICT DO NOTHING
                """
                execute_values(cursor, insert_query, data_tuples, template=None, page_size=2000)
            else:
                # Chunked insert for large tables
                insert_query = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES %s ON CONFLICT DO NOTHING
                """
                batch_size = 5000  # Larger batches for container
                for i in range(0, len(data_tuples), batch_size):
                    batch = data_tuples[i:i + batch_size]
                    execute_values(cursor, insert_query, batch, template=None, page_size=batch_size)
                    
                    # Progress for large tables
                    if i % (batch_size * 20) == 0:  # Every 100k rows
                        logger.info(f"   Progress: {min(i + batch_size, len(data_tuples)):,}/{len(data_tuples):,} rows")
            
            cursor.close()
            connection.commit()
            
            load_time = time.time() - start_time
            rate = len(df) / load_time if load_time > 0 else 0
            logger.info(f"✅ {table_name}: {len(df):,} rows in {load_time:.2f}s ({rate:,.0f} rows/sec)")
        
    except Exception as e:
        logger.error(f"Error loading {csv_file}: {e}")
        connection.rollback()
        raise

def optimize_dataframe(df, table_name):
    """Basic dataframe optimization for each table"""
    
    # Convert numeric columns
    numeric_columns = {
        'colors': ['num_parts', 'num_sets', 'y1', 'y2'],
        'themes': ['parent_id'],
        'sets': ['year', 'theme_id', 'num_parts'],
        'parts': ['part_cat_id'],
        'elements': ['color_id', 'design_id'],
        'inventory_parts': ['inventory_id', 'color_id', 'quantity'],
        'inventories': ['version'],
        'minifigs': ['num_parts'],
        'inventory_minifigs': ['inventory_id', 'quantity'],
        'inventory_sets': ['inventory_id', 'quantity'],
    }
    
    if table_name in numeric_columns:
        for col in numeric_columns[table_name]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Special handling for colors table y1, y2 - use 0 as default
                if table_name == 'colors' and col in ['y1', 'y2']:
                    df[col] = df[col].fillna(0)
    
    # Handle nulls
    df = df.where(pd.notnull(df), None)
    
    return df

def create_essential_tables(connection):
    """Create essential tables without foreign key constraints to avoid dependency issues"""
    essential_sql = [
        """
        CREATE TABLE IF NOT EXISTS part_categories (
            id INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS colors (
            id INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            rgb VARCHAR(6),
            is_trans BOOLEAN,
            num_parts BIGINT NOT NULL DEFAULT 0,
            num_sets BIGINT NOT NULL DEFAULT 0,
            y1 INTEGER NOT NULL DEFAULT 0,
            y2 INTEGER NOT NULL DEFAULT 0
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS themes (
            id INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            parent_id INTEGER
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS parts (
            part_num VARCHAR(20) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            part_cat_id INTEGER,
            part_material VARCHAR(50)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS sets (
            set_num VARCHAR(20) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            year INTEGER NOT NULL,
            theme_id INTEGER,
            num_parts INTEGER,
            img_url VARCHAR(255)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS minifigs (
            fig_num VARCHAR(20) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            num_parts INTEGER,
            img_url VARCHAR(255)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS inventories (
            id INTEGER PRIMARY KEY,
            version INTEGER,
            set_num VARCHAR(20)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS elements (
            element_id VARCHAR(20) PRIMARY KEY,
            part_num VARCHAR(20),
            color_id INTEGER,
            design_id INTEGER
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS inventory_parts (
            inventory_id INTEGER,
            part_num VARCHAR(20),
            color_id INTEGER,
            quantity INTEGER,
            is_spare BOOLEAN
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS inventory_sets (
            inventory_id INTEGER,
            set_num VARCHAR(20),
            quantity INTEGER
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS inventory_minifigs (
            inventory_id INTEGER,
            fig_num VARCHAR(20),
            quantity INTEGER
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS part_relationships (
            rel_type VARCHAR(1),
            child_part_num VARCHAR(20),
            parent_part_num VARCHAR(20)
        );
        """
    ]
    
    tables_created = 0
    for sql in essential_sql:
        if execute_statement_safely(connection, sql.strip()):
            tables_created += 1
    
    return tables_created >= len(essential_sql) // 2  # At least half the tables created

def execute_statement_safely(connection, statement):
    """Execute a SQL statement safely with proper error handling and retries"""
    if not statement or statement.isspace():
        return True
        
    max_retries = 3
    for retry in range(max_retries):
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(statement)
            connection.commit()
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Always rollback on error
            try:
                connection.rollback()
            except Exception:
                pass
            
            # Close cursor
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            
            # Check if we should retry
            if retry < max_retries - 1 and any(x in error_msg for x in ['lock timeout', 'deadlock', 'could not obtain']):
                logger.debug(f"Retrying statement (attempt {retry + 2}/{max_retries}) due to: {e}")
                time.sleep(0.5 * (retry + 1))  # Exponential backoff
                continue
            
            # Log error but don't fail for expected issues
            if any(x in error_msg for x in ['already exists', 'does not exist']):
                logger.debug(f"Statement skipped (expected): {e}")
                return True
            else:
                logger.debug(f"Statement failed: {e}")
                return False
    
    return False

def main():
    """Main execution with signal handling"""
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("🚀 Starting container-optimized data upload")
    
    try:
        # Connect with timeout
        connection = psycopg2.connect(**DB_PARAMS, connect_timeout=30)
        
        # Apply optimizations
        optimize_postgres_for_bulk_load(connection)
        
        # Load data with priority
        success = load_data_with_priority(connection)
        
        if success:
            logger.info("🎉 Data upload completed successfully!")
        else:
            logger.error("❌ Data upload failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        sys.exit(1)
    finally:
        try:
            connection.close()
        except:
            pass

if __name__ == "__main__":
    main()
