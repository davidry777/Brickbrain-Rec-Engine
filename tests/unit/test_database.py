#!/usr/bin/env python3
"""
Quick test script to check database connectivity and basic queries
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os

# Database configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "brickbrain"),
    "user": os.getenv("DB_USER", "brickbrain"),
    "password": os.getenv("DB_PASSWORD", "brickbrain_password")
}

def test_database():
    """Test basic database operations"""
    try:
        print("🔌 Testing database connection...")
        conn = psycopg2.connect(**DATABASE_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Test basic connectivity
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        print(f"✅ Database connection successful: {result}")
        
        # Test sets table
        print("\n📦 Testing sets table...")
        cursor.execute("SELECT COUNT(*) as count FROM sets")
        count = cursor.fetchone()
        print(f"✅ Sets table has {count['count']} records")
        
        # Test themes table
        print("\n🎨 Testing themes table...")
        cursor.execute("SELECT COUNT(*) as count FROM themes")
        count = cursor.fetchone()
        print(f"✅ Themes table has {count['count']} records")
        
        # Test Star Wars search
        print("\n⭐ Testing Star Wars search...")
        query = """
        SELECT s.set_num, s.name, s.year, t.name as theme_name, s.num_parts
        FROM sets s
        LEFT JOIN themes t ON s.theme_id = t.id
        WHERE s.num_parts > 0 
        AND (s.name ILIKE %s OR t.name ILIKE %s)
        AND s.theme_id NOT IN (501, 503, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 777)
        LIMIT 5
        """
        
        cursor.execute(query, ('%star wars%', '%star wars%'))
        results = cursor.fetchall()
        
        print(f"✅ Found {len(results)} Star Wars building sets:")
        for row in results:
            print(f"   - {row['name']} ({row['set_num']}) - {row['num_parts']} pieces - Theme: {row['theme_name']}")
        
        # Test Star Wars themes
        print("\n🌟 Testing Star Wars themes...")
        cursor.execute("SELECT id, name, parent_id FROM themes WHERE name ILIKE %s", ('%star wars%',))
        themes = cursor.fetchall()
        
        print(f"✅ Found {len(themes)} Star Wars themes:")
        for theme in themes:
            print(f"   - {theme['name']} (ID: {theme['id']}, Parent: {theme['parent_id']})")
            
            # Count sets in each theme
            cursor.execute("SELECT COUNT(*) as count FROM sets WHERE theme_id = %s AND num_parts > 0", (theme['id'],))
            theme_count = cursor.fetchone()
            print(f"     └─ {theme_count['count']} building sets")
        
        conn.close()
        print("\n✅ All database tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_search_query():
    """Test the exact search query used by the API"""
    try:
        print("\n🔍 Testing API search query...")
        conn = psycopg2.connect(**DATABASE_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # This is the exact query structure from the API
        user_id = 0  # Use 0 for testing
        query = """
        SELECT s.set_num, s.name, s.year, t.name as theme_name, s.num_parts, s.img_url,
               COALESCE(AVG(ui.rating), 0) as avg_rating,
               COUNT(ui.rating) as total_ratings,
               EXISTS(
                   SELECT 1 FROM user_collections uc 
                   WHERE uc.user_id = %s AND uc.set_num = s.set_num
               ) as is_in_user_collection,
               EXISTS(
                   SELECT 1 FROM user_wishlists uw 
                   WHERE uw.user_id = %s AND uw.set_num = s.set_num
               ) as is_in_user_wishlist
        FROM sets s
        LEFT JOIN themes t ON s.theme_id = t.id
        LEFT JOIN user_interactions ui ON s.set_num = ui.set_num AND ui.interaction_type = 'rating'
        WHERE s.num_parts > 0 
        AND s.theme_id NOT IN %s
        AND (s.name ILIKE %s OR t.name ILIKE %s)
        GROUP BY s.set_num, s.name, s.year, t.name, s.num_parts, s.img_url
        ORDER BY s.name ASC 
        LIMIT %s OFFSET %s
        """
        
        accessory_themes = (501, 503, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 777)
        params = [user_id, user_id, accessory_themes, '%star wars%', '%star wars%', 5, 0]
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        print(f"✅ API search query returned {len(results)} results:")
        for row in results:
            print(f"   - {row['name']} ({row['set_num']}) - {row['num_parts']} pieces")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ API search query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧱 LEGO Database Test Script")
    print("=" * 40)
    
    success = test_database()
    if success:
        test_search_query()
    
    print("\n🏁 Test complete!")
