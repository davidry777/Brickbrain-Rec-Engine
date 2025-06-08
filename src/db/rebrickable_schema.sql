-- Drop tables if they exist (in reverse order to avoid foreign key constraints)
DROP TABLE IF EXISTS inventory_parts CASCADE;
DROP TABLE IF EXISTS inventory_sets CASCADE;
DROP TABLE IF EXISTS inventory_minifigs CASCADE;
DROP TABLE IF EXISTS inventories CASCADE;
DROP TABLE IF EXISTS elements CASCADE;
DROP TABLE IF EXISTS part_relationships CASCADE;
DROP TABLE IF EXISTS parts CASCADE;
DROP TABLE IF EXISTS part_categories CASCADE;
DROP TABLE IF EXISTS minifigs CASCADE;
DROP TABLE IF EXISTS sets CASCADE;
DROP TABLE IF EXISTS themes CASCADE;
DROP TABLE IF EXISTS colors CASCADE;

-- Create tables

-- Colors table
CREATE TABLE colors (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    rgb VARCHAR(6),
    is_trans BOOLEAN,
    num_parts BIGINT NOT NULL,
    num_sets BIGINT NOT NULL,
    y1 INTEGER NOT NULL,
    y2 INTEGER NOT NULL
);

-- Themes table
CREATE TABLE themes (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER,
    FOREIGN KEY (parent_id) REFERENCES themes(id)
);

-- Sets table
CREATE TABLE sets (
    set_num VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    year INTEGER NOT NULL,
    theme_id INTEGER,
    num_parts INTEGER,
    img_url VARCHAR(255),
    FOREIGN KEY (theme_id) REFERENCES themes(id)
);

-- Minifigs table
CREATE TABLE minifigs (
    fig_num VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    num_parts INTEGER,
    img_url VARCHAR(255)
);

-- Part categories table
CREATE TABLE part_categories (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

-- Parts table
CREATE TABLE parts (
    part_num VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    part_cat_id INTEGER,
    part_material VARCHAR(50),
    FOREIGN KEY (part_cat_id) REFERENCES part_categories(id)
);

-- Part relationships table
CREATE TABLE part_relationships (
    rel_type VARCHAR(1),
    child_part_num VARCHAR(20),
    parent_part_num VARCHAR(20),
    PRIMARY KEY (child_part_num, parent_part_num),
    FOREIGN KEY (child_part_num) REFERENCES parts(part_num),
    FOREIGN KEY (parent_part_num) REFERENCES parts(part_num)
);

-- Elements table
CREATE TABLE elements (
    element_id VARCHAR(20) PRIMARY KEY,
    part_num VARCHAR(20),
    color_id INTEGER,
    design_id INTEGER,
    FOREIGN KEY (part_num) REFERENCES parts(part_num),
    FOREIGN KEY (color_id) REFERENCES colors(id)
);

-- Inventories table
CREATE TABLE inventories (
    id INTEGER PRIMARY KEY,
    version INTEGER,
    set_num VARCHAR(20),
    FOREIGN KEY (set_num) REFERENCES sets(set_num)
);

-- Inventory minifigs table
CREATE TABLE inventory_minifigs (
    inventory_id INTEGER,
    fig_num VARCHAR(20),
    quantity INTEGER,
    PRIMARY KEY (inventory_id, fig_num),
    FOREIGN KEY (inventory_id) REFERENCES inventories(id),
    FOREIGN KEY (fig_num) REFERENCES minifigs(fig_num)
);

-- Inventory sets table
CREATE TABLE inventory_sets (
    inventory_id INTEGER,
    set_num VARCHAR(20),
    quantity INTEGER,
    PRIMARY KEY (inventory_id, set_num),
    FOREIGN KEY (inventory_id) REFERENCES inventories(id),
    FOREIGN KEY (set_num) REFERENCES sets(set_num)
);

-- Inventory parts table
CREATE TABLE inventory_parts (
    inventory_id INTEGER,
    part_num VARCHAR(20),
    color_id INTEGER,
    quantity INTEGER,
    is_spare BOOLEAN,
    img_url VARCHAR(255),
    FOREIGN KEY (inventory_id) REFERENCES inventories(id),
    FOREIGN KEY (part_num) REFERENCES parts(part_num),
    FOREIGN KEY (color_id) REFERENCES colors(id),
    PRIMARY KEY (inventory_id, part_num, color_id, is_spare)
);

-- Create indices for better query performance
CREATE INDEX idx_themes_parent_id ON themes(parent_id);
CREATE INDEX idx_sets_theme_id ON sets(theme_id);
CREATE INDEX idx_parts_part_cat_id ON parts(part_cat_id);
CREATE INDEX idx_inventories_set_num ON inventories(set_num);
CREATE INDEX idx_inventory_minifigs_inventory_id ON inventory_minifigs(inventory_id);
CREATE INDEX idx_inventory_sets_inventory_id ON inventory_sets(inventory_id);
CREATE INDEX idx_inventory_parts_inventory_id ON inventory_parts(inventory_id);
CREATE INDEX idx_elements_part_num ON elements(part_num);
CREATE INDEX idx_elements_color_id ON elements(color_id);