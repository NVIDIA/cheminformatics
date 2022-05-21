CREATE TABLE IF NOT EXISTS smiles (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
   	smiles TEXT NULL,
   	model_name TEXT NULL,
	num_samples INTEGER DEFAULT 10,
	scaled_radius REAL,
	dataset_type TEXT NOT NULL,
    processed INTEGER DEFAULT 0,
	UNIQUE(smiles, model_name, scaled_radius)
);

-- Duplicate table for staging smiles
-- CREATE TABLE smiles_tmp LIKE smiles;

CREATE TABLE IF NOT EXISTS smiles_samples (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
   	input_id INTEGER NOT NULL,
   	smiles TEXT NOT NULL,
	embedding TEXT NOT NULL,
	embedding_dim TEXT NOT NULL,
	finger_print TEXT NOT NULL,
	is_valid INTEGER NOT NULL DEFAULT 1,
	is_generated INTEGER NOT NULL DEFAULT 1,
	scaffold TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS smiles_samples_index ON smiles_samples (input_id, smiles, is_valid, is_generated);

CREATE TABLE IF NOT EXISTS meta_data (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
   	name TEXT NOT NULL,
   	value TEXT NOT NULL,
	date DATETIME NOT NULL,
	version TEXT NOT NULL
);
