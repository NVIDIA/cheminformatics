CREATE TABLE IF NOT EXISTS smiles (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
   	model_name TEXT NULL,
   	smiles TEXT NULL,
	num_samples INTEGER DEFAULT 10,
	scaled_radius REAL,
    force_unique INTEGER,
    sanitize INTEGER,
	processed INTEGER DEFAULT 0,
	dataset_type TEXT NOT NULL,
	UNIQUE(model_name, smiles, num_samples, scaled_radius, force_unique, sanitize)
);

CREATE INDEX IF NOT EXISTS smiles_index ON smiles (smiles, num_samples, scaled_radius, force_unique, sanitize);

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

CREATE INDEX IF NOT EXISTS smiles_samples_index ON smiles_samples (input_id, smiles, scaffold, is_valid, is_generated);

CREATE TABLE IF NOT EXISTS meta_data (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
   	name TEXT NOT NULL,
   	value TEXT NOT NULL,
	date DATETIME NOT NULL,
	version TEXT NOT NULL
);
