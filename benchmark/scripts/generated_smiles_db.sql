CREATE TABLE IF NOT EXISTS smiles (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
   	smiles TEXT NULL,
	num_samples INTEGER DEFAULT 10,
	scaled_radius REAL,
	dataset_type TEXT NOT NULL,
    processed INTEGER DEFAULT 0,
	UNIQUE(smiles, scaled_radius)
);

CREATE TABLE IF NOT EXISTS smiles_samples (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
   	input_id INTEGER NOT NULL,
   	smiles TEXT NOT NULL,
	scaffold TEXT NOT NULL,
	embedding TEXT NOT NULL,
	embedding_dim TEXT NOT NULL,
	finger_print TEXT NOT NULL,
	is_valid INTEGER NOT NULL DEFAULT 1,
	is_generated INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS smiles_samples_index ON smiles_samples (input_id, smiles, scaffold, is_valid, is_generated);


CREATE TABLE IF NOT EXISTS meta_data (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
   	name TEXT NOT NULL,
   	value TEXT NOT NULL,
	date DATETIME NOT NULL,
	version TEXT NOT NULL
);
