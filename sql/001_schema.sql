-- Car-part retrieval: core schema. See docs/DATASET_SPEC.md

CREATE TABLE part (
    part_id           TEXT PRIMARY KEY,          -- generated: {brand}-{part_number_norm}
    part_number_raw   TEXT NOT NULL,             -- EXACTLY as printed. Never normalised by hand.
    part_number_norm  TEXT NOT NULL,             -- computed once, in code
    brand             TEXT NOT NULL,
    category          TEXT NOT NULL,
    name              TEXT NOT NULL,
    description       TEXT,
    oem_number        TEXT,                      -- bridge for grade-2 relevance
    source_url        TEXT NOT NULL,
    source_site       TEXT NOT NULL,
    date_collected    DATE NOT NULL,
    collected_by      TEXT NOT NULL,
    licence_note      TEXT,
    is_distractor     BOOLEAN NOT NULL DEFAULT FALSE,  -- never a correct answer
    verified          BOOLEAN NOT NULL DEFAULT FALSE,  -- YOU set this. Unverified never indexed.
    notes             TEXT
);
CREATE INDEX ON part (part_number_norm);
CREATE INDEX ON part (category);
CREATE INDEX ON part (oem_number) WHERE oem_number IS NOT NULL;

CREATE TABLE part_image (
    image_id           TEXT PRIMARY KEY,
    part_id            TEXT REFERENCES part(part_id),
    file_path          TEXT NOT NULL UNIQUE,
    role               TEXT NOT NULL CHECK (role IN ('index','query')),
    image_source       TEXT NOT NULL CHECK (image_source IN ('catalog','own_photo','roboflow')),
    phash              TEXT NOT NULL,            -- dedup + leakage guard
    width              INT  NOT NULL,
    height             INT  NOT NULL,
    capture_session_id TEXT,                     -- query only; session-level split guard
    item_id            TEXT,                     -- query only; links A/B/C shots. PAIRED TESTS NEED THIS.
    shot_type          TEXT CHECK (shot_type IN ('A','B','C')),
    is_primary         BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX ON part_image (phash);
CREATE INDEX ON part_image (capture_session_id);
CREATE INDEX ON part_image (item_id);

CREATE TABLE query_gt (
    image_id          TEXT PRIMARY KEY REFERENCES part_image(image_id),
    true_part_id      TEXT REFERENCES part(part_id),   -- YOU set this, from photo A
    true_category     TEXT NOT NULL,
    id_source         TEXT NOT NULL CHECK (id_source IN
                        ('box','stamped_on_part','shop_told_me','unknown')),
    has_visible_partno BOOLEAN,                  -- RQ2 IS THE MEAN OF THIS COLUMN
    gt_text           TEXT,                      -- verbatim; OCR ground truth
    strata            TEXT[],
    labelled_by       TEXT,
    label_batch       TEXT
);

CREATE TABLE compatibility (          -- DISPLAY ONLY. Never ranked on. Never evaluated.
    part_id     TEXT REFERENCES part(part_id),
    make        TEXT, model TEXT, year_from INT, year_to INT,
    confidence  TEXT, source TEXT
);

CREATE TABLE split (
    image_id TEXT PRIMARY KEY REFERENCES part_image(image_id),
    split    TEXT NOT NULL CHECK (split IN ('valid','test')),
    seed     INT  NOT NULL
);
