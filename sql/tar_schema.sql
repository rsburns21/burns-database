-- Technology-Assisted Review (TAR) scaffolding

CREATE TABLE IF NOT EXISTS review_labels (
  exhibit_id     text NOT NULL,
  label          boolean NOT NULL,      -- true = Relevant, false = Not
  user_id        text NOT NULL,
  method         text NOT NULL,         -- 'seed', 'review', 'qc', 'elusion'
  notes          text,
  created_at     timestamptz DEFAULT now(),
  PRIMARY KEY (exhibit_id, user_id, method, created_at)
);

CREATE TABLE IF NOT EXISTS tar_models (
  model_id       bigserial PRIMARY KEY,
  created_at     timestamptz DEFAULT now(),
  embedding_model text NOT NULL,
  algo           text NOT NULL,
  hyperparams    jsonb NOT NULL,
  train_size     int NOT NULL,
  val_size       int NOT NULL,
  metrics        jsonb NOT NULL,
  seed_hash      text NOT NULL,
  code_hash      text NOT NULL
);

CREATE TABLE IF NOT EXISTS tar_predictions (
  model_id       bigint REFERENCES tar_models(model_id),
  exhibit_id     text NOT NULL,
  score          double precision NOT NULL,
  iteration      int NOT NULL,
  created_at     timestamptz DEFAULT now(),
  PRIMARY KEY (model_id, exhibit_id, iteration)
);

CREATE TABLE IF NOT EXISTS tar_audit (
  at             timestamptz DEFAULT now(),
  actor          text NOT NULL,
  event          text NOT NULL,
  details        jsonb NOT NULL
);

