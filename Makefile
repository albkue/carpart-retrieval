.PHONY: audit validate batch repro toy test
audit:     ; python scripts/data/audit_dataset.py data/raw/roboflow
validate:  ; python scripts/validate_catalog.py data/raw/catalog_images/batch_*.csv --master data/processed/catalog_master.csv
batch:     ; python scripts/make_batch.py data/processed/query --name $(NAME) --size $(or $(SIZE),100) --anonymise
repro:     ; python scripts/run_experiment.py --config configs/$(EXP).yaml $(if $(SPLIT),--split $(SPLIT))
toy:       ; python scripts/make_toy_split.py && $(MAKE) repro EXP=$(or $(EXP),baseline_mvb) SPLIT=data/processed/splits_toy.json
test:      ; pytest -q
