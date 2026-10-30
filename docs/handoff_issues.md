# Issues in Yasmina's outputs (not modified by Kareem)

1. `docs/edge_files.md` lists 16 columns, but the sampled Parquet files also have `body_clean`, `parent_body_clean`, `distinguished`, and the hive partition `week`. The doc does not say which text column is cleaned (`*_clean`) vs original (`body`, `parent_body`, raw markdown with escaped entities).
2. The 1%, 5% and 20% samples are independent draws, not nested (570 of 10,356 1% threads appear in the 5% sample). Intended? Nested samples would let the extraction cost be shared across rates.
3. Label counts: the brief says 107,705 labelled users; `data/processed/labels/venue=shared/set=main.parquet` has 100,108 (35,244 dem + 64,864 rep). The 107,705 in `results/label_versions/summary.md` includes 7,597 neither-side users.
4. `weekly_composition.csv` does not exist; composition covariates will be computed from the edge files.
5. `config/prep_jobs.yaml` has uncommitted local edits (disk and RSS limits). Not touched.
