# Data access

Every dataset the paper relies on, whether it ships in this archive or has to be obtained
elsewhere. Nothing the paper measures is withheld.

| Dataset | Used for | In this archive | How to obtain |
|---|---|---|---|
| **150-prompt evaluation set** (50 Consulting, 50 Medical, 50 Payments; 43 ultra-vague, 62 medium, 45 detailed) | Every coverage number in the paper | ✅ `data/prompts_P01-P50.json` **and** `data/prompts_N01-N100.csv` | included — the set is split across two files, `P01`–`P50` (the original 50, on which the `n=50` rows are computed) and `N01`–`N100` (the extension to 150); the `prompt_id` field of every run record refers to these ids |
| **Three domain parameter spaces** with importance weights | The coverage metric and the stopping thresholds | ✅ `data/taxonomies.json` | included |
| **516 human annotations** over 36 prompts, three labels (`RESOLVED` / `MENTIONED` / `ABSENT`) | Scorer validation, Cohen's kappa = 0.749 | ✅ `data/annotations/human_annotations_516.csv` | included |
| **Second-judge labels** (Claude Sonnet 5, same 516 items) | The cross-family judge comparison | ✅ `data/annotations/claude_judge_labels.json` | included |
| **Per-run records**, ALI (3 configurations, 5 ablation variants, 2 threshold sweeps) | Every ALI row and interval | ✅ `results/ali/` | included — resolved frame and per-turn coverage trajectory for each of the 150 runs |
| **Per-run records**, baselines (6 models, 500 multi-turn runs + 150 single-pass) | Every baseline row and interval | ✅ `results/baselines/` | included — full transcripts |
| **MIMIC-CXR** radiology reports | Training the Rad-Assist extraction model; the 50-report held-out evaluation | ❌ **cannot be redistributed** | Free, but credentialed: complete the CITI training and sign the PhysioNet Credentialed Health Data Use Agreement, then download from PhysioNet. The evaluation script (`code/ali/radassist_extraction_eval.py`) draws its 50 reports with `pandas.DataFrame.sample(n=50, random_state=42)` from a cleaned holdout table, so the same draw is reproducible once that table is rebuilt from the corpus |

## On the two restrictions

**MIMIC-CXR.** The PhysioNet Credentialed Health Data License forbids redistribution, so the
reports themselves cannot travel in this archive and neither can any file quoting them
verbatim. This is a licence constraint on de-identified clinical data, not a choice: any
credentialed reader can obtain the identical corpus, and the selection rule is stated above so
that the same 50-report draw is reconstructible. The evaluation script is included.

**The three deployments' production code.** Out of scope rather than withheld: the paper
measures the ALI pipeline, and the pipeline is here in full. Nothing in `results/` depends on
code absent from `code/`.

## Provenance of the generated training data

The Telos and FinAgent components train on programmatically generated conversations, not on
collected user data — the paper's Limitations section says so and treats it as a threat to
validity. No file in this archive presents generated data as collected data. The only real
corpus used anywhere in this work is MIMIC-CXR, in the row above.

## After acceptance

The same material, plus the fine-tuned adapters omitted here for size, will be published in a
permanent public repository under a licence permitting free use for research purposes.
