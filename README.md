# pi0.5 BEHAVIOR-1K Eval Results

Training progress and per-task eval results for the pi0.5 BEHAVIOR-1K fine-tunes.
Published as a static site via GitHub Pages: see the repo's Pages URL.

- `training_progress_*.png` — loss/grad_norm curves, regenerated from the training log
  by `plot_training_progress.py` (run from the `openpi` repo, has access to the live log).
- `tasks/<task_name>/` — one page per task: summary stats, per-instance table, and
  eval rollout videos (public_test split, from `omnigibson.eval.eval` + `serve_b1k.py`).
