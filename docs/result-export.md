# Export experimental results

Export is optional. It creates a local `.perseval-result.tar.gz` archive containing
predictions, metrics, configuration, and provenance.

## Export from a notebook

After training:

```python
model.predict(trainer)
run = model.last_run
bundle = run.export_result(output_dir="exports/")
print(bundle)
```

Use `run.inspect()` to review the result before exporting. For multiple decoder or
LaMP variants, select the intended run from `model.last_runs` instead.

If `model.last_run` is unexpectedly `None`, check `model.last_run_error`.

## Download

The archive is saved in `exports/`. Copy it to the machine with the results
repository, or download it from Colab:

```python
from google.colab import files
files.download(str(bundle))
```

## Publish to the leaderboard

In an installed, configured, clean PersEval-experimental-results checkout:

```sh
perseval-results publish-bundle /path/to/result.perseval-result.tar.gz \
  --name "Experiment name" --submitter "Research group" --commit
git push
```

Confirm the publication plan when prompted. The command imports the result and
creates a local commit; pushing triggers the GitHub Pages leaderboard deployment.

## Optional artifacts

Include a file or directory explicitly on the first export:

```python
bundle = run.export_result(
    output_dir="exports/",
    artifacts={"diagnostics": "/path/to/file-or-directory"},
)
```

The first successful export freezes the run and artifact selection. Re-exporting
returns the same run ID and archive bytes. Export itself never uploads results.
