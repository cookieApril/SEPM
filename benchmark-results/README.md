# Generated benchmark artifacts

This directory is the default destination for experiment plans, checkpoints,
SQLite state, logs, raw model traces, and generated result tables. Those files
can be large, environment-specific, or contain model transcripts, so they are
excluded from version control.

The only checked-in artifact below this directory is
`smoke/gmemory-snapshots/multiagentbench/`. It is a small, frozen development
snapshot used by the GMemory integration tests. Its `manifest.json` records the
upstream repository, commit, partition rule, and content hash.

Running the evaluation commands recreates the required output directories.
Do not force-add generated traces or databases to Git. Publish archival results
through a dedicated artifact repository when they are needed for review.
