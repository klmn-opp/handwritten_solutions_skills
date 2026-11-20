# Workspace records

Every run uses a new `workspace/runs/<run_id>/` directory. Runs keep input
solutions, rendered pages, comparisons, reports and logs; an actual training or
preparation run need not manufacture a PDF. Successful Job output is under its
`rendered/` directory. Failed attempts retain their reason and original input.

```sh
hwsol workspace --help
hwsol workspace latest
hwsol verify --job workspace/runs/<run_id> --pixels
```

`HWSOL_WORKSPACE` selects a workspace. `HWSOL_DATA_DIR` selects a private Profile
store. Without explicit settings, the application uses the platform's user data
location. `MPLCONFIGDIR` can select a writable local font-cache directory.

Only a verified production delivery should advance latest. Experiments and held
candidates must stay separate. Never manually edit latest, overwrite a finalized
run, alter raw sources, or relabel an exposed test as independent holdout.
An archived manifest contains resolved paths and hashes; replay it into a new
run without changing factors while claiming exact reproduction.

Keep scans, assignments, Profile exports, checkpoints and feedback private.
This public package intentionally contains no actual personal run directory or
local operational handoff. Its source commit and package hashes identify the
distributed implementation, not permission to share a user's assets.
