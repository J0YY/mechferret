# Support

MechFerret is an offline-first research tool. The fastest support path is to
include the command you ran, the JSON or artifact path it produced, and the
smallest synthetic input that reproduces the issue.

## Before Opening an Issue

Run:

```bash
make support
python3 -m mechferret support
python3 -m mechferret doctor --strict
python3 -m mechferret selftest --report runs/selftest/selftest.json
python3 -m mechferret status --json
python3 -m mechferret open all --json
python3 -m mechferret commands --workflow support_report
```

The support report records provider credentials only as configured or missing;
credential values and environment values are omitted from the JSON artifact.

If the issue involves a generated dossier or bundle, also run the relevant
verification command:

```bash
python3 -m mechferret audit --json
python3 -m mechferret verify --json
python3 -m mechferret verify-bundle --select best --strict
```

## Where to Ask

- Bugs: use the bug report issue template and include reproduction steps.
- Feature ideas: use the feature request template and describe the offline path.
- Pull requests: follow `CONTRIBUTING.md` and the pull request checklist.
- Security concerns: follow `SECURITY.md`; do not post secrets or private run
  artifacts in public issues.

## What to Include

- Python version and operating system.
- Install method: editable checkout, wheel, pipx, or direct `python3 -m` run.
- Command provider/backend and optional extras, if any.
- `runs/selftest/selftest.json`, when available.
- Relevant `run.json`, `manifest.json`, `audit.json`, or bundle path.
- Redacted stdout/stderr or JSON payload.
