# Contributing

MechFerret is an offline-first research tool. Contributions should preserve the
core promise: a fresh checkout can run, test, audit, and package useful research
artifacts without API keys or GPUs.

## Local Loop

Use the standard library test runner unless a change explicitly needs an
optional integration:

```bash
make check
```

The expanded local loop is:

```bash
make quickstart
make docs
make workflows
make selftest
make support
python3 -m unittest discover -s tests -q
python3 -m compileall -q mechferret tests
python3 -m mechferret doctor --strict
python3 -m mechferret quickstart --mode ci --json
python3 -m mechferret selftest --json
python3 -m mechferret support --report /tmp/mechferret-support.json --json
git diff --check
```

For packaging or template changes, also run:

```bash
make wheel
python3 -m pip wheel . -w /tmp/mechferret-wheels --no-deps
python3 -m mechferret sae openvla init --project-root /tmp/mechferret-openvla
```

## Expectations

- Keep CLI JSON stable and script-friendly. New JSON commands should expose
  `ok`, concrete paths, and recovery-oriented `next_actions`.
- Keep docs generated from the parser fresh: `docs/CLI.md` and
  `docs/CLI_EXAMPLES.md` are refreshed with `make docs` and checked by tests.
- Keep runtime assets packageable. Seed corpus files, skills, and templates
  must be declared in `pyproject.toml` and covered by wheel smoke tests.
- Prefer small, auditable changes with regression coverage over broad rewrites.
- Preserve offline behavior. Optional providers, Modal, cluster execution, and
  real model backends must remain optional.

## Pull Request Checklist

- Tests pass locally.
- `python3 -m mechferret doctor --strict` passes.
- New commands are documented by the parser and include examples.
- New artifacts have verification or audit coverage where practical.
- README or docs are updated when user-facing behavior changes.
