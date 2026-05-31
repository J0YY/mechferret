## Summary

- Describe the user-facing or internal change.

## Validation

- [ ] `make check`
- [ ] `make support`
- [ ] `python3 -m unittest discover -s tests -q`
- [ ] `python3 -m compileall -q mechferret tests`
- [ ] `python3 -m mechferret doctor --strict`
- [ ] `python3 -m mechferret quickstart --mode ci --json`
- [ ] `python3 -m mechferret selftest --json`
- [ ] `git diff --check`

## Release Impact

- [ ] CLI JSON contracts are preserved or documented.
- [ ] Runtime assets are packageable when seed corpus, skills, or templates change.
- [ ] README/docs are updated for user-facing behavior changes.
- [ ] `runs/selftest/selftest.json` is attached or summarized when the environment matters.
- [ ] New artifacts include verification, audit, or smoke coverage where practical.
