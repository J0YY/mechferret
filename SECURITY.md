# Security Policy

MechFerret is designed to run useful local research workflows without required
network services, API keys, GPUs, Modal, or cluster access. Security reports
should focus on issues that could expose secrets, corrupt artifacts, bypass
verification, or cause unsafe file access.

## Supported Versions

Security fixes target the current `main` branch until versioned releases exist.
If you are using an older checkout, please test against `main` before opening a
report when practical.

## Reporting a Vulnerability

Please do not post secrets, private run artifacts, API keys, provider logs, or
private dataset contents in a public issue. Open a GitHub issue with a minimal
description and indicate that details should be shared privately, or contact the
maintainers through the repository owner profile.

Include:

- A short summary of the risk.
- The affected command, API, template, or artifact type.
- Reproduction steps using synthetic or redacted inputs.
- The expected impact: secret exposure, path traversal, artifact tampering,
  verification bypass, or denial of service.
- Any temporary mitigation you have already tested.

## Handling Secrets

Do not commit provider credentials, cluster host details, private manifests,
private activation caches, or run artifacts containing sensitive source data.
Use `mechferret login` for provider keys and keep `.mechferret/` and generated
`runs/` directories out of public reports unless they are redacted.

## Verification-Sensitive Areas

Please pay particular attention to changes touching:

- `mechferret audit`, `verify`, `bundle`, and `verify-bundle`
- manifest hashing and artifact path resolution
- packaged templates and seed corpus loading
- provider configuration and CLI JSON output
- filesystem operations that read, write, open, or package paths
