# MechFerret Examples

## `version`

- `mechferret version --json`

## `commands`

- `mechferret commands --search "max gpu"`
- `mechferret commands --workflow first_run`
- `mechferret commands run`
- `mechferret commands --examples --markdown --out docs/CLI_EXAMPLES.md`
- `mechferret commands --group research --markdown`

## `completion`

- `mechferret completion zsh --json`

## `run`

- `mechferret run "What should I investigate?" --seed-corpus --out runs/custom`
- `mechferret run "What should I investigate?" --seed-corpus --json`

## `demo`

- `mechferret demo --out runs/demo`
- `mechferret demo --json`

## `login`

- `mechferret login openai --api-key "$OPENAI_API_KEY" --json`

## `api`

- `mechferret api --show --json`

## `goal`

- `mechferret goal "Make this investigation publishable" --seed-corpus --max-iterations 1 --max-rounds 1 --json`

## `discover`

- `mechferret discover --skill ioi-circuit --backend synthetic --json`

## `skills`

- `mechferret skills --json`
- `mechferret skills ioi-circuit`

## `modal`

- `mechferret modal status --json`

## `cluster`

- `mechferret cluster run --skill ioi-circuit --dry-run --json`

## `init`

- `mechferret init`
- `mechferret init --json`

## `status`

- `mechferret status --select best`
- `mechferret status --json`

## `runs`

- `mechferret runs --select best`
- `mechferret runs --json`

## `repl`

- `mechferret repl`

## `open`

- `mechferret open report --select best`
- `mechferret open all`

## `quickstart`

- `mechferret quickstart`
- `mechferret quickstart --run`
- `mechferret quickstart --mode ci --run`

## `selftest`

- `mechferret selftest --json`
- `mechferret selftest --report runs/selftest/selftest.json`
- `mechferret selftest --run --out runs/selftest`

## `support`

- `mechferret support`
- `mechferret support --json`

## `doctor`

- `mechferret doctor --strict`
- `mechferret doctor --json`

## `registry`

- `mechferret registry --kind tool --json`

## `memory`

- `mechferret memory --recent 5`
- `mechferret memory --json`

## `tool-results`

- `mechferret tool-results`
- `mechferret tool-results --clean --json`

## `cost`

- `mechferret cost --select best --json`

## `resume`

- `mechferret resume --select best`

## `inspect`

- `mechferret inspect --select best --json`

## `audit`

- `mechferret audit --select best --strict`
- `mechferret audit --json`

## `verify`

- `mechferret verify --select best --strict`
- `mechferret verify --repair --json`

## `paper`

- `mechferret paper --select best --json`
- `mechferret paper --select best --compile`

## `review-paper`

- `mechferret review-paper --select best --json`

## `bundle`

- `mechferret bundle --select best`
- `mechferret bundle --select best --json`

## `verify-bundle`

- `mechferret verify-bundle --select best --strict`

## `sae`

- `mechferret sae openvla status`
- `mechferret sae openvla plan --json`
