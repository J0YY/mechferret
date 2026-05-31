# MechFerret Commands

## Workflows

### First Run

Create project notes, build a local dossier, inspect status, and write a support report.

- `mechferret init`
- `mechferret quickstart --run`
- `mechferret status`
- `mechferret next`
- `mechferret support`

### Publish Dossier

Generate, audit, verify, and package the selected research dossier.

- `mechferret paper --select best --provider local`
- `mechferret audit --select best --strict`
- `mechferret verify --select best --strict`
- `mechferret bundle --select best`
- `mechferret verify-bundle --select best --strict`

### Support Report

Collect redacted diagnostics and artifact state for issues or pull requests.

- `mechferret support --json`
- `mechferret doctor --strict`
- `mechferret status --json`
- `mechferret next --json`
- `mechferret open all --json`

### OpenVLA SAE

Create and inspect the OpenVLA sparse-autoencoder project scaffold.

- `mechferret quickstart --mode openvla --run`
- `mechferret sae openvla init`
- `mechferret sae openvla status`
- `mechferret sae openvla commands`

## Start

### `version`

Aliases: `/version`, `about`, `/about`

Print package and runtime version information.

Usage:

```text
usage: mechferret version [-h] [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret version --json`

### `commands`

Aliases: `/commands`, `help`, `/help`

List the installed CLI command surface.

Usage:

```text
usage: mechferret commands [-h]
                           [--group {start,research,artifacts,config,compute}]
                           [--search SEARCH] [--workflow [WORKFLOW]]
                           [--examples] [--markdown] [--out OUT] [--json]
                           [name]
```

Positionals:

- `name`: Show details for one command or alias.

Options:

- `-h`, `--help`: show this help message and exit
- `--group`: choices: `start`, `research`, `artifacts`, `config`, `compute`; Limit output to a workflow group.
- `--search`: Filter commands by name, alias, help text, option, positional, or choice.
- `--workflow`: choices: `all`, `first_run`, `publish_dossier`, `support_report`, `openvla_sae`; Show one workflow recipe, or all recipes when omitted.
- `--examples`: Print runnable examples instead of command summaries.
- `--markdown`: Render the selected command reference as Markdown.
- `--out`: Write rendered command output to a file instead of printing it.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret commands --search "max gpu"`
- `mechferret commands --workflow first_run`
- `mechferret commands run`
- `mechferret commands --examples --markdown --out docs/CLI_EXAMPLES.md`
- `mechferret commands --group research --markdown`

### `completion`

Aliases: `/completion`

Print shell completion script for mechferret.

Usage:

```text
usage: mechferret completion [-h] [--command EXECUTABLE] [--json]
                             {bash,zsh,fish}
```

Positionals:

- `shell`: choices: `bash`, `zsh`, `fish`; Shell to generate completions for.

Options:

- `-h`, `--help`: show this help message and exit
- `--command`: Executable name to complete.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret completion zsh --json`

### `init`

Aliases: `/init`

Create MECHFERRET.md project notes for the interactive agent.

Usage:

```text
usage: mechferret init [-h] [--project-root PROJECT_ROOT] [--force] [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--project-root`: Directory where MECHFERRET.md should be written.
- `--force`: Overwrite an existing MECHFERRET.md.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret init`
- `mechferret init --json`

### `status`

Aliases: `/status`

Summarize project setup, selected run, audit state, artifacts, and next actions.

Usage:

```text
usage: mechferret status [-h] [--runs-root RUNS_ROOT] [--db DB]
                         [--notes-root NOTES_ROOT]
                         [--project-root PROJECT_ROOT]
                         [--select {latest,best,ready}] [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search for run artifacts.
- `--db`: SQLite memory path.
- `--notes-root`: Directory containing MECHFERRET.md.
- `--project-root`: OpenVLA project root.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy for run-bound status.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret status --select best`
- `mechferret status --json`

### `next`

Aliases: `/next`

Print the next recommended project actions.

Usage:

```text
usage: mechferret next [-h] [--runs-root RUNS_ROOT] [--db DB]
                       [--notes-root NOTES_ROOT] [--project-root PROJECT_ROOT]
                       [--select {latest,best,ready}] [--limit LIMIT] [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search for run artifacts.
- `--db`: SQLite memory path.
- `--notes-root`: Directory containing MECHFERRET.md.
- `--project-root`: OpenVLA project root.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy for run-bound status.
- `--limit`: Maximum actions to print.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret next`
- `mechferret next --json`
- `mechferret next --select best --limit 3`

### `quickstart`

Aliases: `/quickstart`

Print the recommended first commands for demo, OpenVLA, and CI gates.

Usage:

```text
usage: mechferret quickstart [-h] [--mode {all,demo,openvla,ci}] [--run]
                             [--out OUT] [--db DB]
                             [--project-root PROJECT_ROOT] [--force] [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--mode`: choices: `all`, `demo`, `openvla`, `ci`; Quickstart path to print or run.
- `--run`: Execute a quickstart path. Defaults to the local demo; use --mode openvla or --mode ci for those paths.
- `--out`: Output directory for --run --mode demo.
- `--db`: Memory database for --run --mode demo.
- `--project-root`: Project root for --run --mode openvla.
- `--force`: Overwrite existing files for --run --mode openvla.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret quickstart`
- `mechferret quickstart --run`
- `mechferret quickstart --mode ci --run`

### `selftest`

Aliases: `/selftest`

Run offline readiness checks and optionally execute the local demo artifact path.

Usage:

```text
usage: mechferret selftest [-h] [--run] [--out OUT] [--db DB]
                           [--runs-root RUNS_ROOT] [--notes-root NOTES_ROOT]
                           [--project-root PROJECT_ROOT]
                           [--select {latest,best,ready}] [--report REPORT]
                           [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--run`: Execute the local demo quickstart and verify its artifacts.
- `--out`: Output directory for --run artifacts.
- `--db`: Memory database for --run artifacts.
- `--runs-root`: Root to inspect for existing run artifacts.
- `--notes-root`: Directory containing MECHFERRET.md.
- `--project-root`: OpenVLA project root to inspect.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy for status inspection.
- `--report`: Write the self-test JSON payload to this path for sharing or issue reports.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret selftest --json`
- `mechferret selftest --report runs/selftest/selftest.json`
- `mechferret selftest --run --out runs/selftest`

### `support`

Aliases: `/support`, `diagnostics`, `/diagnostics`

Write a shareable self-test report for issues or PRs.

Usage:

```text
usage: mechferret support [-h] [--run] [--out OUT] [--db DB]
                          [--runs-root RUNS_ROOT] [--notes-root NOTES_ROOT]
                          [--project-root PROJECT_ROOT]
                          [--select {latest,best,ready}] [--report REPORT]
                          [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--run`: Execute the local demo quickstart before writing the support report.
- `--out`: Output directory for --run artifacts.
- `--db`: Memory database for --run artifacts.
- `--runs-root`: Root to inspect for existing run artifacts.
- `--notes-root`: Directory containing MECHFERRET.md.
- `--project-root`: OpenVLA project root to inspect.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy for status inspection.
- `--report`: Path for the support JSON report.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret support`
- `mechferret support --json`

### `doctor`

Aliases: `/doctor`

Check config, packages, corpus, and registry health.

Usage:

```text
usage: mechferret doctor [-h] [--json] [--strict] [--all-integrations]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--json`: Print machine-readable JSON.
- `--strict`: Exit nonzero if core release-critical checks fail.
- `--all-integrations`: Treat optional packages, API keys, cluster, and manifests as required.

Examples:

- `mechferret doctor --strict`
- `mechferret doctor --json`

## Research

### `run`

Run research on a question.

Usage:

```text
usage: mechferret run [-h] [--source SOURCE] [--url URL] [--out OUT] [--db DB]
                      [--max-rounds MAX_ROUNDS] [--openai]
                      [--provider {auto,local,openai,anthropic}]
                      [--model MODEL] [--no-memory] [--seed-corpus] [--json]
                      question
```

Positionals:

- `question`: Research question to investigate.

Options:

- `-h`, `--help`: show this help message and exit
- `--source`: File or directory of seed documents.
- `--url`: URL to fetch as a source.
- `--out`: Output directory.
- `--db`: SQLite memory path.
- `--max-rounds`: Maximum retrieval/synthesis rounds.
- `--openai`: Use OpenAI Responses API web search when available.
- `--provider`: choices: `auto`, `local`, `openai`, `anthropic`; Provider for model-assisted synthesis.
- `--model`: Override the configured provider model.
- `--no-memory`: Do not recall prior-run memory.
- `--seed-corpus`: Use the packaged demo corpus when no sources, memory, or provider research are available.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret run "What should I investigate?" --seed-corpus --out runs/custom`
- `mechferret run "What should I investigate?" --seed-corpus --json`

### `demo`

Run the built-in hackathon demo corpus.

Usage:

```text
usage: mechferret demo [-h] [--out OUT] [--db DB] [--max-rounds MAX_ROUNDS]
                       [--openai] [--provider {auto,local,openai,anthropic}]
                       [--model MODEL] [--with-memory] [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--out`: Output directory for demo artifacts.
- `--db`: SQLite memory path.
- `--max-rounds`: Maximum retrieval/synthesis rounds.
- `--openai`: Use OpenAI web search when available.
- `--provider`: choices: `auto`, `local`, `openai`, `anthropic`; Provider for demo synthesis.
- `--model`: Override the configured provider model.
- `--with-memory`: Recall prior-run memory during the demo.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret demo --out runs/demo`
- `mechferret demo --json`

### `goal`

Aliases: `/goal`, `loop`, `/loop`

Loop research/experiments until a target acceptance probability is reached.

Usage:

```text
usage: mechferret goal [-h] [--venue VENUE] [--target TARGET]
                       [--source SOURCE] [--url URL] [--out OUT] [--db DB]
                       [--max-iterations MAX_ITERATIONS]
                       [--max-rounds MAX_ROUNDS]
                       [--provider {auto,local,openai,anthropic}]
                       [--model MODEL] [--no-memory] [--seed-corpus] [--json]
                       question
```

Positionals:

- `question`: Research question or project goal.

Options:

- `-h`, `--help`: show this help message and exit
- `--venue`: Target venue or acceptance bar.
- `--target`: Target estimated acceptance probability.
- `--source`: File or directory of seed documents.
- `--url`: URL to fetch as a source.
- `--out`: Output directory.
- `--db`: SQLite memory path.
- `--max-iterations`: Maximum improve/evaluate loop iterations.
- `--max-rounds`: Maximum retrieval/synthesis rounds per iteration.
- `--provider`: choices: `auto`, `local`, `openai`, `anthropic`; Provider for model-assisted synthesis.
- `--model`: Override the configured provider model.
- `--no-memory`: Do not recall prior-run memory.
- `--seed-corpus`: Use the packaged demo corpus when no sources, memory, or provider research are available.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret goal "Make this investigation publishable" --seed-corpus --max-iterations 1 --max-rounds 1 --json`

### `discover`

Aliases: `/discover`

Autonomous interpretability discovery loop: hypothesize, experiment, critique, synthesize.

Usage:

```text
usage: mechferret discover [-h] [--skill SKILL]
                           [--task {ioi,induction,greater_than,factual_recall}]
                           [--model MODEL]
                           [--backend {auto,synthetic,transformer_lens}]
                           [--source SOURCE] [--url URL] [--out OUT] [--db DB]
                           [--max-rounds MAX_ROUNDS]
                           [--max-experiments MAX_EXPERIMENTS]
                           [--max-gpu-seconds MAX_GPU_SECONDS]
                           [--provider {auto,local,openai,anthropic}]
                           [--llm-model LLM_MODEL] [--no-memory]
                           [--allow-mismatch] [--seed-corpus] [--json]
                           [question]
```

Positionals:

- `question`: Research question (optional if --skill is given).

Options:

- `-h`, `--help`: show this help message and exit
- `--skill`: Named skill/playbook (see `mechferret /skills`) or a path to a skill JSON.
- `--task`: choices: `ioi`, `induction`, `greater_than`, `factual_recall`; Interpretability task.
- `--model`: Model to investigate (e.g. gpt2, pythia-160m).
- `--backend`: choices: `auto`, `synthetic`, `transformer_lens`; Experiment backend for interpretability probes.
- `--source`: Prior-art documents to ground hypotheses.
- `--url`: URL to fetch as prior art.
- `--out`: Output directory for discovery artifacts.
- `--db`: SQLite memory path.
- `--max-rounds`: Override the budget's max experiment rounds.
- `--max-experiments`: Override the budget's max experiments.
- `--max-gpu-seconds`: Override the budget's GPU-second ceiling.
- `--provider`: choices: `auto`, `local`, `openai`, `anthropic`; Provider for prior-art search and critique.
- `--llm-model`: Override the configured provider model for prior-art search.
- `--no-memory`: Do not recall prior-run memory.
- `--allow-mismatch`: Run even if the prompt appears mismatched to the chosen skill/task.
- `--seed-corpus`: Use the packaged demo corpus as prior art when no explicit sources, memory, or provider research are available.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret discover --skill ioi-circuit --backend synthetic --json`

### `sae`

Aliases: `/sae`

SAE project workflows, including OpenVLA.

Usage:

```text
usage: mechferret sae [-h] [--project-root PROJECT_ROOT] [--manifest MANIFEST]
                      [--image-dir IMAGE_DIR] [--instruction INSTRUCTION]
                      [--action-label ACTION_LABEL] [--limit LIMIT] [--force]
                      [--out OUT] [--cache-dir CACHE_DIR]
                      [--checkpoint CHECKPOINT] [--eval-dir EVAL_DIR]
                      [--features-dir FEATURES_DIR]
                      [--top-features TOP_FEATURES] [--max-files MAX_FILES]
                      [--d-model D_MODEL] [--tokens TOKENS] [--steps STEPS]
                      [--k K] [--json]
                      {openvla}
                      [{status,init,plan,commands,validate-manifest,create-manifest,smoke,eval,features,dossier}]
```

Positionals:

- `project`: choices: `openvla`; SAE project to operate on.
- `action`: choices: `status`, `init`, `plan`, `commands`, `validate-manifest`, `create-manifest`, `smoke`, `eval`, `features`, `dossier`; Project workflow action.

Options:

- `-h`, `--help`: show this help message and exit
- `--project-root`: OpenVLA SAE workflow directory.
- `--manifest`: JSONL manifest of image_path/instruction rows.
- `--image-dir`: Directory of images for create-manifest.
- `--instruction`: Starter instruction for create-manifest rows.
- `--action-label`: Optional action label value for create-manifest rows.
- `--limit`: Maximum images to include.
- `--force`: Overwrite an existing manifest for create-manifest.
- `--out`: Output directory for plan artifacts.
- `--cache-dir`: Activation cache directory for eval.
- `--checkpoint`: SAE checkpoint for eval.
- `--eval-dir`: Eval artifact directory for dossier.
- `--features-dir`: Feature artifact directory for dossier.
- `--top-features`: Number of top SAE features for features.
- `--max-files`: Maximum cached activation files for features.
- `--d-model`: Synthetic hidden size for smoke.
- `--tokens`: Synthetic token count for smoke.
- `--steps`: Training steps for smoke.
- `--k`: Top-K value for smoke.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret sae openvla status`
- `mechferret sae openvla plan --json`

## Artifacts

### `runs`

Aliases: `/runs`, `list-runs`

List recent run artifacts with audit and artifact status.

Usage:

```text
usage: mechferret runs [-h] [--runs-root RUNS_ROOT] [--limit LIMIT]
                       [--no-audit] [--select {latest,best,ready}] [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search for run artifacts.
- `--limit`: Maximum runs to show.
- `--no-audit`: Skip audit checks for faster listing.
- `--select`: choices: `latest`, `best`, `ready`; Print the selected run policy in JSON output.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret runs --select best`
- `mechferret runs --json`

### `open`

Aliases: `/open`

Resolve a generated artifact path, optionally opening it in a browser.

Usage:

```text
usage: mechferret open [-h] [--runs-root RUNS_ROOT]
                       [--project-root PROJECT_ROOT]
                       [--select {latest,best,ready}] [--browser] [--json]
                       [target]
```

Positionals:

- `target`: all | quickstart | ci | report | markdown | graph | evals | trace | experiments | discoveries | paper | review | bundle | manifest | pdf | run | openvla | explicit path

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search for run artifacts.
- `--project-root`: OpenVLA project root for target=openvla.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy for run-bound artifacts.
- `--browser`: Open the resolved artifact with the system browser.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret open report --select best`
- `mechferret open all`

### `cost`

Aliases: `/cost`

Estimate cost/usage from a run artifact.

Usage:

```text
usage: mechferret cost [-h] [--runs-root RUNS_ROOT]
                       [--select {latest,best,ready}] [--json]
                       [run_json]
```

Positionals:

- `run_json`: Run artifact (defaults to selected runs/**/run.json).

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search when run_json is omitted.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy when run_json is omitted.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret cost --select best --json`

### `resume`

Aliases: `/resume`

Summarize a prior run artifact.

Usage:

```text
usage: mechferret resume [-h] [--runs-root RUNS_ROOT]
                         [--select {latest,best,ready}] [--json]
                         [run_json]
```

Positionals:

- `run_json`: Run artifact (defaults to selected runs/**/run.json).

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search when run_json is omitted.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy when run_json is omitted.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret resume --select best`

### `inspect`

Print a compact summary of a run JSON artifact.

Usage:

```text
usage: mechferret inspect [-h] [--runs-root RUNS_ROOT]
                          [--select {latest,best,ready}] [--json]
                          [run_json]
```

Positionals:

- `run_json`: Run artifact (defaults to selected runs/**/run.json).

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search when run_json is omitted.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy when run_json is omitted.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret inspect --select best --json`

### `audit`

Aliases: `/audit`

Run offline paper-readiness gates on a dossier.

Usage:

```text
usage: mechferret audit [-h] [--runs-root RUNS_ROOT]
                        [--select {latest,best,ready}] [--json] [--strict]
                        [run_json]
```

Positionals:

- `run_json`: Run artifact to audit (defaults to selected runs/**/run.json).

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search when run_json is omitted.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy when run_json is omitted.
- `--json`: Print machine-readable JSON.
- `--strict`: Exit nonzero if any audit gate fails.

Examples:

- `mechferret audit --select best --strict`
- `mechferret audit --json`

### `verify`

Aliases: `/verify`

Verify run manifest integrity and artifact existence.

Usage:

```text
usage: mechferret verify [-h] [--runs-root RUNS_ROOT]
                         [--select {latest,best,ready}] [--repair] [--json]
                         [--strict]
                         [run_json]
```

Positionals:

- `run_json`: Run artifact to verify (defaults to selected runs/**/run.json).

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search when run_json is omitted.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy when run_json is omitted.
- `--repair`: Refresh manifest.json when only manifest coverage metadata is stale.
- `--json`: Print machine-readable JSON.
- `--strict`: Exit nonzero if verification fails.

Examples:

- `mechferret verify --select best --strict`
- `mechferret verify --repair --json`

### `paper`

Aliases: `/paper`

Generate main.tex from a run artifact.

Usage:

```text
usage: mechferret paper [-h] [--runs-root RUNS_ROOT]
                        [--select {latest,best,ready}] [--out OUT] [--compile]
                        [--compile-timeout COMPILE_TIMEOUT]
                        [--provider {auto,local,openai,anthropic}]
                        [--model MODEL] [--json]
                        [run_json]
```

Positionals:

- `run_json`: Run artifact to write from (defaults to selected runs/**/run.json).

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search when run_json is omitted.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy when run_json is omitted.
- `--out`: Output directory for main.tex (defaults to the run artifact's paper/ directory).
- `--compile`: Compile main.pdf with tectonic if installed.
- `--compile-timeout`: Seconds to wait for tectonic before reporting a timeout.
- `--provider`: choices: `auto`, `local`, `openai`, `anthropic`; Provider for paper drafting.
- `--model`: Override the configured provider model.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret paper --select best --json`
- `mechferret paper --select best --compile`

### `review-paper`

Aliases: `/review-paper`

Review a generated paper with a configured model.

Usage:

```text
usage: mechferret review-paper [-h] [--runs-root RUNS_ROOT]
                               [--select {latest,best,ready}] [--out OUT]
                               [--provider {auto,openai,anthropic}]
                               [--model MODEL] [--json]
                               [paper_tex]
```

Positionals:

- `paper_tex`: Paper to review (defaults to selected run-bound paper).

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search when paper_tex is omitted.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy when paper_tex is omitted.
- `--out`: Directory for review.md (defaults beside the paper).
- `--provider`: choices: `auto`, `openai`, `anthropic`; Provider for paper review.
- `--model`: Override the configured provider model.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret review-paper --select best --json`

### `bundle`

Aliases: `/bundle`

Create a shareable zip bundle for a run dossier.

Usage:

```text
usage: mechferret bundle [-h] [--runs-root RUNS_ROOT]
                         [--select {latest,best,ready}] [--out OUT]
                         [--notes-root NOTES_ROOT]
                         [--project-root PROJECT_ROOT] [--json]
                         [run_json]
```

Positionals:

- `run_json`: Run artifact to bundle (defaults to selected runs/**/run.json).

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search when run_json is omitted.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy when run_json is omitted.
- `--out`: Output .zip path or directory (defaults beside the run).
- `--notes-root`: Directory containing MECHFERRET.md.
- `--project-root`: OpenVLA project root.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret bundle --select best`
- `mechferret bundle --select best --json`

### `verify-bundle`

Aliases: `/verify-bundle`

Verify a shareable bundle zip using its portable manifest.

Usage:

```text
usage: mechferret verify-bundle [-h] [--runs-root RUNS_ROOT]
                                [--select {latest,best,ready}] [--json]
                                [--strict]
                                [bundle_zip]
```

Positionals:

- `bundle_zip`: Bundle zip to verify (defaults to selected run-bound bundle artifact).

Options:

- `-h`, `--help`: show this help message and exit
- `--runs-root`: Root to search when bundle_zip is omitted.
- `--select`: choices: `latest`, `best`, `ready`; Run-selection policy when bundle_zip is omitted.
- `--json`: Print machine-readable JSON.
- `--strict`: Exit nonzero if bundle verification fails.

Examples:

- `mechferret verify-bundle --select best --strict`

## Config

### `login`

Aliases: `/login`

Store an OpenAI or Anthropic API key.

Usage:

```text
usage: mechferret login [-h] [--api-key API_KEY] [--model MODEL]
                        [--no-default] [--json]
                        {anthropic,openai}
```

Positionals:

- `provider`: choices: `anthropic`, `openai`; Provider to configure.

Options:

- `-h`, `--help`: show this help message and exit
- `--api-key`: API key. If omitted, MechFerret prompts securely.
- `--model`: Default model for this provider.
- `--no-default`: Store key without making this the default provider.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret login openai --api-key "$OPENAI_API_KEY" --json`

### `api`

Aliases: `/api`

Show or change provider configuration.

Usage:

```text
usage: mechferret api [-h] [--provider {anthropic,openai,local}]
                      [--api-key API_KEY] [--model MODEL] [--show]
                      [--clear {anthropic,openai}] [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--provider`: choices: `anthropic`, `openai`, `local`; Set default provider.
- `--api-key`: Store or replace the key for --provider.
- `--model`: Store or replace the default model for --provider.
- `--show`: Show configured provider status.
- `--clear`: choices: `anthropic`, `openai`; Remove a stored provider key.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret api --show --json`

### `skills`

Aliases: `/skills`

List or show interpretability skills/playbooks.

Usage:

```text
usage: mechferret skills [-h] [--json] [name]
```

Positionals:

- `name`: Show details for one skill.

Options:

- `-h`, `--help`: show this help message and exit
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret skills --json`
- `mechferret skills ioi-circuit`

### `repl`

Aliases: `chat`, `/repl`

Launch the interactive prompt (default when run with no arguments).

Usage:

```text
usage: mechferret repl [-h]
```

Options:

- `-h`, `--help`: show this help message and exit

Examples:

- `mechferret repl`

### `registry`

Aliases: `/registry`

List available tools, tasks, playbooks, and evaluators.

Usage:

```text
usage: mechferret registry [-h] [--kind {tool,task,playbook,evaluator}]
                           [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--kind`: choices: `tool`, `task`, `playbook`, `evaluator`; Limit registry output to one item kind.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret registry --kind tool --json`

### `memory`

Aliases: `/memory`

Inspect or clear research memory.

Usage:

```text
usage: mechferret memory [-h] [--db DB] [--recent RECENT] [--clear] [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--db`: SQLite memory path.
- `--recent`: Show recent remembered runs.
- `--clear`: Delete the memory database.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret memory --recent 5`
- `mechferret memory --json`

### `tool-results`

Aliases: `/tool-results`

List or clean saved large agent-tool outputs.

Usage:

```text
usage: mechferret tool-results [-h] [--limit LIMIT] [--clean]
                               [--keep-latest KEEP_LATEST]
                               [--max-age-days MAX_AGE_DAYS] [--confirm]
                               [--json]
```

Options:

- `-h`, `--help`: show this help message and exit
- `--limit`: Maximum saved outputs to show.
- `--clean`: Clean stale saved outputs instead of listing them.
- `--keep-latest`: Keep at least this many newest saved outputs when cleaning.
- `--max-age-days`: Delete saved outputs older than this many days when cleaning.
- `--confirm`: Actually delete files for --clean. Without this, cleanup is a dry run.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret tool-results`
- `mechferret tool-results --clean --json`

## Compute

### `modal`

Aliases: `/modal`

Connect to Modal for GPU compute and run experiments remotely.

Usage:

```text
usage: mechferret modal [-h] [--skill SKILL]
                        [--task {ioi,induction,greater_than,factual_recall}]
                        [--model MODEL] [--out OUT] [--json]
                        [{status,setup,run,deploy}] [question]
```

Positionals:

- `action`: choices: `status`, `setup`, `run`, `deploy`; Modal workflow action.
- `question`: Question for remote run actions.

Options:

- `-h`, `--help`: show this help message and exit
- `--skill`: Skill to run remotely (e.g. ioi-circuit).
- `--task`: choices: `ioi`, `induction`, `greater_than`, `factual_recall`; Interpretability task for remote experiments.
- `--model`: Model to investigate remotely.
- `--out`: Output directory for Modal artifacts.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret modal status --json`

### `cluster`

Aliases: `/cluster`

Run experiments on a generic SLURM cluster over SSH (srun).

Usage:

```text
usage: mechferret cluster [-h] [--skill SKILL]
                          [--task {ioi,induction,greater_than,factual_recall}]
                          [--model MODEL] [--out OUT] [--dry-run] [--json]
                          [{status,setup,run}] [question]
```

Positionals:

- `action`: choices: `status`, `setup`, `run`; Cluster workflow action.
- `question`: Question for remote run actions.

Options:

- `-h`, `--help`: show this help message and exit
- `--skill`: Skill to run remotely (e.g. ioi-circuit).
- `--task`: choices: `ioi`, `induction`, `greater_than`, `factual_recall`; Interpretability task for cluster experiments.
- `--model`: Model to investigate on the cluster.
- `--out`: Output directory for cluster artifacts.
- `--dry-run`: Print the ssh+srun command without executing.
- `--json`: Print machine-readable JSON.

Examples:

- `mechferret cluster run --skill ioi-circuit --dry-run --json`
