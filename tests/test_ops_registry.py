import json
import hashlib
import os
import tempfile
import tomllib
import unittest
import warnings
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mechferret.costs import estimate_run_cost
from mechferret.discovery import DiscoveryController
from mechferret.ops import bundle_run_artifacts, doctor, init_project_notes, list_run_artifacts, memory_recent, memory_summary, print_artifact_result, print_project_status, print_run_list, project_status, quickstart, resolve_artifact, run_quickstart, select_run_artifact, selftest, summarize_run_artifact, verify_bundle_artifacts, verify_run_artifacts
from mechferret.provenance import refresh_run_manifest
from mechferret.registry import all_items, items_by_kind
from mechferret.controller import MechFerret
from mechferret import __version__


class OpsRegistryTest(unittest.TestCase):
    def test_registry_has_core_items(self):
        names = {item.name for item in all_items()}
        self.assertIn("goal_loop", names)
        self.assertIn("provider_research", names)
        self.assertGreaterEqual(len(items_by_kind("tool")), 5)

    def test_cli_version_reports_runtime_and_matches_project_metadata(self):
        from mechferret.cli import main

        metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["version"], __version__)

        out = StringIO()
        with redirect_stdout(out):
            main(["version"])
        self.assertIn(f"MechFerret {__version__}", out.getvalue())

        json_out = StringIO()
        with redirect_stdout(json_out):
            main(["version", "--json"])
        payload = json.loads(json_out.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["name"], "mechferret")
        self.assertEqual(payload["version"], __version__)
        self.assertIn("python", payload)
        self.assertIn("config_path", payload)

        flag_out = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(flag_out):
                main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn(__version__, flag_out.getvalue())

    def test_cli_commands_json_describes_installed_command_surface(self):
        from mechferret.cli import COMMAND_EXAMPLES, main

        help_out = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(help_out):
                main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        rendered_help = help_out.getvalue()
        self.assertIn("usage: mechferret [-h] [--version] COMMAND ...", rendered_help)
        self.assertNotIn("{version,/version", rendered_help)
        self.assertIn("completion (/completion)", rendered_help)

        out = StringIO()
        with redirect_stdout(out):
            main(["commands", "--json"])
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["name"], "mechferret")
        self.assertEqual(payload["version"], __version__)
        self.assertGreaterEqual(payload["count"], 25)
        workflow_names = {workflow["name"] for workflow in payload["workflows"]}
        self.assertIn("first_run", workflow_names)
        self.assertIn("publish_dossier", workflow_names)
        first_run = next(workflow for workflow in payload["workflows"] if workflow["name"] == "first_run")
        self.assertEqual(
            first_run["commands"][:5],
            ["mechferret init", "mechferret quickstart --run", "mechferret status", "mechferret next", "mechferret support"],
        )
        names = {command["name"] for command in payload["commands"]}
        self.assertIn("run", names)
        self.assertIn("commands", names)
        self.assertIn("completion", names)
        self.assertIn("sae", names)
        self.assertIn("selftest", names)
        self.assertIn("support", names)
        self.assertIn("next", names)
        self.assertNotIn("/run", names)
        self.assertEqual(sorted(COMMAND_EXAMPLES), sorted(names))

        by_name = {command["name"]: command for command in payload["commands"]}
        self.assertFalse([name for name, command in by_name.items() if not command["examples"]])
        self.assertIn("/commands", by_name["commands"]["aliases"])
        self.assertIn("help", by_name["commands"]["aliases"])
        self.assertIn("/completion", by_name["completion"]["aliases"])
        self.assertTrue(any(option["flags"] == ["--json"] for option in by_name["commands"]["options"]))
        self.assertTrue(any(option["flags"] == ["--out"] for option in by_name["commands"]["options"]))
        group_option = next(option for option in by_name["commands"]["options"] if option["flags"] == ["--group"])
        self.assertEqual(group_option["choices"], ["start", "research", "artifacts", "config", "compute"])
        workflow_option = next(option for option in by_name["commands"]["options"] if option["flags"] == ["--workflow"])
        self.assertEqual(workflow_option["choices"], ["all", "first_run", "publish_dossier", "support_report", "openvla_sae"])
        self.assertTrue(any(option["flags"] == ["--source"] for option in by_name["run"]["options"]))
        self.assertTrue(any(positional["dest"] == "question" for positional in by_name["run"]["positionals"]))
        self.assertTrue(any(positional["dest"] == "project" for positional in by_name["sae"]["positionals"]))
        self.assertIn('mechferret run "What should I investigate?" --seed-corpus --out runs/custom', by_name["run"]["examples"])
        self.assertIn(
            'mechferret goal "Make this investigation publishable" --seed-corpus --max-iterations 1 --max-rounds 1 --json',
            by_name["goal"]["examples"],
        )
        self.assertIn("mechferret paper --select best --json", by_name["paper"]["examples"])
        self.assertIn("mechferret selftest --json", by_name["selftest"]["examples"])
        self.assertIn("mechferret support", by_name["support"]["examples"])
        self.assertIn("mechferret next --json", by_name["next"]["examples"])

        text_out = StringIO()
        with redirect_stdout(text_out):
            main(["commands"])
        rendered_commands = text_out.getvalue()
        self.assertIn("MechFerret commands", rendered_commands)
        self.assertIn("\nStart:", rendered_commands)
        self.assertIn("\nResearch:", rendered_commands)
        self.assertIn("\nArtifacts:", rendered_commands)
        self.assertIn("\nConfig:", rendered_commands)
        self.assertIn("\nCompute:", rendered_commands)
        self.assertIn("run:", rendered_commands)
        self.assertIn("\nWorkflows:", rendered_commands)
        self.assertIn("first_run:", rendered_commands)
        self.assertIn("quickstart --run", rendered_commands)
        self.assertLess(rendered_commands.index("quickstart"), rendered_commands.index("run:"))
        self.assertLess(rendered_commands.index("run:"), rendered_commands.index("bundle"))
        self.assertLess(rendered_commands.index("bundle"), rendered_commands.index("modal"))

        group_out = StringIO()
        with redirect_stdout(group_out):
            main(["commands", "--group", "research"])
        rendered_group = group_out.getvalue()
        self.assertIn("MechFerret research commands", rendered_group)
        self.assertIn("\nResearch:", rendered_group)
        self.assertIn("run:", rendered_group)
        self.assertIn("sae", rendered_group)
        self.assertNotIn("bundle", rendered_group)

        group_json_out = StringIO()
        with redirect_stdout(group_json_out):
            main(["commands", "--group", "artifacts", "--json"])
        group_payload = json.loads(group_json_out.getvalue())
        self.assertTrue(group_payload["ok"])
        self.assertEqual(group_payload["group"], "artifacts")
        self.assertEqual(group_payload["workflows"], [])
        group_names = {command["name"] for command in group_payload["commands"]}
        self.assertIn("bundle", group_names)
        self.assertNotIn("run", group_names)

        search_out = StringIO()
        with redirect_stdout(search_out):
            main(["commands", "--search", "bundle"])
        rendered_search = search_out.getvalue()
        self.assertIn("MechFerret commands matching 'bundle'", rendered_search)
        self.assertIn("bundle", rendered_search)
        self.assertIn("verify-bundle", rendered_search)
        self.assertNotIn("run:", rendered_search)

        workflow_search_out = StringIO()
        with redirect_stdout(workflow_search_out):
            main(["commands", "--search", "publish dossier", "--json"])
        workflow_search_payload = json.loads(workflow_search_out.getvalue())
        self.assertTrue(workflow_search_payload["ok"])
        self.assertEqual(workflow_search_payload["count"], 1)
        self.assertEqual(workflow_search_payload["workflow_count"], 1)
        self.assertEqual(workflow_search_payload["commands"][0]["name"], "commands")
        self.assertEqual(workflow_search_payload["workflows"][0]["name"], "publish_dossier")

        workflow_search_text_out = StringIO()
        with redirect_stdout(workflow_search_text_out):
            main(["commands", "--search", "publish dossier"])
        rendered_workflow_search = workflow_search_text_out.getvalue()
        self.assertIn("MechFerret commands matching 'publish dossier' (1 command, 1 workflow):", rendered_workflow_search)
        self.assertIn("commands (/commands", rendered_workflow_search)
        self.assertIn("Workflows:", rendered_workflow_search)
        self.assertIn("publish_dossier:", rendered_workflow_search)
        self.assertIn("mechferret bundle --select best", rendered_workflow_search)

        mixed_search_out = StringIO()
        with redirect_stdout(mixed_search_out):
            main(["commands", "--search", "first_run", "--markdown"])
        mixed_search_markdown = mixed_search_out.getvalue()
        self.assertIn("_Filtered by search: `first_run`._", mixed_search_markdown)
        self.assertIn("## Workflows", mixed_search_markdown)
        self.assertIn("### First Run", mixed_search_markdown)
        self.assertIn("### `commands`", mixed_search_markdown)

        group_search_out = StringIO()
        with redirect_stdout(group_search_out):
            main(["commands", "--group", "artifacts", "--search", "bundle"])
        rendered_group_search = group_search_out.getvalue()
        self.assertIn("MechFerret artifacts commands matching 'bundle'", rendered_group_search)
        self.assertIn("verify-bundle", rendered_group_search)
        self.assertNotIn("commands (/commands", rendered_group_search)

        markdown_out = StringIO()
        with redirect_stdout(markdown_out):
            main(["commands", "--group", "research", "--markdown"])
        rendered_markdown = markdown_out.getvalue()
        self.assertIn("# MechFerret Commands", rendered_markdown)
        self.assertIn("_Filtered by group: `research`._", rendered_markdown)
        self.assertIn("## Research", rendered_markdown)
        self.assertNotIn("## Workflows", rendered_markdown)
        self.assertIn("### `run`", rendered_markdown)
        self.assertIn("```text\nusage: mechferret run", rendered_markdown)
        self.assertIn("- `--source`: File or directory of seed documents.", rendered_markdown)
        self.assertIn("- `mechferret run \"What should I investigate?\" --seed-corpus", rendered_markdown)
        self.assertNotIn("## Artifacts", rendered_markdown)

        full_markdown_out = StringIO()
        with redirect_stdout(full_markdown_out):
            main(["commands", "--markdown"])
        full_markdown = full_markdown_out.getvalue()
        self.assertIn("## Workflows", full_markdown)
        self.assertIn("### First Run", full_markdown)
        self.assertIn("- `mechferret quickstart --run`", full_markdown)
        self.assertLess(full_markdown.index("## Workflows"), full_markdown.index("## Start"))

        workflow_out = StringIO()
        with redirect_stdout(workflow_out):
            main(["commands", "--workflow", "first_run"])
        rendered_workflow = workflow_out.getvalue()
        self.assertIn("first_run: First Run", rendered_workflow)
        self.assertIn("mechferret quickstart --run", rendered_workflow)
        self.assertNotIn("usage: mechferret run", rendered_workflow)

        workflow_list_out = StringIO()
        with redirect_stdout(workflow_list_out):
            main(["commands", "--workflow"])
        rendered_workflow_list = workflow_list_out.getvalue()
        self.assertIn("MechFerret workflows (4):", rendered_workflow_list)
        self.assertIn("first_run: First Run", rendered_workflow_list)
        self.assertIn("openvla_sae: OpenVLA SAE", rendered_workflow_list)

        workflow_all_out = StringIO()
        with redirect_stdout(workflow_all_out):
            main(["commands", "--workflow", "all", "--json"])
        workflow_all_payload = json.loads(workflow_all_out.getvalue())
        self.assertTrue(workflow_all_payload["ok"])
        self.assertTrue(workflow_all_payload["workflow_list"])
        self.assertEqual(workflow_all_payload["workflow"], "all")
        self.assertEqual(workflow_all_payload["count"], len(workflow_names))
        self.assertEqual({workflow["name"] for workflow in workflow_all_payload["workflows"]}, workflow_names)

        workflow_json_out = StringIO()
        with redirect_stdout(workflow_json_out):
            main(["commands", "--workflow", "publish_dossier", "--json"])
        workflow_payload = json.loads(workflow_json_out.getvalue())
        self.assertTrue(workflow_payload["ok"])
        self.assertTrue(workflow_payload["workflow_only"])
        self.assertEqual(workflow_payload["workflow"], "publish_dossier")
        self.assertEqual(workflow_payload["workflows"][0]["name"], "publish_dossier")
        self.assertIn("mechferret verify-bundle --select best --strict", workflow_payload["workflows"][0]["commands"])

        workflow_alias_out = StringIO()
        with redirect_stdout(workflow_alias_out):
            main(["commands", "--workflow", "First Run", "--json"])
        workflow_alias_payload = json.loads(workflow_alias_out.getvalue())
        self.assertTrue(workflow_alias_payload["ok"])
        self.assertEqual(workflow_alias_payload["workflow"], "first_run")
        self.assertEqual(workflow_alias_payload["workflow_query"], "First Run")

        workflow_hyphen_out = StringIO()
        with redirect_stdout(workflow_hyphen_out):
            main(["commands", "--workflow", "publish-dossier"])
        self.assertIn("publish_dossier: Publish Dossier", workflow_hyphen_out.getvalue())

        workflow_markdown_out = StringIO()
        with redirect_stdout(workflow_markdown_out):
            main(["commands", "--workflow", "support_report", "--markdown"])
        workflow_markdown = workflow_markdown_out.getvalue()
        self.assertIn("# MechFerret Workflow", workflow_markdown)
        self.assertIn("_Filtered by workflow: `support_report`._", workflow_markdown)
        self.assertIn("## Support Report", workflow_markdown)
        self.assertNotIn("## Workflows", workflow_markdown)
        self.assertIn("- `mechferret support --json`", workflow_markdown)
        self.assertIn("- `mechferret next --json`", workflow_markdown)

        with tempfile.TemporaryDirectory() as tmp:
            markdown_path = Path(tmp) / "refs" / "research.md"
            write_out = StringIO()
            with redirect_stdout(write_out):
                main(["commands", "--group", "research", "--markdown", "--out", str(markdown_path)])
            self.assertIn(f"Wrote command markdown: {markdown_path}", write_out.getvalue())
            written = markdown_path.read_text(encoding="utf-8")
            self.assertIn("# MechFerret Commands", written)
            self.assertIn("## Research", written)
            self.assertIn("### `discover`", written)
            self.assertIn("Options:", written)
            self.assertNotIn("## Artifacts", written)

        search_json_out = StringIO()
        with redirect_stdout(search_json_out):
            main(["commands", "--search", "source", "--json"])
        search_payload = json.loads(search_json_out.getvalue())
        self.assertTrue(search_payload["ok"])
        self.assertEqual(search_payload["search"], "source")
        search_names = {command["name"] for command in search_payload["commands"]}
        self.assertIn("run", search_names)
        self.assertTrue(
            any(
                option["flags"] == ["--source"]
                for command in search_payload["commands"]
                for option in command["options"]
            )
        )

        multi_term_search_out = StringIO()
        with redirect_stdout(multi_term_search_out):
            main(["commands", "--search", "bundle strict", "--json"])
        multi_term_payload = json.loads(multi_term_search_out.getvalue())
        self.assertTrue(multi_term_payload["ok"])
        self.assertEqual([command["name"] for command in multi_term_payload["commands"]], ["verify-bundle"])
        self.assertEqual([workflow["name"] for workflow in multi_term_payload["workflows"]], ["publish_dossier"])

        option_search_out = StringIO()
        with redirect_stdout(option_search_out):
            main(["commands", "--search", "max gpu", "--json"])
        option_search_payload = json.loads(option_search_out.getvalue())
        self.assertTrue(option_search_payload["ok"])
        self.assertGreaterEqual(option_search_payload["count"], 2)
        self.assertEqual(option_search_payload["commands"][0]["name"], "discover")
        self.assertIn("commands", {command["name"] for command in option_search_payload["commands"]})

        examples_out = StringIO()
        with redirect_stdout(examples_out):
            main(["commands", "--examples"])
        rendered_examples = examples_out.getvalue()
        self.assertIn("MechFerret examples:", rendered_examples)
        self.assertIn("quickstart:", rendered_examples)
        self.assertIn("mechferret quickstart --run", rendered_examples)
        self.assertIn("mechferret quickstart --mode ci --run", rendered_examples)
        self.assertNotIn("Options:", rendered_examples)

        run_examples_out = StringIO()
        with redirect_stdout(run_examples_out):
            main(["commands", "run", "--examples"])
        rendered_run_examples = run_examples_out.getvalue()
        self.assertIn("MechFerret examples for run:", rendered_run_examples)
        self.assertIn('mechferret run "What should I investigate?" --seed-corpus', rendered_run_examples)
        self.assertNotIn("usage: mechferret run", rendered_run_examples)

        detail_markdown_out = StringIO()
        with redirect_stdout(detail_markdown_out):
            main(["commands", "run", "--markdown"])
        rendered_detail_markdown = detail_markdown_out.getvalue()
        self.assertIn("## `run`", rendered_detail_markdown)
        self.assertIn("```text\nusage: mechferret run", rendered_detail_markdown)
        self.assertIn("- `--source`: File or directory of seed documents.", rendered_detail_markdown)
        self.assertIn("- `mechferret run \"What should I investigate?\" --seed-corpus", rendered_detail_markdown)

        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "run-command.json"
            write_json_out = StringIO()
            with redirect_stdout(write_json_out):
                main(["commands", "run", "--json", "--out", str(json_path)])
            write_payload = json.loads(write_json_out.getvalue())
            self.assertTrue(write_payload["ok"])
            self.assertEqual(write_payload["format"], "json")
            self.assertEqual(Path(write_payload["path"]), json_path)
            self.assertGreater(write_payload["bytes"], 100)
            written_payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(written_payload["ok"])
            self.assertEqual(written_payload["commands"][0]["name"], "run")

        examples_json_out = StringIO()
        with redirect_stdout(examples_json_out):
            main(["commands", "--search", "bundle", "--examples", "--json"])
        examples_payload = json.loads(examples_json_out.getvalue())
        self.assertTrue(examples_payload["ok"])
        self.assertTrue(examples_payload["examples_only"])
        self.assertEqual(examples_payload["search"], "bundle")
        self.assertIn("bundle", {command["name"] for command in examples_payload["commands"]})
        self.assertTrue(all("options" not in command for command in examples_payload["commands"]))
        self.assertTrue(all(command["examples"] for command in examples_payload["commands"]))

        group_examples_out = StringIO()
        with redirect_stdout(group_examples_out):
            main(["commands", "--group", "start", "--examples", "--json"])
        group_examples_payload = json.loads(group_examples_out.getvalue())
        self.assertTrue(group_examples_payload["ok"])
        self.assertTrue(group_examples_payload["examples_only"])
        self.assertEqual(group_examples_payload["group"], "start")
        group_example_names = {command["name"] for command in group_examples_payload["commands"]}
        self.assertIn("quickstart", group_example_names)
        self.assertNotIn("run", group_example_names)

        examples_markdown_out = StringIO()
        with redirect_stdout(examples_markdown_out):
            main(["commands", "--search", "bundle", "--examples", "--markdown"])
        rendered_examples_markdown = examples_markdown_out.getvalue()
        self.assertIn("# MechFerret Examples", rendered_examples_markdown)
        self.assertIn("_Filtered by search: `bundle`._", rendered_examples_markdown)
        self.assertIn("## `bundle`", rendered_examples_markdown)
        self.assertIn("- `mechferret bundle --select best`", rendered_examples_markdown)

        mixed_out = StringIO()
        mixed_err = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(mixed_out), redirect_stderr(mixed_err):
                main(["commands", "run", "--search", "source", "--json"])
        self.assertEqual(ctx.exception.code, 1)
        mixed_payload = json.loads(mixed_out.getvalue())
        self.assertFalse(mixed_payload["ok"])
        self.assertEqual(mixed_payload["error"], "name cannot be combined with search or group")
        self.assertEqual(mixed_err.getvalue(), "")

        mixed_text_err = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stderr(mixed_text_err):
                main(["commands", "run", "--group", "research"])
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("name cannot be combined with search or group", mixed_text_err.getvalue())
        self.assertIn("commands --group research", mixed_text_err.getvalue())

        format_out = StringIO()
        format_err = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(format_out), redirect_stderr(format_err):
                main(["commands", "--json", "--markdown"])
        self.assertEqual(ctx.exception.code, 1)
        format_payload = json.loads(format_out.getvalue())
        self.assertFalse(format_payload["ok"])
        self.assertEqual(format_payload["error"], "json and markdown are mutually exclusive")
        self.assertEqual(format_err.getvalue(), "")

        workflow_mixed_out = StringIO()
        workflow_mixed_err = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(workflow_mixed_out), redirect_stderr(workflow_mixed_err):
                main(["commands", "--workflow", "first_run", "--examples", "--json"])
        self.assertEqual(ctx.exception.code, 1)
        workflow_mixed_payload = json.loads(workflow_mixed_out.getvalue())
        self.assertFalse(workflow_mixed_payload["ok"])
        self.assertIn("workflow cannot be combined", workflow_mixed_payload["error"])
        self.assertEqual(workflow_mixed_err.getvalue(), "")

        workflow_typo_out = StringIO()
        workflow_typo_err = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(workflow_typo_out), redirect_stderr(workflow_typo_err):
                main(["commands", "--workflow", "first-rn", "--json"])
        self.assertEqual(ctx.exception.code, 1)
        workflow_typo_payload = json.loads(workflow_typo_out.getvalue())
        self.assertFalse(workflow_typo_payload["ok"])
        self.assertEqual(workflow_typo_payload["error"], "unknown workflow")
        self.assertIn("first_run", workflow_typo_payload["suggestions"])
        self.assertEqual(workflow_typo_err.getvalue(), "")

        workflow_typo_text_err = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stderr(workflow_typo_text_err):
                main(["commands", "--workflow", "first-rn"])
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Unknown workflow: first-rn", workflow_typo_text_err.getvalue())
        self.assertIn("Did you mean: first_run", workflow_typo_text_err.getvalue())

        detail_out = StringIO()
        with redirect_stdout(detail_out):
            main(["commands", "run"])
        rendered_detail = detail_out.getvalue()
        self.assertIn("usage: mechferret run", rendered_detail)
        self.assertIn("Positionals:", rendered_detail)
        self.assertIn("question", rendered_detail)
        self.assertIn("--source", rendered_detail)
        self.assertIn("Examples:", rendered_detail)
        self.assertIn('mechferret run "What should I investigate?" --seed-corpus', rendered_detail)

        alias_out = StringIO()
        with redirect_stdout(alias_out):
            main(["commands", "/completion", "--json"])
        alias_payload = json.loads(alias_out.getvalue())
        self.assertTrue(alias_payload["ok"])
        self.assertEqual(alias_payload["query"], "/completion")
        self.assertEqual(alias_payload["count"], 1)
        self.assertEqual(alias_payload["commands"][0]["name"], "completion")
        self.assertIn("mechferret completion zsh --json", alias_payload["commands"][0]["examples"])
        self.assertEqual(alias_payload["commands"][0]["positionals"][0]["choices"], ["bash", "zsh", "fish"])

        example_search_out = StringIO()
        with redirect_stdout(example_search_out):
            main(["commands", "--search", "publishable", "--json"])
        example_search_payload = json.loads(example_search_out.getvalue())
        self.assertTrue(example_search_payload["ok"])
        self.assertEqual([command["name"] for command in example_search_payload["commands"]], ["goal"])

        missing_out = StringIO()
        missing_err = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(missing_out), redirect_stderr(missing_err):
                main(["commands", "verfy", "--json"])
        self.assertEqual(ctx.exception.code, 1)
        missing_payload = json.loads(missing_out.getvalue())
        self.assertFalse(missing_payload["ok"])
        self.assertEqual(missing_payload["error"], "unknown command")
        self.assertIn("verify", missing_payload["suggestions"])
        self.assertIn("run", missing_payload["available"])
        self.assertEqual(missing_err.getvalue(), "")

        typo_err = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stderr(typo_err):
                main(["verfy"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("Unknown command: verfy", typo_err.getvalue())
        self.assertIn("Did you mean: verify", typo_err.getvalue())
        self.assertNotIn("invalid choice", typo_err.getvalue())

        typo_json_out = StringIO()
        typo_json_err = StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(typo_json_out), redirect_stderr(typo_json_err):
                main(["verfy", "--json"])
        self.assertEqual(ctx.exception.code, 2)
        typo_payload = json.loads(typo_json_out.getvalue())
        self.assertFalse(typo_payload["ok"])
        self.assertEqual(typo_payload["query"], "verfy")
        self.assertIn("verify", typo_payload["suggestions"])
        self.assertEqual(typo_json_err.getvalue(), "")

    def test_cli_completion_generates_shell_scripts_and_json_payloads(self):
        from mechferret.cli import main

        for shell, expected in (
            ("bash", "complete -F _mf_completion mf"),
            ("zsh", "compdef _mf_completion mf"),
            ("fish", "complete -c mf"),
        ):
            with self.subTest(shell=shell):
                out = StringIO()
                with redirect_stdout(out):
                    main(["completion", shell, "--command", "mf", "--json"])
                payload = json.loads(out.getvalue())
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["shell"], shell)
                self.assertEqual(payload["command"], "mf")
                self.assertIn(expected, payload["script"])
                self.assertIn("run", payload["script"])
                self.assertIn("completion", payload["script"])
                self.assertIn("bash", payload["script"])
                self.assertIn("zsh", payload["script"])
                self.assertIn("fish", payload["script"])
                self.assertIn("install_hint", payload)
                if shell in {"bash", "zsh"}:
                    self.assertIn("--json", payload["script"])
                if shell == "zsh":
                    self.assertIn("/completion", payload["script"])
                    self.assertIn("compadd -- $matches", payload["script"])
                    self.assertIn("'--command'", payload["script"])
                if shell == "fish":
                    self.assertIn("__fish_seen_subcommand_from completion /completion", payload["script"])
                    self.assertIn("-l json", payload["script"])
                    self.assertIn("-a bash", payload["script"])

        text_out = StringIO()
        with redirect_stdout(text_out):
            main(["completion", "bash"])
        rendered = text_out.getvalue()
        self.assertIn("complete -F _mechferret_completion mechferret", rendered)
        self.assertIn("--json", rendered)
        self.assertIn("bash zsh fish", rendered)

    def test_cli_registry_json_lists_filtered_items(self):
        from mechferret.cli import main

        out = StringIO()
        with redirect_stdout(out):
            main(["registry", "--kind", "tool", "--json"])
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["kind"], "tool")
        self.assertEqual(payload["count"], len(payload["items"]))
        self.assertTrue(payload["items"])
        self.assertTrue(all(item["kind"] == "tool" for item in payload["items"]))

    def test_cli_skills_json_lists_and_describes_playbooks(self):
        from mechferret.cli import main

        out = StringIO()
        with redirect_stdout(out):
            main(["skills", "--json"])
        listed = json.loads(out.getvalue())
        self.assertTrue(listed["ok"])
        self.assertEqual(listed["count"], len(listed["skills"]))
        self.assertTrue(any(skill["name"] == "ioi-circuit" for skill in listed["skills"]))

        detail_out = StringIO()
        with redirect_stdout(detail_out):
            main(["skills", "ioi-circuit", "--json"])
        detail = json.loads(detail_out.getvalue())
        self.assertTrue(detail["ok"])
        self.assertEqual(detail["skill"]["name"], "ioi-circuit")
        self.assertIn("budget", detail["skill"])
        self.assertIn("min_rigor", detail["skill"])

    def test_cli_api_json_redacts_keys_and_reports_updates(self):
        from mechferret.cli import main

        old_config = os.environ.get("MECHFERRET_CONFIG")
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            os.environ["MECHFERRET_CONFIG"] = str(config_path)
            try:
                show_out = StringIO()
                with redirect_stdout(show_out):
                    main(["api", "--show", "--json"])
                shown = json.loads(show_out.getvalue())
                self.assertTrue(shown["ok"])
                self.assertEqual(shown["default_provider"], "local")
                self.assertEqual(shown["providers"]["openai"]["key"], "missing")

                update_out = StringIO()
                with redirect_stdout(update_out):
                    main(["api", "--provider", "openai", "--api-key", "sk-test-secret", "--model", "gpt-test", "--json"])
                updated = json.loads(update_out.getvalue())
                self.assertTrue(updated["ok"])
                self.assertEqual(updated["action"], "update")
                self.assertEqual(updated["provider"], "openai")
                self.assertEqual(updated["default_provider"], "openai")
                self.assertEqual(updated["providers"]["openai"]["key"], "configured")
                self.assertNotIn("sk-test-secret", update_out.getvalue())

                clear_out = StringIO()
                with redirect_stdout(clear_out):
                    main(["api", "--clear", "openai", "--json"])
                cleared = json.loads(clear_out.getvalue())
                self.assertTrue(cleared["ok"])
                self.assertEqual(cleared["action"], "clear")
                self.assertEqual(cleared["default_provider"], "local")
                self.assertEqual(cleared["providers"]["openai"]["key"], "missing")

                bad_out = StringIO()
                with self.assertRaises(SystemExit) as ctx:
                    with redirect_stdout(bad_out):
                        main(["api", "--api-key", "sk-test-secret", "--json"])
                self.assertEqual(ctx.exception.code, 2)
                bad = json.loads(bad_out.getvalue())
                self.assertFalse(bad["ok"])
                self.assertNotIn("sk-test-secret", bad_out.getvalue())
            finally:
                if old_config is None:
                    os.environ.pop("MECHFERRET_CONFIG", None)
                else:
                    os.environ["MECHFERRET_CONFIG"] = old_config

    def test_cli_login_json_redacts_keys_and_reports_missing_key(self):
        from mechferret.cli import main

        old_config = os.environ.get("MECHFERRET_CONFIG")
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            os.environ["MECHFERRET_CONFIG"] = str(config_path)
            try:
                missing_out = StringIO()
                with self.assertRaises(SystemExit) as ctx:
                    with redirect_stdout(missing_out):
                        main(["login", "openai", "--json"])
                self.assertEqual(ctx.exception.code, 2)
                missing = json.loads(missing_out.getvalue())
                self.assertFalse(missing["ok"])
                self.assertEqual(missing["action"], "login")
                self.assertEqual(missing["provider"], "openai")
                self.assertIn("--api-key", missing["error"])

                login_out = StringIO()
                with redirect_stdout(login_out):
                    main(["login", "anthropic", "--api-key", "sk-test-secret", "--model", "claude-test", "--no-default", "--json"])
                payload = json.loads(login_out.getvalue())
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["action"], "login")
                self.assertEqual(payload["provider"], "anthropic")
                self.assertEqual(payload["default_provider"], "local")
                self.assertEqual(payload["providers"]["anthropic"]["key"], "configured")
                self.assertEqual(payload["providers"]["anthropic"]["model"], "claude-test")
                self.assertNotIn("sk-test-secret", login_out.getvalue())

                update_out = StringIO()
                with redirect_stdout(update_out):
                    main(["login", "openai", "--api-key", "sk-openai-secret", "--json"])
                updated = json.loads(update_out.getvalue())
                self.assertTrue(updated["ok"])
                self.assertEqual(updated["default_provider"], "openai")
                self.assertEqual(updated["providers"]["openai"]["key"], "configured")
                self.assertNotIn("sk-openai-secret", update_out.getvalue())
            finally:
                if old_config is None:
                    os.environ.pop("MECHFERRET_CONFIG", None)
                else:
                    os.environ["MECHFERRET_CONFIG"] = old_config

    def test_doctor_returns_checks(self):
        result = doctor()
        self.assertIn("ok", result)
        self.assertIn("checks", result)
        self.assertTrue(any(check["name"] == "example_corpus" for check in result["checks"]))
        names = {check["name"] for check in result["checks"]}
        self.assertIn("paper_generator", names)
        self.assertIn("openvla_sae_project", names)
        self.assertIn("latest_run_audit", names)
        self.assertIn("strict_passed", result)
        self.assertIn("all_integrations_passed", result)
        self.assertIn("next_actions", result)
        self.assertIn("optional_next_actions", result)
        self.assertIn("strict_next_actions", result)
        self.assertIn("all_integrations_next_actions", result)
        self.assertEqual(result["ok"], result["strict_passed"])
        self.assertEqual(result["passed"], result["strict_passed"])
        self.assertLessEqual(int(result["all_integrations_passed"]), int(result["strict_passed"]))

    def test_quickstart_returns_guided_command_sections(self):
        from mechferret.cli import main

        result = quickstart("all")
        self.assertIn("ok", result)
        self.assertEqual(result["ok"], result["doctor_strict_passed"])
        names = {section["name"] for section in result["sections"]}
        self.assertIn("local_demo", names)
        self.assertIn("openvla_sae", names)
        self.assertIn("release_gates", names)
        commands = "\n".join(command for section in result["sections"] for command in section["commands"])
        self.assertIn("mechferret init", commands)
        self.assertIn("mechferret doctor", commands)
        self.assertIn("mechferret sae openvla init", commands)
        demo_commands = next(section["commands"] for section in result["sections"] if section["name"] == "local_demo")
        self.assertEqual(
            demo_commands[:4],
            [
                "mechferret init",
                "mechferret quickstart --run",
                "mechferret status",
                "mechferret support",
            ],
        )
        self.assertLess(demo_commands.index("mechferret open report --select best --browser"), demo_commands.index("mechferret audit runs/demo/run.json --strict"))
        self.assertIn("mechferret verify runs/demo/run.json --strict", demo_commands)
        self.assertIn("doctor_all_integrations_passed", result)
        ci_commands = next(section["commands"] for section in result["sections"] if section["name"] == "release_gates")
        self.assertIn("mechferret quickstart --mode ci --run", ci_commands)
        self.assertIn("mechferret quickstart --mode demo --run", ci_commands)
        self.assertIn("mechferret audit runs/demo/run.json --strict", ci_commands)
        self.assertIn("mechferret verify runs/demo/run.json --strict", ci_commands)
        self.assertIn("mechferret bundle --select best", ci_commands)
        self.assertIn("mechferret verify-bundle --select best --strict", ci_commands)

        for args in (["doctor", "--json"], ["quickstart", "--mode", "demo", "--json"]):
            with self.subTest(args=args):
                out = StringIO()
                with redirect_stdout(out):
                    main(args)
                payload = json.loads(out.getvalue())
                self.assertIn("ok", payload)

    def test_selftest_reports_core_readiness_and_can_run_demo_path(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = selftest(
                runs_root=root / "runs",
                db_path=root / "memory.sqlite",
                notes_root=root,
                project_root=root / "openvla",
                report_path=root / "selftest.json",
            )
            self.assertTrue(result["ok"], result["next_actions"])
            self.assertEqual(result["command"], "selftest")
            self.assertEqual(result["mode"], "core")
            self.assertEqual(Path(result["artifacts"]["selftest_report"]), root / "selftest.json")
            self.assertEqual(json.loads((root / "selftest.json").read_text(encoding="utf-8"))["mode"], "core")
            step_names = {step["name"] for step in result["steps"]}
            self.assertIn("doctor_strict", step_names)
            self.assertIn("quickstart_guidance", step_names)
            self.assertIn("project_status", step_names)
            self.assertIn("local_demo", result["quickstart_sections"])
            self.assertIn("selftest --run", " ".join(result["next_actions"]))
            self.assertEqual(result["project_status"]["run_selection"], "best")
            self.assertIn("artifact_summary", result["project_status"])
            self.assertIn("artifact_readiness", result["project_status"])
            self.assertIn("readiness", result["project_status"])
            self.assertIn("readiness_summary", result["project_status"])
            result_summary = {item["name"]: item for item in result["project_status"]["readiness_summary"]}
            self.assertFalse(result_summary["setup"]["ready"])
            self.assertFalse(result_summary["run"]["ready"])
            self.assertFalse(result["project_status"]["run_ready"])
            self.assertFalse(result["project_status"]["share_ready"])
            self.assertIn("quickstart --run", " ".join(result["project_status"]["suggested_next_actions"]))

            json_out = StringIO()
            with redirect_stdout(json_out):
                main(
                    [
                        "selftest",
                        "--runs-root",
                        str(root / "runs"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--notes-root",
                        str(root),
                        "--project-root",
                        str(root / "openvla"),
                        "--report",
                        str(root / "cli-selftest.json"),
                        "--json",
                    ]
                )
            payload = json.loads(json_out.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["mode"], "core")
            self.assertEqual(Path(payload["artifacts"]["selftest_report"]), root / "cli-selftest.json")
            self.assertIn("readiness", payload["project_status"])
            self.assertIn("readiness_summary", payload["project_status"])
            self.assertIn("quickstart --run", " ".join(payload["project_status"]["suggested_next_actions"]))
            self.assertTrue((root / "cli-selftest.json").exists())

            support_out = StringIO()
            with redirect_stdout(support_out):
                main(
                    [
                        "support",
                        "--runs-root",
                        str(root / "runs"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--notes-root",
                        str(root),
                        "--project-root",
                        str(root / "openvla"),
                        "--report",
                        str(root / "support.json"),
                        "--json",
                    ]
                )
            support_payload = json.loads(support_out.getvalue())
            self.assertTrue(support_payload["ok"])
            self.assertEqual(support_payload["schema_version"], 1)
            self.assertEqual(support_payload["command"], "support")
            self.assertEqual(support_payload["project_status"]["run_selection"], "best")
            self.assertIn("artifact_summary", support_payload["project_status"])
            self.assertIn("readiness", support_payload["project_status"])
            self.assertIn("readiness_summary", support_payload["project_status"])
            support_summary = {item["name"]: item for item in support_payload["project_status"]["readiness_summary"]}
            self.assertIn("project_notes", support_summary["setup"]["reason"])
            self.assertEqual(support_payload["report"]["kind"], "support")
            self.assertTrue(support_payload["report"]["shareable"])
            self.assertEqual(support_payload["report"]["privacy"]["credential_values"], "omitted")
            self.assertEqual(Path(support_payload["artifacts"]["selftest_report"]), root / "support.json")
            self.assertIn("quickstart --run", " ".join(support_payload["project_status"]["suggested_next_actions"]))
            self.assertTrue((root / "support.json").exists())

            plain_out = StringIO()
            with redirect_stdout(plain_out):
                main(
                    [
                        "selftest",
                        "--runs-root",
                        str(root / "runs"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--notes-root",
                        str(root),
                        "--project-root",
                        str(root / "openvla"),
                    ]
                )
            plain_text = plain_out.getvalue()
            self.assertIn("Status selection: best", plain_text)
            self.assertIn("Readiness lanes:", plain_text)
            self.assertIn("setup: BLOCKED", plain_text)
            self.assertIn("Project status next actions:", plain_text)
            self.assertIn("Suggested next actions:", plain_text)
            self.assertIn("quickstart --run", plain_text)

            run_result = selftest(
                run=True,
                out_dir=root / "runs" / "selftest",
                db_path=root / "selftest.sqlite",
                runs_root=root / "runs",
                notes_root=root,
                project_root=root / "openvla",
                report_path=root / "runs" / "selftest" / "selftest.json",
            )
            self.assertTrue(run_result["ok"], run_result["next_actions"])
            self.assertEqual(run_result["mode"], "demo")
            self.assertTrue(Path(run_result["artifacts"]["run_json"]).exists())
            self.assertTrue(Path(run_result["artifacts"]["selftest_report"]).exists())
            self.assertTrue(run_result["verification"]["passed"])
            run_steps = {step["name"] for step in run_result["steps"]}
            self.assertIn("demo_quickstart", run_steps)
            self.assertIn("verify_manifest", run_steps)

    def test_support_report_omits_configured_credentials(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret_openai = "sk-test-support-openai-secret"
            secret_anthropic = "sk-test-support-anthropic-secret"
            env = {
                "MECHFERRET_CONFIG": str(root / "config.json"),
                "HOME": str(root),
                "OPENAI_API_KEY": secret_openai,
                "ANTHROPIC_API_KEY": secret_anthropic,
            }
            run_dir = root / "runs" / "leaky"
            run_dir.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": "leaky",
                        "question": f"Can this summary leak {secret_openai} or sk-local-run-token123?",
                        "metrics": {"readiness_score": 0.4},
                        "claims": [],
                        "evidence": [],
                        "gaps": [f"Remove {secret_anthropic} from diagnostics."],
                        "artifacts": {"trace": f"contains {secret_openai}"},
                    }
                ),
                encoding="utf-8",
            )
            out = StringIO()
            with patch.dict(os.environ, env, clear=False), redirect_stdout(out):
                main(
                    [
                        "support",
                        "--runs-root",
                        str(root / "runs"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--notes-root",
                        str(root),
                        "--project-root",
                        str(root / "openvla"),
                        "--report",
                        str(root / "support.json"),
                        "--json",
                    ]
                )
            report_text = (root / "support.json").read_text(encoding="utf-8")
            combined = out.getvalue() + report_text
            self.assertNotIn(secret_openai, combined)
            self.assertNotIn(secret_anthropic, combined)
            self.assertNotIn("sk-local-run-token123", combined)
            self.assertNotIn(str(root), combined)
            self.assertIn("[redacted]", combined)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["command"], "support")
            self.assertEqual(payload["report"]["privacy"]["credential_values"], "omitted")
            self.assertTrue(payload["report"]["redaction"]["applied"])
            self.assertGreaterEqual(payload["report"]["redaction"]["configured_values"], 3)
            self.assertGreaterEqual(payload["report"]["redaction"]["credential_patterns"], 1)
            self.assertEqual(
                payload["report"]["redaction"]["total"],
                payload["report"]["redaction"]["field_values"]
                + payload["report"]["redaction"]["configured_values"]
                + payload["report"]["redaction"]["credential_patterns"],
            )
            self.assertIn("[redacted]", payload["project_status"]["selected_run"]["summary"]["question"])
            self.assertEqual(payload["artifacts"]["selftest_report"], "~/support.json")
            report_payload = json.loads(report_text)
            self.assertEqual(report_payload["command"], "support")
            self.assertEqual(report_payload["report"]["redaction"], payload["report"]["redaction"])

            plain_out = StringIO()
            with patch.dict(os.environ, env, clear=False), redirect_stdout(plain_out):
                main(
                    [
                        "support",
                        "--runs-root",
                        str(root / "runs"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--notes-root",
                        str(root),
                        "--project-root",
                        str(root / "openvla"),
                        "--report",
                        str(root / "plain-support.json"),
                    ]
                )
            plain_text = plain_out.getvalue() + (root / "plain-support.json").read_text(encoding="utf-8")
            self.assertIn("Support report: PASS", plain_text)
            self.assertIn("Redaction: applied", plain_text)
            self.assertNotIn(secret_openai, plain_text)
            self.assertNotIn(secret_anthropic, plain_text)
            self.assertNotIn(str(root), plain_text)

    def test_runs_json_handles_empty_root(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out = StringIO()
            with redirect_stdout(out):
                main(["runs", "--runs-root", str(Path(tmp) / "runs"), "--json"])
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["count"], 0)
            self.assertEqual(payload["selected_path"], "")
            self.assertTrue(payload["next_actions"])

    def test_init_project_notes_creates_agent_context_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = init_project_notes(root)
            path = root / "MECHFERRET.md"
            self.assertTrue(result["ok"])
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("MechFerret Project Notes", text)
            self.assertIn("Paper Acceptance Bar", text)

            blocked = init_project_notes(root)
            self.assertTrue(blocked["ok"])
            self.assertFalse(blocked["created"])
            self.assertIn("--force", blocked["next_actions"][0])

            forced = init_project_notes(root, force=True)
            self.assertTrue(forced["ok"])

    def test_run_quickstart_demo_creates_passing_local_dossier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_quickstart("all", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            self.assertTrue(result["ok"])
            self.assertEqual(result["mode"], "demo")
            self.assertTrue((root / "runs" / "demo" / "run.json").exists())
            self.assertTrue((root / "MECHFERRET.md").exists())
            self.assertEqual(Path(result["project_notes"]["path"]), root / "MECHFERRET.md")
            self.assertTrue(result["project_notes"]["created"])
            suggestions = "\n".join(result["suggested_next_actions"])
            self.assertIn("mechferret status", suggestions)
            self.assertIn("mechferret support", suggestions)
            self.assertIn("mechferret open report --select best --browser", suggestions)
            self.assertIn("mechferret bundle --select best", suggestions)
            self.assertTrue(Path(result["artifacts"]["paper"]).exists())
            self.assertEqual(Path(result["artifacts"]["project_notes"]), root / "MECHFERRET.md")
            self.assertTrue((root / "runs" / "demo" / "quickstart.json").exists())
            quickstart_payload = json.loads((root / "runs" / "demo" / "quickstart.json").read_text(encoding="utf-8"))
            self.assertTrue(quickstart_payload["project_notes"]["created"])
            self.assertEqual(quickstart_payload["suggested_next_actions"], result["suggested_next_actions"])
            self.assertTrue((root / "runs" / "demo" / "QUICKSTART.md").exists())
            self.assertTrue(result["audit"]["passed"])
            self.assertIn("project_notes", {step["name"] for step in result["steps"]})
            text = (root / "runs" / "demo" / "QUICKSTART.md").read_text(encoding="utf-8")
            self.assertIn("MechFerret Quickstart Run", text)
            self.assertIn("## Project Notes", text)
            self.assertIn("created:", text)
            self.assertIn("## Suggested Next Actions", text)
            self.assertIn("mechferret status", text)
            self.assertIn("project_notes", text)
            self.assertIn("local demo quickstart audit passed", text)
            from mechferret.ops import print_quickstart_run

            quickstart_out = StringIO()
            with redirect_stdout(quickstart_out):
                print_quickstart_run(result)
            self.assertIn("Suggested next actions:", quickstart_out.getvalue())
            self.assertIn("mechferret support", quickstart_out.getvalue())
            resolved = resolve_artifact("quickstart", runs_root=root / "runs")
            self.assertEqual(resolved["ok"], resolved["exists"])
            self.assertTrue(resolved["exists"])
            self.assertEqual(Path(resolved["path"]), root / "runs" / "demo" / "QUICKSTART.md")
            self.assertTrue(resolve_artifact("report", runs_root=root / "runs")["exists"])
            paper = resolve_artifact("paper", runs_root=root / "runs")
            self.assertEqual(paper["ok"], paper["exists"])
            self.assertTrue(paper["exists"])
            self.assertEqual(Path(paper["path"]), root / "runs" / "demo" / "paper" / "main.tex")
            self.assertTrue(resolve_artifact("run", runs_root=root / "runs")["exists"])
            index = resolve_artifact("all", runs_root=root / "runs", project_root=root / "openvla")
            self.assertEqual(index["ok"], index["exists"])
            self.assertTrue(index["exists"])
            self.assertFalse(index["complete"])
            self.assertFalse(index["share_ready"])
            self.assertFalse(index["setup_ready"])
            self.assertIn("quickstart", index["artifacts"])
            self.assertTrue(index["artifacts"]["quickstart"]["exists"])
            self.assertTrue(index["artifacts"]["report"]["exists"])
            self.assertTrue(index["artifacts"]["paper"]["exists"])
            self.assertIn("review", index["artifacts"])
            self.assertFalse(index["artifacts"]["review"]["exists"])
            self.assertIn("pdf", index["artifacts"])
            self.assertFalse(index["artifacts"]["pdf"]["exists"])
            self.assertTrue(index["artifacts"]["markdown"]["exists"])

            self.assertTrue(index["artifacts"]["graph"]["exists"])
            self.assertTrue(index["artifacts"]["evals"]["exists"])
            self.assertTrue(index["artifacts"]["trace"]["exists"])
            self.assertTrue(index["artifacts"]["manifest"]["exists"])
            self.assertEqual(Path(resolve_artifact("markdown", runs_root=root / "runs")["path"]), root / "runs" / "demo" / "report.md")
            self.assertEqual(Path(resolve_artifact("graph", runs_root=root / "runs")["path"]), root / "runs" / "demo" / "graph.json")
            self.assertEqual(Path(resolve_artifact("evals", runs_root=root / "runs")["path"]), root / "runs" / "demo" / "evals.json")
            self.assertEqual(Path(resolve_artifact("trace", runs_root=root / "runs")["path"]), root / "runs" / "demo" / "trace.jsonl")
            self.assertEqual(Path(resolve_artifact("html", runs_root=root / "runs")["path"]), root / "runs" / "demo" / "report.html")
            self.assertEqual(Path(resolve_artifact("manifest", runs_root=root / "runs")["path"]), root / "runs" / "demo" / "manifest.json")
            from mechferret.cli import main

            open_out = StringIO()
            with redirect_stdout(open_out):
                main(["open", "run", "--runs-root", str(root / "runs"), "--json"])
            open_payload = json.loads(open_out.getvalue())
            self.assertTrue(open_payload["ok"])
            self.assertTrue(open_payload["exists"])

            index_out = StringIO()
            with redirect_stdout(index_out):
                main(["open", "all", "--runs-root", str(root / "runs"), "--project-root", str(root / "openvla"), "--json"])
            index_payload = json.loads(index_out.getvalue())
            self.assertTrue(index_payload["ok"])
            self.assertFalse(index_payload["complete"])
            self.assertFalse(index_payload["share_ready"])
            self.assertEqual(index_payload["artifacts"]["report"]["ok"], index_payload["artifacts"]["report"]["exists"])
            self.assertEqual(index_payload["artifacts"]["bundle"]["ok"], index_payload["artifacts"]["bundle"]["exists"])
            run_payload = json.loads((root / "runs" / "demo" / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(run_payload["artifacts"]["paper"]), root / "runs" / "demo" / "paper" / "main.tex")
            self.assertEqual(Path(run_payload["artifacts"]["manifest"]), root / "runs" / "demo" / "manifest.json")
            self.assertTrue(verify_run_artifacts(root / "runs" / "demo" / "run.json")["passed"])
            self.assertIn("bundle", index["artifacts"])
            self.assertFalse(index["artifacts"]["bundle"]["exists"])
            index_actions = " ".join(index["next_actions"])
            self.assertIn("mechferret review-paper --select latest", index_actions)
            self.assertIn("mechferret paper --select latest --compile", index_actions)
            self.assertIn("mechferret bundle --select latest", index_actions)
            self.assertNotIn("generate a run-bound draft", index_actions)
            self.assertLess(index_actions.index("review-paper"), index_actions.index("bundle --select latest"))
            self.assertLess(index_actions.index("review-paper"), index_actions.index("quickstart --mode ci --run"))
            self.assertIn(
                "mechferret bundle --select latest",
                " ".join(index["artifacts"]["bundle"]["next_actions"]),
            )
            missing = resolve_artifact("missing-target", runs_root=root / "runs")
            self.assertEqual(missing["ok"], missing["exists"])
            self.assertFalse(missing["exists"])
            self.assertEqual(missing["reason"], "explicit path")
            self.assertIn("mechferret open all", " ".join(missing["next_actions"]))
            missing_out = StringIO()
            with self.assertRaises(SystemExit) as ctx:
                with redirect_stdout(missing_out):
                    main(["open", str(root / "missing-target"), "--runs-root", str(root / "runs"), "--json"])
            self.assertEqual(ctx.exception.code, 1)
            missing_payload = json.loads(missing_out.getvalue())
            self.assertFalse(missing_payload["ok"])
            self.assertFalse(missing_payload["exists"])
            empty_index = resolve_artifact("all", runs_root=root / "missing-runs", project_root=root / "missing-openvla")
            self.assertFalse(empty_index["exists"])
            self.assertFalse(empty_index["complete"])
            self.assertFalse(empty_index["run_ready"])
            self.assertFalse(empty_index["share_ready"])
            self.assertFalse(empty_index["setup_ready"])
            self.assertFalse(empty_index["artifacts"]["paper"]["exists"])
            self.assertNotIn("mechferret paper", " ".join(empty_index["next_actions"]))

            Path(run_payload["artifacts"]["paper"]).unlink()
            missing_index = resolve_artifact("all", runs_root=root / "runs", project_root=root / "openvla")
            missing_index_actions = " ".join(missing_index["next_actions"])
            self.assertIn("mechferret paper --select latest", missing_index_actions)
            self.assertNotIn("review-paper", missing_index_actions)
            self.assertNotIn("bundle --select latest", missing_index_actions)
            missing_paper = resolve_artifact("paper", runs_root=root / "runs")
            self.assertFalse(missing_paper["exists"])
            self.assertIn("mechferret paper --select latest", " ".join(missing_paper["next_actions"]))
            self.assertNotIn("quickstart --run", " ".join(missing_paper["next_actions"]))
            missing_review = resolve_artifact("review", runs_root=root / "runs")
            self.assertIn("mechferret review-paper --select latest", " ".join(missing_review["next_actions"]))

            (root / "paper").mkdir()
            (root / "paper" / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
            no_run_paper = resolve_artifact("paper", runs_root=root / "missing-runs")
            self.assertFalse(no_run_paper["exists"])

    def test_run_quickstart_demo_preserves_existing_project_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes = root / "MECHFERRET.md"
            notes.write_text("# Custom Notes\n", encoding="utf-8")
            result = run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            self.assertTrue(result["ok"], result["next_actions"])
            self.assertEqual(notes.read_text(encoding="utf-8"), "# Custom Notes\n")
            self.assertFalse(result["project_notes"]["created"])
            self.assertEqual(Path(result["artifacts"]["project_notes"]), notes)
            text = (root / "runs" / "demo" / "QUICKSTART.md").read_text(encoding="utf-8")
            self.assertIn("preserved:", text)

    def test_project_status_summarizes_latest_run_and_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")

            status = project_status(
                runs_root=root / "runs",
                db_path=root / "memory.sqlite",
                notes_root=root,
                project_root=root / "openvla",
            )
            self.assertEqual(status["state"], "ready")
            self.assertEqual(status["project_notes"]["ok"], status["project_notes"]["exists"])
            self.assertEqual(status["doctor"]["ok"], status["doctor"]["strict_passed"])
            self.assertTrue(status["latest_run"]["exists"])
            self.assertEqual(status["latest_run"]["ok"], status["latest_run"]["exists"])
            self.assertEqual(Path(status["latest_run"]["path"]), root / "runs" / "demo" / "run.json")
            self.assertTrue(status["audit"]["passed"])
            self.assertTrue(status["verification"]["passed"])
            self.assertTrue(status["advisories"])
            self.assertIn("local_synthesis_not_final", {item["name"] for item in status["advisories"]})
            self.assertIn("paper", status["available_artifacts"])
            self.assertIn("review", status["missing_artifacts"])
            self.assertEqual(status["artifacts"]["paper"]["ok"], status["artifacts"]["paper"]["exists"])
            self.assertEqual(status["artifacts"]["review"]["ok"], status["artifacts"]["review"]["exists"])
            self.assertEqual(status["artifact_summary"]["total"], len(status["artifacts"]))
            self.assertEqual(status["artifact_summary"]["found"], len(status["available_artifacts"]))
            self.assertEqual(status["artifact_summary"]["missing"], len(status["missing_artifacts"]))
            self.assertEqual(status["artifact_summary"]["groups"]["run"]["total"], 13)
            self.assertEqual(status["artifact_summary"]["groups"]["setup"]["total"], 3)
            self.assertIn("artifact_readiness", status)
            self.assertFalse(status["artifact_readiness"]["sharing"]["ok"])
            self.assertIn("review", status["artifact_readiness"]["sharing"]["missing_artifacts"])
            self.assertTrue(status["readiness"]["project"]["ok"])
            self.assertTrue(status["readiness"]["selected_run"]["ok"])
            self.assertFalse(status["readiness"]["sharing"]["ok"])
            self.assertFalse(status["share_ready"])
            self.assertIn("review", status["readiness"]["sharing"]["missing_artifacts"])
            summary_by_name = {item["name"]: item for item in status["readiness_summary"]}
            self.assertFalse(summary_by_name["setup"]["ready"])
            self.assertTrue(summary_by_name["run"]["ready"])
            self.assertFalse(summary_by_name["sharing"]["ready"])
            self.assertIn("openvla", summary_by_name["setup"]["reason"])
            self.assertEqual(summary_by_name["sharing"]["status"], "blocked")
            self.assertIn("review", summary_by_name["sharing"]["reason"])
            self.assertEqual(status["artifact_summary"]["groups"]["dossier"]["missing"], 0)
            self.assertIn("experiments", status["artifact_summary"]["groups"]["discovery"]["missing_artifacts"])
            self.assertIn("review", status["artifact_summary"]["groups"]["sharing"]["missing_artifacts"])
            self.assertIn("mechferret review-paper --select latest", " ".join(status["next_actions"]))
            suggestions = "\n".join(status["suggested_next_actions"])
            self.assertIn("mechferret open report --select latest --browser", suggestions)
            self.assertIn("mechferret bundle --select latest", suggestions)
            self.assertIn("mechferret review-paper --select latest", suggestions)
            self.assertGreaterEqual(status["memory"]["runs"], 1)

            status_out = StringIO()
            with redirect_stdout(status_out):
                print_project_status(status)
            self.assertIn("Suggested next actions:", status_out.getvalue())
            self.assertIn("Artifact summary:", status_out.getvalue())
            self.assertIn("Run artifacts:", status_out.getvalue())
            self.assertIn("Setup artifacts:", status_out.getvalue())
            self.assertIn("dossier 5/5", status_out.getvalue())
            self.assertIn("Run readiness: READY", status_out.getvalue())
            self.assertIn("Share readiness: BLOCKED", status_out.getvalue())
            self.assertIn("Setup readiness: BLOCKED", status_out.getvalue())
            self.assertIn("mechferret open report --select latest --browser", status_out.getvalue())

    def test_project_status_keeps_publish_guidance_when_only_setup_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            missing_notes_root = root / "missing-notes"

            status = project_status(
                runs_root=root / "runs",
                db_path=root / "memory.sqlite",
                notes_root=missing_notes_root,
                project_root=root / "openvla",
                selection="best",
            )

            self.assertEqual(status["state"], "needs_setup")
            self.assertTrue(status["run_ready"])
            self.assertTrue(status["audit"]["passed"])
            self.assertTrue(status["verification"]["passed"])
            actions = "\n".join(status["next_actions"])
            self.assertIn("mechferret init", actions)
            self.assertIn("mechferret bundle --select best", actions)
            self.assertIn("mechferret review-paper --select best", actions)
            self.assertIn("configured provider", actions)
            self.assertLess(actions.index("review-paper --select best"), actions.index("quickstart --mode ci --run"))
            suggestions = "\n".join(status["suggested_next_actions"])
            self.assertIn("mechferret init", suggestions)
            self.assertIn("mechferret bundle --select best", suggestions)
            self.assertIn("mechferret review-paper --select best", suggestions)

            status_out = StringIO()
            with redirect_stdout(status_out):
                print_project_status(status)
            self.assertIn("Project status: needs_setup", status_out.getvalue())
            self.assertIn("Run readiness: READY", status_out.getvalue())

    def test_relative_run_artifact_paths_resolve_from_other_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "other"
            other.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(root)
                run_quickstart("demo", out_dir=Path("runs") / "demo", db_path=root / "memory.sqlite")
                run_json = root / "runs" / "demo" / "run.json"
                payload = json.loads(run_json.read_text(encoding="utf-8"))
                self.assertFalse(Path(payload["artifacts"]["markdown"]).is_absolute())
                manifest_path = root / "runs" / "demo" / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["artifacts"]["markdown"]["path"] = str(root / "runs" / "demo" / "report.md")
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

                os.chdir(other)
                shadow_report = other / "runs" / "demo" / "report.html"
                shadow_report.parent.mkdir(parents=True)
                shadow_report.write_text("<p>wrong report</p>", encoding="utf-8")
                (other / "runs" / "demo" / "report.md").write_text("wrong markdown\n", encoding="utf-8")
                verification = verify_run_artifacts(run_json)
                self.assertTrue(verification["passed"], verification["failed_checks"])
                report = resolve_artifact("report", runs_root=root / "runs")
                self.assertTrue(report["exists"])
                self.assertEqual(Path(report["path"]), root / "runs" / "demo" / "report.html")
                status = project_status(runs_root=root / "runs", db_path=root / "memory.sqlite", notes_root=root)
                self.assertTrue(status["verification"]["passed"])
            finally:
                os.chdir(previous)

    def test_project_status_warns_on_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            (root / "runs" / "demo" / "report.md").write_text("tampered\n", encoding="utf-8")

            status = project_status(
                runs_root=root / "runs",
                db_path=root / "memory.sqlite",
                notes_root=root,
                project_root=root / "openvla",
            )
            self.assertEqual(status["state"], "needs_attention")
            self.assertFalse(status["verification"]["passed"])
            self.assertIn("artifact_sha256:markdown", status["verification"]["failed_checks"])
            self.assertIn("Regenerate", " ".join(status["next_actions"]))
            self.assertIn("mechferret verify --select latest", " ".join(status["next_actions"]))
            actions = "\n".join(status["next_actions"])
            self.assertNotIn("mechferret bundle --select latest", actions)
            self.assertNotIn("mechferret review-paper --select latest", actions)
            self.assertNotIn("mechferret paper --select latest --compile", actions)
            suggestions = "\n".join(status["suggested_next_actions"])
            self.assertIn("mechferret open report --select latest --browser", suggestions)
            self.assertNotIn("mechferret bundle --select latest", suggestions)
            self.assertNotIn("mechferret review-paper --select latest", suggestions)

    def test_project_status_collapses_duplicate_paper_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            (root / "runs" / "demo" / "paper" / "main.tex").unlink()

            status = project_status(
                runs_root=root / "runs",
                db_path=root / "memory.sqlite",
                notes_root=root,
                project_root=root / "openvla",
                selection="best",
            )
            actions = "\n".join(status["next_actions"])
            self.assertIn("mechferret paper --select best --provider local", actions)
            self.assertIn("paper/main.tex", actions)
            self.assertNotIn("mechferret paper --provider local", actions)
            self.assertNotIn("generate a run-bound draft", actions)

    def test_project_status_warns_on_bundle_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
            entries["audit.json"] = b'{"passed": true}\n'
            with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(runs_root=root / "runs", selection="best")
            self.assertFalse(verification["passed"])
            self.assertIn("mechferret bundle --select best", " ".join(verification["next_actions"]))

            status = project_status(
                runs_root=root / "runs",
                db_path=root / "memory.sqlite",
                notes_root=root,
                project_root=root / "openvla",
            )
            self.assertEqual(status["state"], "needs_attention")
            self.assertFalse(status["bundle_verification"]["passed"])
            self.assertIn("bundle_metadata_sha256:audit", status["bundle_verification"]["failed_checks"])
            self.assertIn("Recreate the bundle", " ".join(status["next_actions"]))
            self.assertIn("mechferret bundle --select latest", " ".join(status["next_actions"]))

    def test_project_status_keeps_run_ready_when_only_bundle_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
            entries["audit.json"] = b'{"passed": true}\n'
            with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)
            run_json = root / "runs" / "demo" / "run.json"
            refresh_run_manifest(run_json)

            status = project_status(
                runs_root=root / "runs",
                db_path=root / "memory.sqlite",
                notes_root=root,
                project_root=root / "openvla",
            )
            self.assertEqual(status["state"], "needs_attention")
            self.assertTrue(status["run_ready"])
            self.assertFalse(status["share_ready"])
            self.assertTrue(status["readiness"]["selected_run"]["ok"])
            self.assertFalse(status["readiness"]["sharing"]["ok"])
            self.assertFalse(status["readiness"]["sharing"]["bundle_verified"])
            self.assertTrue(status["audit"]["passed"])
            self.assertTrue(status["verification"]["passed"])
            self.assertFalse(status["bundle_verification"]["passed"])
            self.assertIn("Recreate the bundle", " ".join(status["next_actions"]))
            self.assertIn("mechferret bundle --select latest", " ".join(status["next_actions"]))

            status_out = StringIO()
            with redirect_stdout(status_out):
                print_project_status(status)
            self.assertIn("Run readiness: READY", status_out.getvalue())
            self.assertIn("Share readiness: BLOCKED", status_out.getvalue())
            self.assertIn("Bundle verify: WARN", status_out.getvalue())

    def test_project_status_guides_empty_project(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = project_status(
                runs_root=root / "runs",
                db_path=root / "memory.sqlite",
                notes_root=root,
                project_root=root / "openvla",
            )
            self.assertEqual(status["state"], "needs_setup")
            self.assertFalse(status["latest_run"]["exists"])
            summary_by_name = {item["name"]: item for item in status["readiness_summary"]}
            self.assertFalse(summary_by_name["setup"]["ready"])
            self.assertFalse(summary_by_name["run"]["ready"])
            self.assertFalse(summary_by_name["sharing"]["ready"])
            self.assertIn("project_notes", summary_by_name["setup"]["reason"])
            self.assertIn("mechferret init", " ".join(status["next_actions"]))
            self.assertIn("quickstart", " ".join(status["next_actions"]))
            suggestions = " ".join(status["suggested_next_actions"])
            self.assertIn("mechferret init", suggestions)
            self.assertIn("mechferret quickstart --run", suggestions)

            status_out = StringIO()
            with redirect_stdout(status_out):
                print_project_status(status)
            self.assertIn("Run readiness: BLOCKED", status_out.getvalue())
            self.assertIn("Share readiness: BLOCKED", status_out.getvalue())
            self.assertIn("Setup readiness: BLOCKED", status_out.getvalue())

            next_out = StringIO()
            with redirect_stdout(next_out):
                main(
                    [
                        "next",
                        "--runs-root",
                        str(root / "runs"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--notes-root",
                        str(root),
                        "--project-root",
                        str(root / "openvla"),
                        "--json",
                    ]
                )
            payload = json.loads(next_out.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["command"], "next")
            self.assertEqual(payload["state"], "needs_setup")
            self.assertFalse(payload["run_ready"])
            self.assertFalse(payload["share_ready"])
            self.assertEqual(payload["actions"], [item["action"] for item in payload["action_plan"]])
            self.assertEqual(payload["action_plan"][0]["category"], "required")
            self.assertTrue(payload["action_plan"][0]["required"])
            self.assertIn("reason", payload["action_plan"][0])
            next_summary = {item["name"]: item for item in payload["readiness_summary"]}
            self.assertFalse(next_summary["setup"]["ready"])
            self.assertFalse(next_summary["run"]["ready"])
            self.assertIn("mechferret init", " ".join(payload["actions"]))
            self.assertIn("mechferret quickstart --run", " ".join(payload["actions"]))

    def test_next_command_summarizes_ready_project_actions(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")

            json_out = StringIO()
            with redirect_stdout(json_out):
                main(
                    [
                        "next",
                        "--runs-root",
                        str(root / "runs"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--notes-root",
                        str(root),
                        "--project-root",
                        str(root / "openvla"),
                        "--select",
                        "best",
                        "--limit",
                        "2",
                        "--json",
                    ]
                )
            payload = json.loads(json_out.getvalue())
            self.assertEqual(payload["run_selection"], "best")
            self.assertTrue(payload["selected_run"]["exists"])
            self.assertEqual(len(payload["actions"]), 2)
            self.assertEqual(len(payload["action_plan"]), 2)
            self.assertEqual(payload["actions"], [item["action"] for item in payload["action_plan"]])
            self.assertTrue(all(item["reason"] for item in payload["action_plan"]))
            summary_by_name = {item["name"]: item for item in payload["readiness_summary"]}
            self.assertTrue(summary_by_name["run"]["ready"])
            self.assertFalse(summary_by_name["setup"]["ready"])
            self.assertTrue(any("review-paper --select best" in action for action in payload["actions"]))

            text_out = StringIO()
            with redirect_stdout(text_out):
                main(
                    [
                        "next",
                        "--runs-root",
                        str(root / "runs"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--notes-root",
                        str(root),
                        "--project-root",
                        str(root / "openvla"),
                        "--select",
                        "best",
                        "--limit",
                        "1",
                    ]
                )
            rendered = text_out.getvalue()
            self.assertIn("Project state:", rendered)
            self.assertIn("Run readiness: READY", rendered)
            self.assertIn("Setup readiness: BLOCKED", rendered)
            self.assertIn("Next actions:", rendered)
            self.assertIn("reason:", rendered)

    def test_list_run_artifacts_orders_recent_runs_and_reports_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "one", db_path=root / "memory.sqlite")
            run_quickstart("demo", out_dir=root / "runs" / "two", db_path=root / "memory.sqlite")
            older = root / "runs" / "one" / "run.json"
            newer = root / "runs" / "two" / "run.json"
            os.utime(older, (1_700_000_000, 1_700_000_000))
            os.utime(newer, (1_700_001_000, 1_700_001_000))

            result = list_run_artifacts(runs_root=root / "runs", limit=2)
            self.assertEqual(result["count"], 2)
            self.assertEqual(result["shown"], 2)
            self.assertEqual(Path(result["runs"][0]["path"]), newer)
            self.assertEqual(Path(result["runs"][1]["path"]), older)
            self.assertEqual(Path(result["selected_path"]), newer)
            self.assertEqual(result["selected_rank"], 1)
            self.assertTrue(result["selected_visible"])
            self.assertTrue(result["runs"][0]["selected"])
            self.assertEqual(result["runs"][0]["selection"], "best")
            self.assertTrue(result["runs"][0]["audit"]["passed"])
            self.assertTrue(result["runs"][0]["artifacts"]["report"])
            self.assertTrue(result["runs"][0]["artifacts"]["paper"])
            self.assertTrue(result["runs"][0]["artifacts"]["graph"])
            self.assertTrue(result["runs"][0]["artifacts"]["evals"])
            self.assertFalse(result["runs"][0]["artifacts"]["bundle"])
            self.assertIn("artifact_summary", result["runs"][0])
            self.assertIn("artifact_readiness", result["runs"][0])
            self.assertFalse(result["runs"][0]["artifact_readiness"]["run"]["ok"])
            self.assertFalse(result["runs"][0]["artifact_readiness"]["sharing"]["ok"])
            self.assertFalse(result["runs"][0]["artifact_readiness"]["setup"]["ok"])

            limited = list_run_artifacts(runs_root=root / "runs", limit=1, include_audit=False)
            self.assertEqual(limited["shown"], 1)
            self.assertTrue(limited["selected_visible"])
            self.assertNotIn("audit", limited["runs"][0])

            out = StringIO()
            with redirect_stdout(out):
                print_run_list(result)
            self.assertIn("[selected: best]", out.getvalue())
            self.assertIn("lanes: run=BLOCKED share=BLOCKED setup=BLOCKED", out.getvalue())

    def test_list_run_artifacts_reports_ready_selection_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "broken", db_path=root / "memory.sqlite")
            report = root / "runs" / "broken" / "report.md"
            report.write_text("tampered\n", encoding="utf-8")

            result = list_run_artifacts(runs_root=root / "runs", selection="ready")
            self.assertEqual(result["count"], 1)
            self.assertIsNone(result["selected"])
            self.assertEqual(result["selected_path"], "")
            self.assertIsNotNone(result["selection_failure"])
            self.assertEqual(Path(result["selection_failure"]["nearest_path"]), root / "runs" / "broken" / "run.json")
            self.assertIn("manifest_integrity", result["selection_failure"]["failed_checks"])
            actions = " ".join(result["next_actions"])
            self.assertIn("No audit-passing run found", actions)
            self.assertIn("manifest_integrity", actions)
            out = StringIO()
            with redirect_stdout(out):
                print_run_list(result)
            self.assertIn("Closest run:", out.getvalue())
            self.assertIn("Blocking checks: manifest_integrity", out.getvalue())

    def test_ops_public_helpers_tolerate_malformed_inputs_and_run_rows(self):
        rows = [
            "bad",
            {
                "ok": True,
                "path": "/tmp/good/run.json",
                "mtime": "bad",
                "readiness_score": "bad",
                "audit": {"passed": True, "readiness_score": float("nan"), "advisories": ["bad"]},
                "artifacts": ["bad"],
            },
        ]
        selected = select_run_artifact(policy=["bad"], _rows=rows)  # type: ignore[arg-type]
        self.assertEqual(selected["policy"], "latest")
        self.assertEqual(selected["path"], "/tmp/good/run.json")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_run = root / "runs" / "bad" / "run.json"
            bad_run.parent.mkdir(parents=True)
            bad_run.write_text("[]", encoding="utf-8")

            listed = list_run_artifacts(runs_root=root / "runs", limit="bad", include_audit=False, selection=["bad"])  # type: ignore[arg-type]
            self.assertTrue(listed["ok"])
            self.assertEqual(listed["count"], 1)
            self.assertEqual(listed["selection"], "best")
            self.assertFalse(listed["runs"][0]["ok"])

            status = project_status(runs_root=root / "runs", db_path=object(), notes_root=root, selection=object())  # type: ignore[arg-type]
            self.assertEqual(status["run_selection"], "latest")
            self.assertEqual(status["selected_run"]["summary"]["run_id"], "")

            resolved = resolve_artifact(["bad"], runs_root=root / "runs", project_root=object(), selection=object())  # type: ignore[arg-type]
            self.assertEqual(resolved["target"], "quickstart")
            self.assertFalse(resolved["exists"])

            explicit = resolve_artifact(str(root / "missing.html"), runs_root=root / "runs")
            self.assertFalse(explicit["exists"])
            self.assertEqual(explicit["reason"], "explicit path")
            self.assertIn("mechferret open all", " ".join(explicit["next_actions"]))

            bundle = bundle_run_artifacts(run_json=object(), out=object(), notes_root=root, project_root=object())  # type: ignore[arg-type]
            self.assertFalse(bundle["ok"])
            self.assertIn("run.json", bundle["missing"])

            verification = verify_run_artifacts(run_json=object(), repair="false")  # type: ignore[arg-type]
            self.assertFalse(verification["passed"])
            self.assertIn("run_json_exists", verification["failed_checks"])

            bundle_verification = verify_bundle_artifacts(bundle_zip=object())  # type: ignore[arg-type]
            self.assertFalse(bundle_verification["passed"])
            self.assertIn("bundle_exists", bundle_verification["failed_checks"])

            summary = summarize_run_artifact(bad_run)
            self.assertEqual(summary["run_id"], "")
            self.assertEqual(summary["artifacts"], {})

    def test_run_selection_can_prefer_ready_run_over_newer_broken_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "good", db_path=root / "memory.sqlite")
            run_quickstart("demo", out_dir=root / "runs" / "bad", db_path=root / "memory.sqlite")

            good = root / "runs" / "good" / "run.json"
            bad = root / "runs" / "bad" / "run.json"
            (root / "runs" / "bad" / "paper" / "main.tex").unlink()
            payload = json.loads(bad.read_text(encoding="utf-8"))
            payload.get("artifacts", {}).pop("paper", None)
            bad.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.utime(good, (1_700_000_000, 1_700_000_000))
            os.utime(bad, (1_700_001_000, 1_700_001_000))

            latest = select_run_artifact(runs_root=root / "runs", policy="latest")
            best = select_run_artifact(runs_root=root / "runs", policy="best")
            ready = select_run_artifact(runs_root=root / "runs", policy="ready")
            self.assertEqual(Path(latest["path"]), bad)
            self.assertEqual(Path(best["path"]), good)
            self.assertEqual(Path(ready["path"]), good)

            listing = list_run_artifacts(runs_root=root / "runs", limit=2, selection="best")
            self.assertEqual(Path(listing["selected"]["path"]), good)
            self.assertEqual(Path(listing["runs"][0]["path"]), bad)
            self.assertFalse(listing["runs"][0]["selected"])
            self.assertTrue(listing["runs"][1]["selected"])
            self.assertEqual(listing["selected_rank"], 2)
            self.assertTrue(listing["selected_visible"])
            out = StringIO()
            with redirect_stdout(out):
                print_run_list(listing)
            self.assertIn("[selected: best]", out.getvalue())

            limited_listing = list_run_artifacts(runs_root=root / "runs", limit=1, selection="best")
            self.assertEqual(Path(limited_listing["selected"]["path"]), good)
            self.assertEqual(limited_listing["selected_rank"], 2)
            self.assertFalse(limited_listing["selected_visible"])
            self.assertIn("Increase `--limit` to at least 2", " ".join(limited_listing["next_actions"]))
            limited_out = StringIO()
            with redirect_stdout(limited_out):
                print_run_list(limited_listing)
            self.assertIn("increase --limit to at least 2", limited_out.getvalue())

            missing_latest = resolve_artifact("paper", runs_root=root / "runs", selection="latest")
            selected_best = resolve_artifact("paper", runs_root=root / "runs", selection="best")
            self.assertFalse(missing_latest["exists"])
            self.assertTrue(selected_best["exists"])
            self.assertEqual(Path(selected_best["path"]), root / "runs" / "good" / "paper" / "main.tex")
            self.assertEqual(selected_best["selection"], "best")
            self.assertEqual(Path(selected_best["selected_run"]), good)
            stray_ci = root / "runs" / "stray" / "CI_QUICKSTART.md"
            stray_ci.parent.mkdir()
            stray_ci.write_text("# stale CI summary\n", encoding="utf-8")
            latest_ci = resolve_artifact("ci", runs_root=root / "runs", selection="latest")
            selected_ci = resolve_artifact("ci", runs_root=root / "runs", selection="best")
            self.assertTrue(latest_ci["exists"])
            self.assertEqual(Path(latest_ci["path"]), stray_ci)
            self.assertFalse(selected_ci["exists"])
            self.assertEqual(selected_ci["selection"], "best")
            self.assertEqual(selected_ci.get("selected_run", ""), "")
            self.assertIn("quickstart --mode ci --run", " ".join(selected_ci["next_actions"]))

            status = project_status(runs_root=root / "runs", db_path=root / "memory.sqlite", notes_root=root, selection="best")
            self.assertEqual(status["run_selection"], "best")
            self.assertEqual(Path(status["selected_run"]["path"]), good)
            self.assertEqual(Path(status["latest_run"]["path"]), good)
            self.assertEqual(status["state"], "ready")
            self.assertTrue((root / "MECHFERRET.md").exists())

    def test_resolve_artifact_keeps_selection_guidance_when_no_run_has_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            review = resolve_artifact("review", runs_root=root / "runs", selection="best")
            self.assertFalse(review["exists"])
            self.assertEqual(Path(review["selected_run"]), root / "runs" / "demo" / "run.json")
            review_actions = " ".join(review["next_actions"])
            self.assertNotIn("No run with `review` found", review_actions)
            self.assertNotIn("paper --select best` first", review_actions)
            self.assertIn("configured provider", review_actions)

            bundle = resolve_artifact("bundle", runs_root=root / "runs", selection="best")
            self.assertFalse(bundle["exists"])
            self.assertEqual(Path(bundle["selected_run"]), root / "runs" / "demo" / "run.json")
            self.assertNotIn("No run with `bundle` found", " ".join(bundle["next_actions"]))

            (root / "runs" / "demo" / "paper" / "main.tex").unlink()
            run_json = root / "runs" / "demo" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload.get("artifacts", {}).pop("paper", None)
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            best = resolve_artifact("paper", runs_root=root / "runs", selection="best")
            ready = resolve_artifact("paper", runs_root=root / "runs", selection="ready")

            self.assertFalse(best["exists"])
            self.assertEqual(Path(best["selected_run"]), root / "runs" / "demo" / "run.json")
            self.assertNotIn("No run with `paper` found", " ".join(best["next_actions"]))
            self.assertIn("mechferret paper --select best", " ".join(best["next_actions"]))

            self.assertFalse(ready["exists"])
            self.assertEqual(ready.get("selected_run", ""), "")
            self.assertIn("No audit-passing run found", " ".join(ready["next_actions"]))
            self.assertIn("mechferret paper --select ready", " ".join(ready["next_actions"]))

            (root / "runs" / "demo" / "QUICKSTART.md").unlink()
            latest_quickstart = resolve_artifact("quickstart", runs_root=root / "runs", selection="latest")
            self.assertFalse(latest_quickstart["exists"])
            self.assertIn("fresh local quickstart dossier", " ".join(latest_quickstart["next_actions"]))
            quickstart = resolve_artifact("quickstart", runs_root=root / "runs", selection="best")
            quickstart_actions = " ".join(quickstart["next_actions"])
            self.assertFalse(quickstart["exists"])
            self.assertIn("fresh local quickstart dossier", quickstart_actions)
            self.assertEqual(quickstart_actions.count("fresh local quickstart dossier"), 1)
            stale_quickstart_guidance = "write a " + "quickstart index"
            self.assertNotIn(stale_quickstart_guidance, quickstart_actions)

            index = resolve_artifact("all", runs_root=root / "runs", selection="best")
            index_actions = " ".join(index["next_actions"])
            self.assertFalse(index["artifacts"]["quickstart"]["exists"])
            self.assertIn("fresh local quickstart dossier", " ".join(index["artifacts"]["quickstart"]["next_actions"]))
            self.assertIn("fresh local quickstart dossier", index_actions)
            self.assertNotIn(stale_quickstart_guidance, index_actions)
            index_out = StringIO()
            with redirect_stdout(index_out):
                print_artifact_result(index)
            self.assertIn("action: Run `mechferret quickstart --run`", index_out.getvalue())

    def test_verify_bundle_artifacts_supports_run_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "good", db_path=root / "memory.sqlite")
            run_quickstart("demo", out_dir=root / "runs" / "bad", db_path=root / "memory.sqlite")
            good = root / "runs" / "good" / "run.json"
            bad = root / "runs" / "bad" / "run.json"
            (root / "runs" / "bad" / "paper" / "main.tex").unlink()
            payload = json.loads(bad.read_text(encoding="utf-8"))
            payload.get("artifacts", {}).pop("paper", None)
            bad.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.utime(good, (1_700_000_000, 1_700_000_000))
            os.utime(bad, (1_700_001_000, 1_700_001_000))
            bundle = bundle_run_artifacts(
                runs_root=root / "runs",
                selection="best",
                out=root / "exports",
                notes_root=root,
                project_root=root / "openvla",
            )
            os.utime(good, (1_700_000_000, 1_700_000_000))
            os.utime(bad, (1_700_001_000, 1_700_001_000))

            latest = verify_bundle_artifacts(runs_root=root / "runs", selection="latest")
            best = verify_bundle_artifacts(runs_root=root / "runs", selection="best")
            self.assertFalse(latest["passed"])
            self.assertIn("bundle --select latest", " ".join(latest["next_actions"]))
            self.assertTrue(best["passed"])
            self.assertEqual(Path(best["path"]), Path(bundle["path"]))

    def test_bundle_run_artifacts_self_verifies_after_review_artifact_is_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            review = root / "runs" / "demo" / "paper" / "review.md"
            review.write_text("Soundness: 6/10\n", encoding="utf-8")
            run_json = root / "runs" / "demo" / "run.json"
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload.setdefault("artifacts", {})["review"] = str(review)
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            refresh_run_manifest(run_json)

            bundle = bundle_run_artifacts(
                runs_root=root / "runs",
                selection="best",
                out=root / "exports",
                notes_root=root,
                project_root=root / "openvla",
            )

            self.assertTrue(bundle["ok"], bundle["bundle_verification"]["failed_checks"])
            verification = verify_bundle_artifacts(bundle["path"])
            self.assertTrue(verification["passed"], verification["failed_checks"])
            self.assertNotIn("bundle_run_manifest_ledger_sha256", verification["failed_checks"])
            self.assertNotIn("bundle_run_manifest_ledger_bytes", verification["failed_checks"])

    def test_run_selection_prefers_fewer_advisories_for_best_and_ready(self):
        rows = [
            {
                "ok": True,
                "path": "/tmp/advisory/run.json",
                "mtime": 1_700_001_000,
                "readiness_score": 0.9,
                "audit": {
                    "passed": True,
                    "readiness_score": 0.9,
                    "failed_checks": [],
                    "advisories": [{"name": "synthetic_backend_not_final", "severity": "warning"}],
                },
                "artifacts": {"report": True, "paper": True},
            },
            {
                "ok": True,
                "path": "/tmp/clean/run.json",
                "mtime": 1_700_000_000,
                "readiness_score": 0.9,
                "audit": {"passed": True, "readiness_score": 0.9, "failed_checks": [], "advisories": []},
                "artifacts": {"report": True, "paper": True},
            },
        ]
        best = select_run_artifact(policy="best", _rows=rows)
        ready = select_run_artifact(policy="ready", _rows=rows)
        latest = select_run_artifact(policy="latest", _rows=rows)
        self.assertEqual(best["path"], "/tmp/clean/run.json")
        self.assertEqual(ready["path"], "/tmp/clean/run.json")
        self.assertEqual(latest["path"], "/tmp/advisory/run.json")

        stronger_with_advisory = [
            {**rows[0], "path": "/tmp/strong-advisory/run.json", "readiness_score": 0.95, "audit": {**rows[0]["audit"], "readiness_score": 0.95}},
            {**rows[1], "path": "/tmp/weaker-clean/run.json", "readiness_score": 0.7, "audit": {**rows[1]["audit"], "readiness_score": 0.7}},
        ]
        self.assertEqual(select_run_artifact(policy="best", _rows=stronger_with_advisory)["path"], "/tmp/strong-advisory/run.json")
        self.assertEqual(select_run_artifact(policy="ready", _rows=stronger_with_advisory)["path"], "/tmp/strong-advisory/run.json")

    def test_bundle_run_artifacts_creates_shareable_zip_and_ledger_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            run_json = root / "runs" / "demo" / "run.json"
            legacy_payload = json.loads(run_json.read_text(encoding="utf-8"))
            legacy_payload.get("artifacts", {}).pop("manifest", None)
            run_json.write_text(json.dumps(legacy_payload, indent=2, sort_keys=True), encoding="utf-8")

            result = bundle_run_artifacts(
                runs_root=root / "runs",
                out=root / "exports",
                notes_root=root,
                project_root=root / "openvla",
            )
            bundle = Path(result["path"])
            self.assertTrue(result["ok"])
            self.assertTrue(bundle.exists())
            self.assertEqual(bundle, root / "exports" / "mechferret-bundle.zip")
            with zipfile.ZipFile(bundle) as archive:
                names = set(archive.namelist())
                self.assertIn("manifest.json", names)
                self.assertIn("audit.json", names)
                self.assertIn("status.json", names)
                self.assertIn("README.md", names)
                self.assertIn("run/run.json", names)
                self.assertIn("run/manifest.json", names)
                self.assertIn("run/report.html", names)
                self.assertIn("paper/main.tex", names)
                self.assertIn("project/MECHFERRET.md", names)
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                archived_run = json.loads(archive.read("run/run.json").decode("utf-8"))
                run_manifest = json.loads(archive.read("run/manifest.json").decode("utf-8"))
            self.assertEqual(manifest["run_id"], result["manifest"]["run_id"])
            self.assertIn("advisories", manifest)
            self.assertTrue(all(item.get("sha256") for item in manifest["files"]))
            self.assertTrue(all(item.get("sha256") for item in manifest["metadata_files"]))
            self.assertIn("manifest", archived_run["artifacts"])
            self.assertIn("bundle", archived_run["artifacts"])
            self.assertIn("bundle", run_manifest["artifacts"])
            self.assertTrue(run_manifest["artifacts"]["bundle"]["exists"])
            bundle_verification = result["bundle_verification"]
            self.assertTrue(bundle_verification["passed"])
            self.assertEqual(bundle_verification["path"], str(bundle))
            payload = json.loads((root / "runs" / "demo" / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(payload["artifacts"]["bundle"]), bundle)
            self.assertTrue(verify_run_artifacts(root / "runs" / "demo" / "run.json")["passed"])
            resolved = resolve_artifact("bundle", runs_root=root / "runs")
            self.assertTrue(resolved["exists"])
            self.assertEqual(Path(resolved["path"]), bundle)

    def test_bundle_uses_run_relative_paths_not_caller_shadow_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            other = root / "other"
            shadow = other / "runs" / "demo" / "report.md"
            shadow.parent.mkdir(parents=True)
            shadow.write_text("wrong markdown\n", encoding="utf-8")
            (other / "runs" / "demo" / "report.html").write_text("<p>wrong report</p>", encoding="utf-8")

            previous = Path.cwd()
            try:
                os.chdir(other)
                result = bundle_run_artifacts(
                    root / "runs" / "demo" / "run.json",
                    out=root / "exports",
                    project_root=root / "openvla",
                )
            finally:
                os.chdir(previous)

            self.assertTrue(result["ok"])
            with zipfile.ZipFile(result["path"]) as archive:
                markdown = archive.read("run/report.md").decode("utf-8")
                html = archive.read("run/report.html").decode("utf-8")
            self.assertNotEqual(markdown, "wrong markdown\n")
            self.assertNotIn("wrong report", html)

    def test_bundle_status_metadata_is_bound_to_selected_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "good", db_path=root / "memory.sqlite")
            run_quickstart("demo", out_dir=root / "runs" / "bad", db_path=root / "memory.sqlite")
            good = root / "runs" / "good" / "run.json"
            bad = root / "runs" / "bad" / "run.json"
            (root / "runs" / "bad" / "paper" / "main.tex").unlink()
            payload = json.loads(bad.read_text(encoding="utf-8"))
            payload.get("artifacts", {}).pop("paper", None)
            bad.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.utime(good, (1_700_000_000, 1_700_000_000))
            os.utime(bad, (1_700_001_000, 1_700_001_000))

            result = bundle_run_artifacts(
                runs_root=root / "runs",
                selection="best",
                out=root / "exports",
                notes_root=root,
                project_root=root / "openvla",
            )

            with zipfile.ZipFile(result["path"]) as archive:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                status = json.loads(archive.read("status.json").decode("utf-8"))
            self.assertEqual(Path(result["run_json"]), good.resolve())
            self.assertEqual(manifest["selection"], "best")
            self.assertEqual(Path(manifest["status_run_json"]), good.resolve())
            self.assertEqual(Path(status["selected_run"]["path"]), good.resolve())
            self.assertEqual(Path(status["latest_run"]["path"]), good.resolve())
            self.assertNotEqual(Path(status["selected_run"]["path"]), bad.resolve())

    def test_bundle_run_artifacts_fails_when_self_verification_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            failed_verification = {
                "path": str(root / "exports" / "mechferret-bundle.zip"),
                "passed": False,
                "checks": [{"name": "bundle_entries_declared", "passed": False}],
                "failed_checks": ["bundle_entries_declared"],
                "next_actions": ["Recreate the bundle with `mechferret bundle`."],
            }

            with patch("mechferret.ops.verify_bundle_artifacts", return_value=failed_verification):
                result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")

            self.assertFalse(result["ok"])
            self.assertTrue(result["created"])
            self.assertEqual(result["bundle_verification"], failed_verification)
            self.assertEqual(result["next_actions"], failed_verification["next_actions"])

    def test_verify_bundle_artifacts_detects_archive_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
            entries["run/report.md"] = b"tampered\n"
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertIn("bundle_file_sha256:markdown_report", verification["failed_checks"])

            metadata_tampered = root / "exports" / "metadata-tampered.zip"
            with zipfile.ZipFile(bundle) as original:
                metadata_entries = {name: original.read(name) for name in original.namelist()}
            metadata_entries["audit.json"] = b'{"passed": true}\n'
            with zipfile.ZipFile(metadata_tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in metadata_entries.items():
                    archive.writestr(name, data)
            metadata_verification = verify_bundle_artifacts(metadata_tampered)
            self.assertFalse(metadata_verification["passed"])
            self.assertIn("bundle_metadata_sha256:audit", metadata_verification["failed_checks"])

    def test_verify_bundle_artifacts_checks_run_metadata_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            inconsistent = root / "exports" / "inconsistent-status.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                status = json.loads(entries["status.json"].decode("utf-8"))

            status["selected_run"]["path"] = str(root / "runs" / "other" / "run.json")
            status_bytes = json.dumps(status, indent=2, sort_keys=True).encode("utf-8")
            entries["status.json"] = status_bytes
            for row in manifest["metadata_files"]:
                if row["label"] == "status":
                    row["bytes"] = len(status_bytes)
                    row["sha256"] = hashlib.sha256(status_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(inconsistent, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(inconsistent)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_metadata_sha256:status", verification["failed_checks"])
            self.assertIn("bundle_status_selected_run_matches_manifest", verification["failed_checks"])

    def test_verify_bundle_artifacts_binds_status_metadata_to_archived_run_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            inconsistent = root / "exports" / "inconsistent-status-details.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                status = json.loads(entries["status.json"].decode("utf-8"))

            status["run_selection"] = "forged"
            status["selected_run"]["summary"]["run_id"] = "forged-run"
            status["latest_run"]["summary"]["question"] = "forged question"
            status["audit"]["passed"] = not status["audit"]["passed"]
            status["advisories"] = []
            status_bytes = json.dumps(status, indent=2, sort_keys=True).encode("utf-8")
            entries["status.json"] = status_bytes
            for row in manifest["metadata_files"]:
                if row["label"] == "status":
                    row["bytes"] = len(status_bytes)
                    row["sha256"] = hashlib.sha256(status_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(inconsistent, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(inconsistent)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_metadata_sha256:status", verification["failed_checks"])
            self.assertIn("bundle_status_selection_matches_manifest", verification["failed_checks"])
            self.assertIn("bundle_status_selected_run_summary_matches_run_json", verification["failed_checks"])
            self.assertIn("bundle_status_latest_run_summary_matches_run_json", verification["failed_checks"])
            self.assertIn("bundle_status_audit_matches_audit_json", verification["failed_checks"])
            self.assertIn("bundle_status_advisories_match_audit_json", verification["failed_checks"])

    def test_verify_bundle_artifacts_binds_manifest_question_and_selection_to_run(self):
        from mechferret.ops import _bundle_readme

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            inconsistent = root / "exports" / "inconsistent-manifest-metadata.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                audit = json.loads(entries["audit.json"].decode("utf-8"))
                status = json.loads(entries["status.json"].decode("utf-8"))

            manifest["question"] = "forged question"
            manifest["selection"] = "forged"
            status["run_selection"] = manifest["selection"]
            status_bytes = json.dumps(status, indent=2, sort_keys=True).encode("utf-8")
            readme_bytes = _bundle_readme(manifest, audit).encode("utf-8")
            entries["status.json"] = status_bytes
            entries["README.md"] = readme_bytes
            for row in manifest["metadata_files"]:
                if row["label"] == "status":
                    row["bytes"] = len(status_bytes)
                    row["sha256"] = hashlib.sha256(status_bytes).hexdigest()
                if row["label"] == "readme":
                    row["bytes"] = len(readme_bytes)
                    row["sha256"] = hashlib.sha256(readme_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(inconsistent, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(inconsistent)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_metadata_sha256:status", verification["failed_checks"])
            self.assertNotIn("bundle_metadata_sha256:readme", verification["failed_checks"])
            self.assertNotIn("bundle_status_selection_matches_manifest", verification["failed_checks"])
            self.assertNotIn("bundle_readme_matches_manifest_audit", verification["failed_checks"])
            self.assertIn("bundle_manifest_question_matches_run_json", verification["failed_checks"])
            self.assertIn("bundle_manifest_selection_supported", verification["failed_checks"])

    def test_verify_bundle_artifacts_binds_audit_metadata_to_archived_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            inconsistent = root / "exports" / "inconsistent-audit.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                audit = json.loads(entries["audit.json"].decode("utf-8"))

            audit["run_id"] = "forged-run"
            audit["question"] = "forged question"
            audit["readiness_score"] = 1.0
            if audit.get("checks"):
                audit["checks"][0]["observed"] = "forged"
            audit_bytes = json.dumps(audit, indent=2, sort_keys=True).encode("utf-8")
            entries["audit.json"] = audit_bytes
            for row in manifest["metadata_files"]:
                if row["label"] == "audit":
                    row["bytes"] = len(audit_bytes)
                    row["sha256"] = hashlib.sha256(audit_bytes).hexdigest()
            manifest["readiness_score"] = audit["readiness_score"]
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(inconsistent, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(inconsistent)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_metadata_sha256:audit", verification["failed_checks"])
            self.assertIn("bundle_audit_run_id_matches_run_json", verification["failed_checks"])
            self.assertIn("bundle_audit_question_matches_run_json", verification["failed_checks"])
            self.assertIn("bundle_audit_readiness_matches_run_json", verification["failed_checks"])
            self.assertIn("bundle_audit_checks_match_run_json", verification["failed_checks"])

    def test_verify_bundle_artifacts_binds_readme_metadata_to_manifest_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            inconsistent = root / "exports" / "inconsistent-readme.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))

            readme_bytes = b"# MechFerret Research Bundle\n\nRun: `forged`\n"
            entries["README.md"] = readme_bytes
            for row in manifest["metadata_files"]:
                if row["label"] == "readme":
                    row["bytes"] = len(readme_bytes)
                    row["sha256"] = hashlib.sha256(readme_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(inconsistent, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(inconsistent)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_metadata_sha256:readme", verification["failed_checks"])
            self.assertIn("bundle_readme_matches_manifest_audit", verification["failed_checks"])

    def test_verify_bundle_artifacts_checks_archived_run_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-run-ledger.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_payload = json.loads(entries["run/run.json"].decode("utf-8"))

            run_payload["answer"] = "This archived answer was rewritten after bundling."
            run_bytes = json.dumps(run_payload, indent=2, sort_keys=True).encode("utf-8")
            entries["run/run.json"] = run_bytes
            for row in manifest["files"]:
                if row["label"] == "run_json":
                    row["bytes"] = len(run_bytes)
                    row["sha256"] = hashlib.sha256(run_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:run_json", verification["failed_checks"])
            self.assertIn("bundle_run_manifest_ledger_sha256", verification["failed_checks"])

    def test_verify_bundle_artifacts_checks_archived_run_manifest_file_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-report-ledger.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))

            report_bytes = b"# Rewritten archived report\n"
            entries["run/report.md"] = report_bytes
            for row in manifest["files"]:
                if row["label"] == "markdown_report":
                    row["bytes"] = len(report_bytes)
                    row["sha256"] = hashlib.sha256(report_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:markdown_report", verification["failed_checks"])
            self.assertIn("bundle_run_manifest_artifact_sha256:markdown", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_malformed_archived_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "malformed-archived-artifact-hash.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_manifest = json.loads(entries["run/manifest.json"].decode("utf-8"))

            run_manifest["artifacts"]["markdown"]["sha256"] = "z" * 64
            manifest_bytes = json.dumps(run_manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["run/manifest.json"] = manifest_bytes
            for row in manifest["files"]:
                if row["label"] == "run_manifest":
                    row["bytes"] = len(manifest_bytes)
                    row["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:run_manifest", verification["failed_checks"])
            self.assertIn("bundle_run_manifest_artifact_sha256_declared:markdown", verification["failed_checks"])
            self.assertNotIn("bundle_run_manifest_artifact_sha256:markdown", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_malformed_archived_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-artifact-bytes.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_manifest = json.loads(entries["run/manifest.json"].decode("utf-8"))

            run_manifest["artifacts"]["markdown"]["bytes"] = True
            manifest_bytes = json.dumps(run_manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["run/manifest.json"] = manifest_bytes
            for row in manifest["files"]:
                if row["label"] == "run_manifest":
                    row["bytes"] = len(manifest_bytes)
                    row["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:run_manifest", verification["failed_checks"])
            self.assertIn("bundle_run_manifest_artifact_bytes_declared:markdown", verification["failed_checks"])
            self.assertNotIn("bundle_run_manifest_artifact_bytes:markdown", verification["failed_checks"])

    def test_verify_bundle_artifacts_checks_archived_run_manifest_tracks_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-missing-manifest-row.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_manifest = json.loads(entries["run/manifest.json"].decode("utf-8"))

            run_manifest["artifacts"].pop("markdown")
            manifest_bytes = json.dumps(run_manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["run/manifest.json"] = manifest_bytes
            for row in manifest["files"]:
                if row["label"] == "run_manifest":
                    row["bytes"] = len(manifest_bytes)
                    row["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:run_manifest", verification["failed_checks"])
            self.assertIn("bundle_run_manifest_tracks_artifact:markdown", verification["failed_checks"])

    def test_verify_bundle_artifacts_checks_archived_run_manifest_sources_and_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-manifest-source.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_manifest = json.loads(entries["run/manifest.json"].decode("utf-8"))

            source_id = run_manifest["sources"][0]["id"]
            run_manifest["schema_version"] = 0
            run_manifest["provenance"] = {"engine": "changed"}
            run_manifest["sources"][0]["text_sha256"] = "0" * 64
            manifest_bytes = json.dumps(run_manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["run/manifest.json"] = manifest_bytes
            for row in manifest["files"]:
                if row["label"] == "run_manifest":
                    row["bytes"] = len(manifest_bytes)
                    row["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:run_manifest", verification["failed_checks"])
            self.assertIn("bundle_run_manifest_schema_version_supported", verification["failed_checks"])
            self.assertIn("bundle_run_manifest_provenance_matches_run_json", verification["failed_checks"])
            self.assertIn(f"bundle_run_manifest_source_text_sha256_matches:{source_id}", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_malformed_archived_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-manifest-source-hash.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_manifest = json.loads(entries["run/manifest.json"].decode("utf-8"))

            source_id = run_manifest["sources"][0]["id"]
            run_manifest["sources"][0]["text_sha256"] = "z" * 64
            manifest_bytes = json.dumps(run_manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["run/manifest.json"] = manifest_bytes
            for row in manifest["files"]:
                if row["label"] == "run_manifest":
                    row["bytes"] = len(manifest_bytes)
                    row["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:run_manifest", verification["failed_checks"])
            self.assertIn(f"bundle_run_manifest_source_text_sha256_declared:{source_id}", verification["failed_checks"])
            self.assertNotIn(f"bundle_run_manifest_source_text_sha256_matches:{source_id}", verification["failed_checks"])

    def test_verify_bundle_artifacts_checks_archived_run_graph_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-run-graph.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_payload = json.loads(entries["run/run.json"].decode("utf-8"))

            run_payload["hypotheses"] = [
                {
                    "id": "h1",
                    "statement": "Closed ledgers make packaged findings inspectable.",
                    "rationale": "Archived findings need their graph references intact.",
                    "task": "bundle_verification",
                    "predicted_effect": "fewer broken shared dossiers",
                    "experiment_ids": ["missing_experiment"],
                    "source_ids": [run_payload["sources"][0]["id"]],
                }
            ]
            run_payload["experiments"] = [
                {
                    "id": "x1",
                    "spec_id": "spec_1",
                    "probe": "synthetic_probe",
                    "status": "ran",
                    "effect_size": 0.4,
                    "baseline": 0.1,
                }
            ]
            run_payload["discoveries"] = [
                {
                    "id": "d1",
                    "statement": "Broken archive graph references should fail bundle verification.",
                    "confidence": 0.8,
                    "novelty": 0.5,
                    "effect_size": 0.4,
                    "reproducibility": 1.0,
                    "supporting_experiments": ["missing_experiment"],
                    "claim_ids": ["missing_claim"],
                    "hypothesis_id": "missing_hypothesis",
                }
            ]
            run_bytes = json.dumps(run_payload, indent=2, sort_keys=True).encode("utf-8")
            entries["run/run.json"] = run_bytes
            for row in manifest["files"]:
                if row["label"] == "run_json":
                    row["bytes"] = len(run_bytes)
                    row["sha256"] = hashlib.sha256(run_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:run_json", verification["failed_checks"])
            self.assertIn("bundle_hypothesis_experiment_tracked:h1:missing_experiment", verification["failed_checks"])
            self.assertIn("bundle_discovery_experiment_tracked:d1:missing_experiment", verification["failed_checks"])
            self.assertIn("bundle_discovery_claim_tracked:d1:missing_claim", verification["failed_checks"])
            self.assertIn("bundle_discovery_hypothesis_tracked:d1:missing_hypothesis", verification["failed_checks"])

    def test_verify_bundle_artifacts_checks_archived_sidecar_ledgers_match_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            DiscoveryController(root / "memory.sqlite").run(
                skill="ioi-circuit", backend="synthetic", out_dir=root / "runs" / "discovery", include_memory=False
            )
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-sidecars.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_manifest = json.loads(entries["run/manifest.json"].decode("utf-8"))

            experiments_bytes = b"[]\n"
            discoveries_bytes = json.dumps(
                {
                    "run_id": run_manifest["run_id"],
                    "question": "Changed packaged sidecar question",
                    "discoveries": [],
                    "hypotheses": [],
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            entries["run/experiments.json"] = experiments_bytes
            entries["run/discoveries.json"] = discoveries_bytes
            run_manifest["artifacts"]["experiments"]["bytes"] = len(experiments_bytes)
            run_manifest["artifacts"]["experiments"]["sha256"] = hashlib.sha256(experiments_bytes).hexdigest()
            run_manifest["artifacts"]["discoveries"]["bytes"] = len(discoveries_bytes)
            run_manifest["artifacts"]["discoveries"]["sha256"] = hashlib.sha256(discoveries_bytes).hexdigest()
            run_manifest_bytes = json.dumps(run_manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["run/manifest.json"] = run_manifest_bytes
            for row in manifest["files"]:
                if row["label"] == "experiments":
                    row["bytes"] = len(experiments_bytes)
                    row["sha256"] = hashlib.sha256(experiments_bytes).hexdigest()
                if row["label"] == "discoveries":
                    row["bytes"] = len(discoveries_bytes)
                    row["sha256"] = hashlib.sha256(discoveries_bytes).hexdigest()
                if row["label"] == "run_manifest":
                    row["bytes"] = len(run_manifest_bytes)
                    row["sha256"] = hashlib.sha256(run_manifest_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:experiments", verification["failed_checks"])
            self.assertNotIn("bundle_run_manifest_artifact_sha256:experiments", verification["failed_checks"])
            self.assertIn("bundle_experiments_sidecar_matches_run", verification["failed_checks"])
            self.assertIn("bundle_discoveries_sidecar_question_matches_run", verification["failed_checks"])
            self.assertIn("bundle_discoveries_sidecar_discoveries_matches_run", verification["failed_checks"])

    def test_verify_bundle_artifacts_checks_archived_graph_and_evals_sidecars_match_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-generated-sidecars.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_manifest = json.loads(entries["run/manifest.json"].decode("utf-8"))

            graph_bytes = json.dumps(
                {"run_id": "changed", "question": "changed", "nodes": [], "edges": []},
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            evals_bytes = json.dumps(
                {"run_id": "changed", "passed": True, "checks": [], "readiness_score": 1.0},
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            entries["run/graph.json"] = graph_bytes
            entries["run/evals.json"] = evals_bytes
            run_manifest["artifacts"]["graph"]["bytes"] = len(graph_bytes)
            run_manifest["artifacts"]["graph"]["sha256"] = hashlib.sha256(graph_bytes).hexdigest()
            run_manifest["artifacts"]["evals"]["bytes"] = len(evals_bytes)
            run_manifest["artifacts"]["evals"]["sha256"] = hashlib.sha256(evals_bytes).hexdigest()
            run_manifest_bytes = json.dumps(run_manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["run/manifest.json"] = run_manifest_bytes
            for row in manifest["files"]:
                if row["label"] == "evidence_graph":
                    row["bytes"] = len(graph_bytes)
                    row["sha256"] = hashlib.sha256(graph_bytes).hexdigest()
                if row["label"] == "evals":
                    row["bytes"] = len(evals_bytes)
                    row["sha256"] = hashlib.sha256(evals_bytes).hexdigest()
                if row["label"] == "run_manifest":
                    row["bytes"] = len(run_manifest_bytes)
                    row["sha256"] = hashlib.sha256(run_manifest_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:evidence_graph", verification["failed_checks"])
            self.assertNotIn("bundle_run_manifest_artifact_sha256:graph", verification["failed_checks"])
            self.assertIn("bundle_graph_sidecar_matches_run", verification["failed_checks"])
            self.assertIn("bundle_evals_sidecar_matches_run", verification["failed_checks"])

    def test_verify_bundle_artifacts_checks_archived_report_sidecars_match_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-reports.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_manifest = json.loads(entries["run/manifest.json"].decode("utf-8"))

            markdown_bytes = b"# Rewritten report\n"
            html_bytes = b"<!doctype html><title>rewritten</title>\n"
            entries["run/report.md"] = markdown_bytes
            entries["run/report.html"] = html_bytes
            run_manifest["artifacts"]["markdown"]["bytes"] = len(markdown_bytes)
            run_manifest["artifacts"]["markdown"]["sha256"] = hashlib.sha256(markdown_bytes).hexdigest()
            run_manifest["artifacts"]["html"]["bytes"] = len(html_bytes)
            run_manifest["artifacts"]["html"]["sha256"] = hashlib.sha256(html_bytes).hexdigest()
            run_manifest_bytes = json.dumps(run_manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["run/manifest.json"] = run_manifest_bytes
            for row in manifest["files"]:
                if row["label"] == "markdown_report":
                    row["bytes"] = len(markdown_bytes)
                    row["sha256"] = hashlib.sha256(markdown_bytes).hexdigest()
                if row["label"] == "html_report":
                    row["bytes"] = len(html_bytes)
                    row["sha256"] = hashlib.sha256(html_bytes).hexdigest()
                if row["label"] == "run_manifest":
                    row["bytes"] = len(run_manifest_bytes)
                    row["sha256"] = hashlib.sha256(run_manifest_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:markdown_report", verification["failed_checks"])
            self.assertNotIn("bundle_run_manifest_artifact_sha256:markdown", verification["failed_checks"])
            self.assertIn("bundle_markdown_sidecar_matches_run", verification["failed_checks"])
            self.assertIn("bundle_html_sidecar_matches_run", verification["failed_checks"])

    def test_verify_bundle_artifacts_checks_archived_paper_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-paper.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_manifest = json.loads(entries["run/manifest.json"].decode("utf-8"))

            paper_bytes = b"\\documentclass{article}\\begin{document}ok\\end{document}\n"
            entries["paper/main.tex"] = paper_bytes
            run_manifest["artifacts"]["paper"]["bytes"] = len(paper_bytes)
            run_manifest_bytes = json.dumps(run_manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["run/manifest.json"] = run_manifest_bytes
            for row in manifest["files"]:
                if row["label"] == "paper_tex":
                    row["bytes"] = len(paper_bytes)
                    row["sha256"] = hashlib.sha256(paper_bytes).hexdigest()
                if row["label"] == "run_manifest":
                    row["bytes"] = len(run_manifest_bytes)
                    row["sha256"] = hashlib.sha256(run_manifest_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:paper_tex", verification["failed_checks"])
            self.assertIn("bundle_paper_artifact_latex_structure", verification["failed_checks"])
            structure = next(c for c in verification["checks"] if c["name"] == "bundle_paper_artifact_latex_structure")
            self.assertIn("Evidence Ledger", structure["threshold"])

    def test_verify_bundle_artifacts_checks_archived_review_and_pdf_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            run_json = root / "runs" / "demo" / "run.json"
            paper_dir = root / "runs" / "demo" / "paper"
            review = paper_dir / "review.md"
            pdf = paper_dir / "main.pdf"
            review.write_text("Recommendation: Borderline\n", encoding="utf-8")
            pdf.write_bytes(b"%PDF fake\n")
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["artifacts"]["review"] = str(review)
            payload["artifacts"]["pdf"] = str(pdf)
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            refresh_run_manifest(run_json)

            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-review-pdf.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_manifest = json.loads(entries["run/manifest.json"].decode("utf-8"))

            review_bytes = b"\n"
            pdf_bytes = b"not a pdf\n"
            entries["paper/review.md"] = review_bytes
            entries["paper/main.pdf"] = pdf_bytes
            run_manifest["artifacts"]["review"]["bytes"] = len(review_bytes)
            run_manifest["artifacts"]["pdf"]["bytes"] = len(pdf_bytes)
            run_manifest_bytes = json.dumps(run_manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["run/manifest.json"] = run_manifest_bytes
            for row in manifest["files"]:
                if row["label"] == "paper_review":
                    row["bytes"] = len(review_bytes)
                    row["sha256"] = hashlib.sha256(review_bytes).hexdigest()
                if row["label"] == "paper_pdf":
                    row["bytes"] = len(pdf_bytes)
                    row["sha256"] = hashlib.sha256(pdf_bytes).hexdigest()
                if row["label"] == "run_manifest":
                    row["bytes"] = len(run_manifest_bytes)
                    row["sha256"] = hashlib.sha256(run_manifest_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:paper_review", verification["failed_checks"])
            self.assertNotIn("bundle_file_sha256:paper_pdf", verification["failed_checks"])
            self.assertIn("bundle_review_artifact_nonempty", verification["failed_checks"])
            self.assertIn("bundle_pdf_artifact_header", verification["failed_checks"])

    def test_verify_bundle_artifacts_checks_archived_trace_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            tampered = root / "exports" / "tampered-trace.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))
                run_manifest = json.loads(entries["run/manifest.json"].decode("utf-8"))

            trace_bytes = (
                json.dumps(
                    {
                        "trace_id": "trace",
                        "run_id": "other_run",
                        "span_id": "span",
                        "phase": "event",
                        "name": "artifacts_written",
                        "time_unix_ms": 1,
                        "elapsed_ms": 0.0,
                        "attributes": {},
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            entries["run/trace.jsonl"] = trace_bytes
            run_manifest["artifacts"]["trace"]["bytes"] = len(trace_bytes)
            run_manifest_bytes = json.dumps(run_manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["run/manifest.json"] = run_manifest_bytes
            for row in manifest["files"]:
                if row["label"] == "trace":
                    row["bytes"] = len(trace_bytes)
                    row["sha256"] = hashlib.sha256(trace_bytes).hexdigest()
                if row["label"] == "run_manifest":
                    row["bytes"] = len(run_manifest_bytes)
                    row["sha256"] = hashlib.sha256(run_manifest_bytes).hexdigest()
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(tampered)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:trace", verification["failed_checks"])
            self.assertIn("bundle_trace_artifact_run_id", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_extra_or_duplicate_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            extra = root / "exports" / "extra.zip"
            duplicate = root / "exports" / "duplicate.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = [(name, original.read(name)) for name in original.namelist()]
            with zipfile.ZipFile(extra, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries:
                    archive.writestr(name, data)
                archive.writestr("unexpected/payload.txt", "surprise\n")
            with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries:
                    archive.writestr(name, data)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr("run/report.md", dict(entries)["run/report.md"])

            extra_verification = verify_bundle_artifacts(extra)
            self.assertFalse(extra_verification["passed"])
            self.assertIn("bundle_entries_declared", extra_verification["failed_checks"])
            duplicate_verification = verify_bundle_artifacts(duplicate)
            self.assertFalse(duplicate_verification["passed"])
            self.assertIn("bundle_entries_unique", duplicate_verification["failed_checks"])
            duplicate_check_names = {check["name"] for check in duplicate_verification["checks"]}
            self.assertNotIn("bundle_run_json_parseable", duplicate_check_names)

    def test_verify_bundle_artifacts_rejects_non_object_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            malformed = root / "exports" / "manifest-array.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
            entries["manifest.json"] = b"[]"
            with zipfile.ZipFile(malformed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(malformed)
            self.assertFalse(verification["passed"])
            self.assertIn("bundle_manifest_object", verification["failed_checks"])
            self.assertNotIn("bundle_manifest_files_parseable", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_non_object_file_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            malformed = root / "exports" / "manifest-bad-row.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))

            manifest["files"].append("not-a-row")
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(malformed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(malformed)
            self.assertFalse(verification["passed"])
            self.assertIn("bundle_file_entry_parseable", verification["failed_checks"])
            self.assertIn("bundle_file_labels_declared", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_malformed_file_row_byte_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            malformed = root / "exports" / "manifest-bad-bytes.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))

            for row in manifest["files"]:
                if row["label"] == "run_json":
                    row["bytes"] = "large"
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(malformed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(malformed)
            self.assertFalse(verification["passed"])
            self.assertIn("bundle_file_bytes_declared:run_json", verification["failed_checks"])
            self.assertNotIn("bundle_file_sha256:run_json", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_malformed_file_row_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            malformed = root / "exports" / "manifest-bad-hash.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))

            for row in manifest["files"]:
                if row["label"] == "run_json":
                    row["sha256"] = "z" * 64
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(malformed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(malformed)
            self.assertFalse(verification["passed"])
            self.assertIn("bundle_file_sha256_declared:run_json", verification["failed_checks"])
            self.assertNotIn("bundle_file_sha256:run_json", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_duplicate_manifest_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            duplicate = root / "exports" / "duplicate-label.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))

            copy_arcname = "run/run-copy.json"
            copy_bytes = entries["run/run.json"]
            entries[copy_arcname] = copy_bytes
            duplicate_row = dict(next(row for row in manifest["files"] if row["label"] == "run_json"))
            duplicate_row["arcname"] = copy_arcname
            duplicate_row["bytes"] = len(copy_bytes)
            duplicate_row["sha256"] = hashlib.sha256(copy_bytes).hexdigest()
            manifest["files"].append(duplicate_row)
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(duplicate)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:run_json", verification["failed_checks"])
            self.assertNotIn("bundle_entries_declared", verification["failed_checks"])
            self.assertIn("bundle_file_labels_unique", verification["failed_checks"])

    def test_verify_bundle_artifacts_requires_core_manifest_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            renamed = root / "exports" / "renamed-core-label.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))

            for row in manifest["files"]:
                if row["label"] == "run_json":
                    row["label"] = "payload_json"
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(renamed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(renamed)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_sha256:payload_json", verification["failed_checks"])
            self.assertNotIn("bundle_entries_declared", verification["failed_checks"])
            self.assertIn("bundle_file_labels_complete", verification["failed_checks"])

    def test_verify_bundle_artifacts_requires_core_manifest_label_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            moved = root / "exports" / "moved-core-entry.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))

            entries["run/payload.json"] = entries.pop("run/run.json")
            for row in manifest["files"]:
                if row["label"] == "run_json":
                    row["arcname"] = "run/payload.json"
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(moved, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(moved)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_labels_complete", verification["failed_checks"])
            self.assertNotIn("bundle_file_sha256:run_json", verification["failed_checks"])
            self.assertNotIn("bundle_entries_declared", verification["failed_checks"])
            self.assertIn("bundle_file_label_arcnames_canonical", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_duplicate_manifest_arcnames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            duplicate = root / "exports" / "duplicate-arcname.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))

            duplicate_row = dict(next(row for row in manifest["files"] if row["label"] == "run_json"))
            duplicate_row["label"] = "run_json_copy"
            manifest["files"].append(duplicate_row)
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(duplicate)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_file_labels_unique", verification["failed_checks"])
            self.assertNotIn("bundle_entries_declared", verification["failed_checks"])
            self.assertIn("bundle_file_arcnames_unique", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_metadata_reusing_file_arcname(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            duplicate = root / "exports" / "metadata-reuses-file-arcname.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))

            run_bytes = entries["run/run.json"]
            manifest["metadata_files"].append(
                {
                    "label": "metadata_points_at_run",
                    "arcname": "run/run.json",
                    "bytes": len(run_bytes),
                    "sha256": hashlib.sha256(run_bytes).hexdigest(),
                }
            )
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(duplicate)
            self.assertFalse(verification["passed"])
            self.assertNotIn("bundle_metadata_labels_unique", verification["failed_checks"])
            self.assertNotIn("bundle_metadata_arcnames_unique", verification["failed_checks"])
            self.assertNotIn("bundle_entries_declared", verification["failed_checks"])
            self.assertIn("bundle_metadata_arcnames_unclaimed", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_path_traversal_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            traversal = root / "exports" / "traversal.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(original.read("manifest.json").decode("utf-8"))
            payload = b"outside\n"
            manifest["files"].append(
                {
                    "label": "traversal",
                    "arcname": "../outside.txt",
                    "path": "/tmp/outside.txt",
                    "bytes": len(payload),
                    "sha256": "b8314c1e1d492e953c3f5945cd41390d1987f46d8ea53aa6cc671a4f678b3f23",
                }
            )
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            entries["../outside.txt"] = payload
            with zipfile.ZipFile(traversal, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(traversal)
            self.assertFalse(verification["passed"])
            self.assertIn("bundle_entries_path_safe", verification["failed_checks"])
            self.assertIn("bundle_file_path_safe:traversal", verification["failed_checks"])
            self.assertNotIn("bundle_file_bytes:traversal", verification["failed_checks"])
            self.assertNotIn("bundle_file_sha256:traversal", verification["failed_checks"])

    def test_verify_bundle_artifacts_rejects_non_string_manifest_arcnames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            malformed = root / "exports" / "malformed-arcname.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(original.read("manifest.json").decode("utf-8"))
            payload = entries["run/report.md"]
            manifest["files"].append(
                {
                    "label": "bad_arcname",
                    "arcname": ["run/report.md"],
                    "path": "/tmp/report.md",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(malformed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(malformed)
            self.assertFalse(verification["passed"])
            self.assertIn("bundle_file_path_safe:bad_arcname", verification["failed_checks"])
            self.assertIn("bundle_file_exists:bad_arcname", verification["failed_checks"])
            self.assertNotIn("bundle_file_bytes:bad_arcname", verification["failed_checks"])
            self.assertNotIn("bundle_file_sha256:bad_arcname", verification["failed_checks"])

    def test_verify_bundle_artifacts_refuses_unsafe_semantic_arcnames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            result = bundle_run_artifacts(runs_root=root / "runs", out=root / "exports", project_root=root / "openvla")
            bundle = Path(result["path"])
            unsafe = root / "exports" / "unsafe-semantic-arcname.zip"
            with zipfile.ZipFile(bundle) as original:
                entries = {name: original.read(name) for name in original.namelist()}
                manifest = json.loads(entries["manifest.json"].decode("utf-8"))

            audit_bytes = entries.pop("audit.json")
            entries["../audit.json"] = audit_bytes
            for row in manifest["metadata_files"]:
                if row["label"] == "audit":
                    row["arcname"] = "../audit.json"
            entries["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            with zipfile.ZipFile(unsafe, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)

            verification = verify_bundle_artifacts(unsafe)
            self.assertFalse(verification["passed"])
            self.assertIn("bundle_entries_path_safe", verification["failed_checks"])
            self.assertIn("bundle_metadata_path_safe:audit", verification["failed_checks"])
            self.assertIn("bundle_metadata_exists:audit", verification["failed_checks"])
            self.assertIn("bundle_audit_parseable", verification["failed_checks"])

    def test_resolve_artifact_exposes_discovery_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            DiscoveryController(root / "memory.sqlite").run(
                skill="ioi-circuit",
                backend="synthetic",
                out_dir=root / "runs" / "discovery",
                include_memory=False,
            )

            experiments = resolve_artifact("experiments", runs_root=root / "runs")
            discoveries = resolve_artifact("discoveries", runs_root=root / "runs")
            self.assertTrue(experiments["exists"])
            self.assertTrue(discoveries["exists"])
            self.assertEqual(Path(experiments["path"]), root / "runs" / "discovery" / "experiments.json")
            self.assertEqual(Path(discoveries["path"]), root / "runs" / "discovery" / "discoveries.json")
            index = resolve_artifact("all", runs_root=root / "runs", project_root=root / "openvla")
            self.assertTrue(index["artifacts"]["experiments"]["exists"])
            self.assertTrue(index["artifacts"]["discoveries"]["exists"])
            self.assertEqual(index["artifact_summary"]["total"], len(index["artifacts"]))
            self.assertIn("experiments", index["artifact_summary"]["groups"]["discovery"]["found_artifacts"])
            self.assertIn("bundle", index["artifact_summary"]["groups"]["sharing"]["missing_artifacts"])
            self.assertIn("artifact_readiness", index)
            self.assertFalse(index["artifact_readiness"]["run"]["ok"])
            self.assertFalse(index["artifact_readiness"]["sharing"]["ok"])
            self.assertFalse(index["complete"])
            self.assertFalse(index["run_ready"])
            self.assertFalse(index["share_ready"])
            self.assertIn("bundle", index["artifact_readiness"]["sharing"]["missing_artifacts"])
            index_actions = " ".join(index["next_actions"])
            self.assertIn("mechferret paper --select latest", index_actions)
            self.assertNotIn("review-paper", index_actions)
            self.assertNotIn("bundle --select latest", index_actions)
            out = StringIO()
            with redirect_stdout(out):
                print_artifact_result(index)
            self.assertIn("Summary:", out.getvalue())
            self.assertIn("Readiness:", out.getvalue())
            self.assertIn("Complete: no", out.getvalue())
            self.assertIn("discovery 2/2", out.getvalue())

    def test_run_quickstart_ci_executes_release_gates_with_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_run(_command, **_kwargs):
                return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

            with patch("mechferret.ops.subprocess.run", fake_run):
                result = run_quickstart("ci", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")

            self.assertTrue(result["ok"])
            self.assertEqual(result["mode"], "ci")
            names = {step["name"] for step in result["steps"]}
            self.assertIn("unit_tests", names)
            self.assertIn("compileall", names)
            self.assertIn("verify_manifest", names)
            self.assertIn("bundle_artifacts", names)
            self.assertIn("verify_bundle", names)
            self.assertIn("audit_strict", names)
            bundle_path = root / "runs" / "demo" / "mechferret-bundle.zip"
            self.assertTrue(bundle_path.exists())
            with zipfile.ZipFile(bundle_path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["selection"], "best")
            self.assertTrue((root / "runs" / "demo" / "ci_quickstart.json").exists())
            self.assertTrue((root / "runs" / "demo" / "CI_QUICKSTART.md").exists())
            resolved = resolve_artifact("ci", runs_root=root / "runs")
            self.assertTrue(resolved["exists"])
            self.assertEqual(Path(resolved["path"]), root / "runs" / "demo" / "CI_QUICKSTART.md")

    def test_run_quickstart_ci_bundles_the_fresh_run_even_when_other_runs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "older", db_path=root / "memory.sqlite")
            older_run = root / "runs" / "older" / "run.json"
            os.utime(older_run, (2_000_000_000, 2_000_000_000))

            def fake_run(_command, **_kwargs):
                return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

            with patch("mechferret.ops.subprocess.run", fake_run):
                result = run_quickstart("ci", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")

            self.assertTrue(result["ok"])
            bundle_path = Path(result["artifacts"]["bundle"])
            with zipfile.ZipFile(bundle_path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(Path(manifest["run_json"]), (root / "runs" / "demo" / "run.json").resolve())
            self.assertNotEqual(Path(manifest["run_json"]), older_run.resolve())

    def test_run_quickstart_openvla_scaffolds_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "openvla"
            result = run_quickstart("openvla", project_root=root)
            self.assertTrue(result["ok"])
            self.assertTrue((root / "src" / "train_sae_from_cache.py").exists())
            self.assertTrue((root / "quickstart.json").exists())
            self.assertTrue((root / "QUICKSTART.md").exists())
            self.assertIn("OpenVLA SAE Quickstart", (root / "QUICKSTART.md").read_text(encoding="utf-8"))
            resolved = resolve_artifact("openvla", project_root=root)
            self.assertTrue(resolved["exists"])
            self.assertEqual(Path(resolved["path"]), root / "QUICKSTART.md")

    def test_memory_resume_and_cost_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text(
                "# Evidence\nAutoresearch systems need claims, citations, memory, retrieval, and critic loops.",
                encoding="utf-8",
            )
            run = MechFerret(root / "memory.sqlite").run(
                "What does autoresearch need?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )
            summary = memory_summary(root / "memory.sqlite")
            self.assertEqual(summary["runs"], 1)
            self.assertGreaterEqual(len(memory_recent(root / "memory.sqlite", 1)), 1)
            resume = summarize_run_artifact(root / "run" / "run.json")
            self.assertEqual(resume["run_id"], run.run_id)
            cost = estimate_run_cost(root / "run" / "run.json")
            self.assertGreater(cost["estimated_tokens_processed"], 0)

    def test_cli_memory_json_reports_summary_recent_and_clear(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text(
                "# Evidence\nAutoresearch systems need claims, citations, memory, retrieval, and critic loops.",
                encoding="utf-8",
            )
            MechFerret(root / "memory.sqlite").run(
                "What does autoresearch need?",
                source_paths=[str(source)],
                out_dir=root / "run",
                include_memory=False,
            )
            out = StringIO()
            with redirect_stdout(out):
                main(["memory", "--db", str(root / "memory.sqlite"), "--recent", "1", "--json"])
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["db"], str(root / "memory.sqlite"))
            self.assertEqual(payload["summary"]["runs"], 1)
            self.assertEqual(len(payload["recent"]), 1)

            clear_out = StringIO()
            with redirect_stdout(clear_out):
                main(["memory", "--db", str(root / "memory.sqlite"), "--clear", "--json"])
            cleared = json.loads(clear_out.getvalue())
            self.assertTrue(cleared["ok"])
            self.assertTrue(cleared["cleared"])

    def test_cost_estimator_tolerates_malformed_artifact_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(
                json.dumps(
                    {
                        "run_id": 123,
                        "answer": ["not text"],
                        "sources": [
                            {"kind": "openai_web_search", "text": "provider source text"},
                            ["not a source"],
                            {"kind": "local", "text": {"not": "text"}},
                        ],
                        "plan": {"steps": "not a list"},
                    }
                ),
                encoding="utf-8",
            )

            cost = estimate_run_cost(path)

            self.assertEqual(cost["run_id"], "")
            self.assertGreater(cost["estimated_tokens_processed"], 0)
            self.assertEqual(cost["estimated_provider_calls"], 1)
            self.assertEqual(cost["local_steps"], 0)

    def test_cli_cost_json_reports_estimate_and_missing_runs(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            out = StringIO()
            with redirect_stdout(out):
                main(["cost", "--runs-root", str(root / "runs"), "--json"])
            estimate = json.loads(out.getvalue())
            self.assertTrue(estimate["ok"])
            self.assertEqual(Path(estimate["path"]), root / "runs" / "demo" / "run.json")
            self.assertGreater(estimate["estimated_tokens_processed"], 0)

            missing_out = StringIO()
            missing_err = StringIO()
            with self.assertRaises(SystemExit) as ctx:
                with redirect_stdout(missing_out), redirect_stderr(missing_err):
                    main(["cost", "--runs-root", str(root / "missing-runs"), "--select", "ready", "--json"])
            self.assertEqual(ctx.exception.code, 1)
            missing = json.loads(missing_out.getvalue())
            self.assertFalse(missing["ok"])
            self.assertEqual(missing["error"], "no run selected")
            self.assertEqual(missing["selection"], "ready")
            self.assertTrue(missing["next_actions"])
            self.assertEqual(missing_err.getvalue(), "")

    def test_cli_resume_and_inspect_json_report_summary_and_missing_runs(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            for command in ("resume", "inspect"):
                with self.subTest(command=command, mode="success"):
                    out = StringIO()
                    with redirect_stdout(out):
                        main([command, "--runs-root", str(root / "runs"), "--json"])
                    payload = json.loads(out.getvalue())
                    self.assertTrue(payload["ok"])
                    self.assertEqual(Path(payload["path"]), root / "runs" / "demo" / "run.json")
                    self.assertEqual(payload["selection"], "latest")
                    self.assertIn("question", payload)
                    self.assertIn("audit", payload)
                    self.assertIn("artifact_readiness", payload)
                    self.assertIn("run", payload["artifact_readiness"])
                    self.assertFalse(payload["artifact_readiness"]["sharing"]["ok"])
                    self.assertIn("next_actions", payload)

                with self.subTest(command=command, mode="missing"):
                    missing_out = StringIO()
                    missing_err = StringIO()
                    with self.assertRaises(SystemExit) as ctx:
                        with redirect_stdout(missing_out), redirect_stderr(missing_err):
                            main([command, "--runs-root", str(root / "missing-runs"), "--select", "ready", "--json"])
                    self.assertEqual(ctx.exception.code, 1)
                    missing = json.loads(missing_out.getvalue())
                    self.assertFalse(missing["ok"])
                    self.assertEqual(missing["error"], "no run selected")
                    self.assertEqual(missing["selection"], "ready")
                    self.assertTrue(missing["next_actions"])
                    self.assertEqual(missing_err.getvalue(), "")

                with self.subTest(command=command, mode="missing-explicit-text"):
                    missing_path = root / "runs" / "missing" / "run.json"
                    missing_out = StringIO()
                    missing_err = StringIO()
                    with self.assertRaises(SystemExit) as ctx:
                        with redirect_stdout(missing_out), redirect_stderr(missing_err):
                            main([command, str(missing_path)])
                    self.assertEqual(ctx.exception.code, 1)
                    self.assertEqual(missing_out.getvalue(), "")
                    self.assertIn(f"Run artifact not found: {missing_path}", missing_err.getvalue())
                    self.assertIn("quickstart --run", missing_err.getvalue())

    def test_cli_run_artifact_json_commands_report_missing_explicit_path_as_json(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "runs" / "missing" / "run.json"
            for command in ("audit", "verify", "bundle"):
                with self.subTest(command=command):
                    out = StringIO()
                    err = StringIO()
                    with self.assertRaises(SystemExit) as ctx:
                        with redirect_stdout(out), redirect_stderr(err):
                            main([command, str(missing_path), "--json"])
                    self.assertEqual(ctx.exception.code, 1)
                    payload = json.loads(out.getvalue())
                    self.assertFalse(payload["ok"])
                    self.assertEqual(payload["error"], "run artifact not found")
                    self.assertEqual(Path(payload["path"]), missing_path)
                    self.assertTrue(payload["next_actions"])
                    self.assertEqual(err.getvalue(), "")

    def test_cli_verification_json_payloads_include_ok_alias(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")

            for command in ("audit", "verify"):
                with self.subTest(command=command):
                    out = StringIO()
                    with redirect_stdout(out):
                        main([command, "--runs-root", str(root / "runs"), "--json"])
                    payload = json.loads(out.getvalue())
                    self.assertEqual(payload["ok"], payload["passed"])

            bundle_out = StringIO()
            with redirect_stdout(bundle_out):
                main([
                    "bundle",
                    "--runs-root",
                    str(root / "runs"),
                    "--out",
                    str(root / "exports"),
                    "--notes-root",
                    str(root),
                    "--project-root",
                    str(root / "openvla"),
                    "--json",
                ])
            bundle = json.loads(bundle_out.getvalue())
            out = StringIO()
            with redirect_stdout(out):
                main(["verify-bundle", bundle["path"], "--json"])
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["ok"], payload["passed"])

    def test_cli_run_artifact_helpers_default_to_latest_run(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")

            for command, expected in (
                ("resume", "Run:"),
                ("inspect", "Question:"),
                ("cost", "Estimated tokens processed:"),
                ("status", "Project status:"),
                ("runs", "Runs:"),
                ("bundle", "Bundle:"),
                ("verify-bundle", "Bundle verify:"),
                ("verify", "Verify:"),
            ):
                out = StringIO()
                with redirect_stdout(out):
                    args = [command, "--runs-root", str(root / "runs")]
                    if command == "status":
                        args.extend(["--db", str(root / "memory.sqlite"), "--notes-root", str(root), "--project-root", str(root / "openvla")])
                    elif command == "bundle":
                        args.extend(["--out", str(root / "bundle-out"), "--notes-root", str(root), "--project-root", str(root / "openvla")])
                    elif command == "verify-bundle":
                        args.extend(["--select", "latest"])
                    main(args)
                self.assertIn(expected, out.getvalue())
                if command in {"resume", "inspect"}:
                    self.assertIn("Audit: PASS", out.getvalue())
                    self.assertIn("Artifacts: run=", out.getvalue())

    def test_cli_inspect_tolerates_malformed_run_artifact(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            run_json = Path(tmp) / "run.json"
            run_json.write_text("[]", encoding="utf-8")

            out = StringIO()
            with redirect_stdout(out):
                main(["inspect", str(run_json)])
            rendered = out.getvalue()
            self.assertIn("Question:", rendered)
            self.assertIn("Readiness: 0.00", rendered)
            self.assertIn("Artifacts: run=READY", rendered)
            self.assertIn("Claims: 0", rendered)

    def test_status_helpers_accept_direct_run_json_path(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            run_json = root / "runs" / "demo" / "run.json"

            status = project_status(runs_root=run_json, db_path=root / "memory.sqlite", notes_root=root)
            self.assertTrue(status["selected_run"]["exists"])
            self.assertEqual(Path(status["selected_run"]["path"]), run_json)
            listing = list_run_artifacts(runs_root=run_json, include_audit=False)
            self.assertEqual(listing["count"], 1)
            self.assertEqual(Path(listing["selected_path"]), run_json)
            out = StringIO()
            with redirect_stdout(out):
                main(["status", "--runs-root", str(run_json), "--db", str(root / "memory.sqlite"), "--notes-root", str(root)])
            self.assertIn(f"Latest run: {run_json}", out.getvalue())

    def test_cli_bundle_preserves_selection_metadata(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_project_notes(root)
            run_quickstart("demo", out_dir=root / "runs" / "good", db_path=root / "memory.sqlite")
            run_quickstart("demo", out_dir=root / "runs" / "bad", db_path=root / "memory.sqlite")
            good = root / "runs" / "good" / "run.json"
            bad = root / "runs" / "bad" / "run.json"
            (root / "runs" / "bad" / "paper" / "main.tex").unlink()
            payload = json.loads(bad.read_text(encoding="utf-8"))
            payload.get("artifacts", {}).pop("paper", None)
            bad.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.utime(good, (1_700_000_000, 1_700_000_000))
            os.utime(bad, (1_700_001_000, 1_700_001_000))

            out = StringIO()
            with redirect_stdout(out):
                main([
                    "bundle",
                    "--runs-root",
                    str(root / "runs"),
                    "--select",
                    "best",
                    "--out",
                    str(root / "exports"),
                    "--notes-root",
                    str(root),
                    "--project-root",
                    str(root / "openvla"),
                ])
            bundle_path = root / "exports" / "mechferret-bundle.zip"
            with zipfile.ZipFile(bundle_path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["selection"], "best")
            self.assertEqual(Path(manifest["run_json"]), good.resolve())

    def test_cli_verify_repair_refreshes_stale_manifest_coverage(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_quickstart("demo", out_dir=root / "runs" / "demo", db_path=root / "memory.sqlite")
            run_json = root / "runs" / "demo" / "run.json"
            paper = root / "runs" / "demo" / "paper" / "extra.tex"
            paper.write_text("\\documentclass{article}", encoding="utf-8")
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            payload["artifacts"]["paper_extra"] = str(paper)
            run_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

            out = StringIO()
            with redirect_stdout(out):
                main(["verify", str(run_json), "--repair", "--json"])
            result = json.loads(out.getvalue())
            self.assertTrue(result["passed"])
            self.assertTrue(result["repair_attempted"])
            self.assertIn("manifest_tracks_declared_artifact:paper_extra", result["before_failed_checks"])

    def test_cli_tool_results_lists_and_dry_runs_cleanup(self):
        from mechferret import tools
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_dir = tools.RESULTS_DIR
            tools.RESULTS_DIR = root / "tool-results"
            try:
                tools.RESULTS_DIR.mkdir(parents=True)
                old = tools.RESULTS_DIR / "verify_bundle_old.json"
                new = tools.RESULTS_DIR / "bash_new.txt"
                old.write_text(json.dumps({"passed": True}), encoding="utf-8")
                new.write_text("plain output", encoding="utf-8")
                os.utime(old, (1_700_000_000, 1_700_000_000))
                os.utime(new, (1_700_100_000, 1_700_100_000))

                out = StringIO()
                with redirect_stdout(out):
                    main(["tool-results", "--limit", "5"])
                self.assertIn("Tool results: 2 saved", out.getvalue())
                self.assertIn(str(old), out.getvalue())

                out = StringIO()
                with redirect_stdout(out):
                    main(["tool-results", "--clean", "--keep-latest", "1", "--max-age-days", "10000", "--json"])
                result = json.loads(out.getvalue())
                self.assertTrue(result["dry_run"])
                self.assertEqual([row["path"] for row in result["would_delete"]], [str(old)])
                self.assertTrue(old.exists())
                self.assertTrue(new.exists())
            finally:
                tools.RESULTS_DIR = original_dir

    def test_cli_run_requires_sources_or_explicit_seed_corpus(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            err = StringIO()
            with self.assertRaises(SystemExit) as ctx, redirect_stderr(err):
                main(
                    [
                        "run",
                        "What should I build next?",
                        "--db",
                        str(root / "memory.sqlite"),
                        "--out",
                        str(root / "blocked"),
                        "--provider",
                        "local",
                        "--no-memory",
                    ]
                )
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn("No source material", err.getvalue())
            out = StringIO()
            with redirect_stdout(out):
                main(
                    [
                        "run",
                        "What should I build next?",
                        "--db",
                        str(root / "memory.sqlite"),
                        "--out",
                        str(root / "seeded"),
                        "--provider",
                        "local",
                        "--no-memory",
                        "--seed-corpus",
                    ]
                )
            self.assertIn("Run:", out.getvalue())
            payload = json.loads((root / "seeded" / "run.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["provenance"]["used_packaged_seed_corpus"])

    def test_cli_discover_seed_corpus_is_explicit(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = StringIO()
            with redirect_stdout(out):
                main(
                    [
                        "discover",
                        "--skill",
                        "ioi-circuit",
                        "--backend",
                        "synthetic",
                        "--db",
                        str(root / "memory.sqlite"),
                        "--out",
                        str(root / "discovery"),
                        "--provider",
                        "local",
                        "--no-memory",
                        "--max-rounds",
                        "1",
                        "--max-experiments",
                        "20",
                        "--seed-corpus",
                    ]
                )
            self.assertIn("Run:", out.getvalue())
            payload = json.loads((root / "discovery" / "run.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["provenance"]["allow_seed_corpus"])
            self.assertTrue(payload["provenance"]["used_packaged_seed_corpus"])

    def test_cli_run_demo_goal_and_discover_json_summaries(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "notes.md"
            source.write_text(
                "Induction heads copy previous tokens in simple sequence-completion tasks. "
                "Reliable claims need controls, reproducible probes, and explicit evidence.",
                encoding="utf-8",
            )

            commands = (
                (
                    [
                        "run",
                        "What evidence supports a small induction-head investigation?",
                        "--source",
                        str(source),
                        "--out",
                        str(root / "run"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--provider",
                        "local",
                        "--no-memory",
                        "--max-rounds",
                        "1",
                        "--json",
                    ],
                    "run",
                ),
                (
                    [
                        "demo",
                        "--out",
                        str(root / "demo"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--provider",
                        "local",
                        "--max-rounds",
                        "1",
                        "--json",
                    ],
                    "demo",
                ),
                (
                    [
                        "discover",
                        "--skill",
                        "ioi-circuit",
                        "--backend",
                        "synthetic",
                        "--db",
                        str(root / "memory.sqlite"),
                        "--out",
                        str(root / "discovery"),
                        "--provider",
                        "local",
                        "--no-memory",
                        "--max-rounds",
                        "1",
                        "--max-experiments",
                        "20",
                        "--json",
                    ],
                    "discover",
                ),
            )

            for args, command in commands:
                out = StringIO()
                with redirect_stdout(out):
                    main(args)
                payload = json.loads(out.getvalue())
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["command"], command)
                self.assertTrue(payload["run_id"])
                self.assertTrue(Path(payload["path"]).exists())
                self.assertIn("artifacts", payload)
                self.assertIn("summary", payload)

            goal_out = StringIO()
            with redirect_stdout(goal_out):
                main(
                    [
                        "goal",
                        "Make the induction-head investigation publishable.",
                        "--source",
                        str(source),
                        "--out",
                        str(root / "goal"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--provider",
                        "local",
                        "--no-memory",
                        "--max-iterations",
                        "1",
                        "--max-rounds",
                        "1",
                        "--json",
                    ]
                )
            goal_payload = json.loads(goal_out.getvalue())
            self.assertTrue(goal_payload["ok"])
            self.assertEqual(goal_payload["command"], "goal")
            self.assertTrue((root / "goal" / "goal.json").exists())
            self.assertEqual(len(goal_payload["iterations"]), 1)

    def test_cli_run_json_reports_recovery_for_missing_grounding(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = StringIO()
            with self.assertRaises(SystemExit) as ctx:
                with redirect_stdout(out):
                    main(
                        [
                            "run",
                            "What should I investigate?",
                            "--out",
                            str(root / "run"),
                            "--db",
                            str(root / "memory.sqlite"),
                            "--provider",
                            "local",
                            "--no-memory",
                            "--json",
                        ]
                    )
            self.assertEqual(ctx.exception.code, 2)
            payload = json.loads(out.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["command"], "run")
            self.assertIn("next_actions", payload)

    def test_cli_run_and_goal_json_report_missing_source_without_traceback(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_source = root / "notes"
            for command in ("run", "goal"):
                with self.subTest(command=command):
                    out = StringIO()
                    err = StringIO()
                    with self.assertRaises(SystemExit) as ctx:
                        with redirect_stdout(out), redirect_stderr(err):
                            main(
                                [
                                    command,
                                    "What should I investigate?",
                                    "--source",
                                    str(missing_source),
                                    "--out",
                                    str(root / command),
                                    "--db",
                                    str(root / "memory.sqlite"),
                                    "--provider",
                                    "local",
                                    "--no-memory",
                                    "--json",
                                ]
                            )
                    self.assertEqual(ctx.exception.code, 2)
                    payload = json.loads(out.getvalue())
                    self.assertFalse(payload["ok"])
                    self.assertEqual(payload["command"], command)
                    self.assertIn("Source path not found", payload["error"])
                    self.assertIn("next_actions", payload)
                    self.assertEqual(err.getvalue(), "")

    def test_cli_goal_seed_corpus_is_explicit(self):
        from mechferret.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = StringIO()
            with redirect_stdout(out):
                main(
                    [
                        "goal",
                        "Make this investigation publishable.",
                        "--seed-corpus",
                        "--out",
                        str(root / "goal"),
                        "--db",
                        str(root / "memory.sqlite"),
                        "--provider",
                        "local",
                        "--no-memory",
                        "--max-iterations",
                        "1",
                        "--max-rounds",
                        "1",
                        "--json",
                    ]
                )
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["command"], "goal")
            self.assertEqual(len(payload["iterations"]), 1)


if __name__ == "__main__":
    unittest.main()
