import re
import shlex
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


class DocsIntegrityTest(unittest.TestCase):
    def test_markdown_local_links_resolve(self):
        missing = []
        for md_path in _markdown_files():
            text = md_path.read_text(encoding="utf-8")
            for target in _markdown_targets(text):
                cleaned = target.split("#", 1)[0]
                if not cleaned or "://" in cleaned or cleaned.startswith("mailto:"):
                    continue
                if not (md_path.parent / cleaned).exists():
                    missing.append(f"{md_path}:{target}")
        self.assertEqual(missing, [])

    def test_readme_images_have_exported_assets(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        images = [target for target in _markdown_targets(readme, images_only=True)]
        self.assertGreaterEqual(len(images), 3)
        missing = [target for target in images if not (Path("README.md").parent / target).exists()]
        self.assertEqual(missing, [])

    def test_readme_links_top_level_markdown_docs(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        targets = set(_markdown_targets(readme))
        missing = [
            path.as_posix()
            for path in sorted(Path("docs").glob("*.md"))
            if path.as_posix() not in targets
        ]
        self.assertEqual(missing, [])

    def test_readme_links_contributor_and_license_files(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        targets = set(_markdown_targets(readme))
        self.assertIn("CONTRIBUTING.md", targets)
        self.assertIn("CODE_OF_CONDUCT.md", targets)
        self.assertIn("SUPPORT.md", targets)
        self.assertIn("SECURITY.md", targets)
        self.assertIn("CITATION.cff", targets)
        self.assertIn("CHANGELOG.md", targets)
        self.assertIn("LICENSE", targets)

    def test_readme_starts_with_short_offline_first_run_path(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertEqual(readme.count("\n## Quickstart\n"), 1)
        quickstart = readme.split("## Quickstart", 1)[1].split("For an OpenVLA", 1)[0]
        first_block = re.search(r"```bash\n(.*?)\n```", quickstart, re.S)
        self.assertIsNotNone(first_block)
        first_commands = [
            line.strip()
            for line in first_block.group(1).splitlines()
            if line.strip().startswith("python3 -m mechferret")
        ]
        self.assertEqual(
            first_commands,
            [
                "python3 -m mechferret init",
                "python3 -m mechferret quickstart --run",
                "python3 -m mechferret status",
                "python3 -m mechferret support",
            ],
        )
        self.assertIn("After `pipx install .`, replace", quickstart)
        self.assertIn("Copy this path first", quickstart)
        self.assertIn("docs/CLI.md", quickstart)
        self.assertIn("docs/CLI_EXAMPLES.md", quickstart)
        self.assertIn("python3 -m mechferret commands --workflow", quickstart)
        self.assertIn("python3 -m mechferret commands --workflow publish_dossier", quickstart)

    def test_workflow_recipe_commands_are_documented(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        support = Path("SUPPORT.md").read_text(encoding="utf-8")
        contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
        self.assertIn("python3 -m mechferret commands --workflow", readme)
        self.assertIn("python3 -m mechferret commands --workflow publish_dossier", readme)
        self.assertIn("python3 -m mechferret commands --workflow support_report", support)
        self.assertIn("make workflows", contributing)

    def test_demo_script_bare_command_matches_cli_default(self):
        script = Path("docs/DEMO_SCRIPT.md").read_text(encoding="utf-8")
        self.assertIn("bare command = interactive prompt", script)
        self.assertNotIn("bare command = headline discovery", script)

    def test_documented_mechferret_commands_parse(self):
        from mechferret.cli import build_parser

        parser = build_parser()
        failures = []
        commands = _documented_mechferret_commands()
        self.assertGreaterEqual(len(commands), 20)
        for source, line_number, command in commands:
            with redirect_stderr(StringIO()):
                try:
                    parser.parse_args(_cli_args(command))
                except SystemExit as exc:
                    failures.append(f"{source}:{line_number}: {command} exited {exc.code}")
        self.assertEqual(failures, [])

    def test_documented_discovery_examples_do_not_imply_model_defaults(self):
        missing_model = []
        for source, line_number, command in _documented_mechferret_commands():
            args = _cli_args(command)
            if "discover" not in args:
                continue
            if "--skill" not in args and "--task" not in args:
                continue
            if "--model" not in args:
                missing_model.append(f"{source}:{line_number}: {command}")
        self.assertEqual(missing_model, [])

    def test_cli_reference_is_generated_from_parser(self):
        from mechferret.cli import _command_index_payload, _command_markdown, build_parser

        expected = _command_markdown(_command_index_payload(build_parser()))
        actual = Path("docs/CLI.md").read_text(encoding="utf-8")
        self.assertEqual(actual, expected)
        self.assertIn("### `run`", actual)
        self.assertIn("Usage:\n\n```text\nusage: mechferret run", actual)
        self.assertIn("- `--source`: File or directory of seed documents.", actual)
        self.assertIn("### `verify-bundle`", actual)

    def test_cli_examples_reference_is_generated_from_parser(self):
        from mechferret.cli import _command_examples_payload, _command_index_payload, _command_markdown, build_parser

        expected = _command_markdown(_command_examples_payload(_command_index_payload(build_parser())))
        actual = Path("docs/CLI_EXAMPLES.md").read_text(encoding="utf-8")
        self.assertEqual(actual, expected)


def _markdown_targets(text: str, *, images_only: bool = False) -> list[str]:
    prefix = r"!" if images_only else r"!?"
    pattern = re.compile(prefix + r"\[[^\]]*\]\(([^)]+)\)")
    return [match.group(1).strip() for match in pattern.finditer(text)]


def _markdown_files() -> list[Path]:
    community = [
        Path("README.md"),
        Path("CONTRIBUTING.md"),
        Path("CODE_OF_CONDUCT.md"),
        Path("SECURITY.md"),
        Path("SUPPORT.md"),
        Path("CHANGELOG.md"),
    ]
    return [*community, *Path("docs").rglob("*.md")]


def _documented_mechferret_commands() -> list[tuple[str, int, str]]:
    commands = []
    for md_path in _markdown_files():
        for line_number, line in enumerate(md_path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if _looks_like_mechferret_command(stripped):
                commands.append((md_path.as_posix(), line_number, stripped.split("#", 1)[0].strip()))
            if stripped.startswith(("- `", "* `")):
                for command in _backticked_mechferret_commands(stripped):
                    commands.append((md_path.as_posix(), line_number, command))
    return commands


def _backticked_mechferret_commands(line: str) -> list[str]:
    commands = []
    for match in re.finditer(r"`([^`]*(?:python3 -m mechferret|mechferret)[^`]*)`", line):
        command = match.group(1).strip()
        if _looks_like_mechferret_command(command):
            commands.append(command)
    return commands


def _looks_like_mechferret_command(value: str) -> bool:
    return (
        value.startswith("python3 -m mechferret")
        or value == "mechferret"
        or value.startswith("mechferret ")
    )


def _cli_args(command: str) -> list[str]:
    parts = shlex.split(command)
    if parts[:3] == ["python3", "-m", "mechferret"]:
        return parts[3:]
    if parts[:1] == ["mechferret"]:
        return parts[1:]
    raise AssertionError(f"unexpected command format: {command}")


if __name__ == "__main__":
    unittest.main()
