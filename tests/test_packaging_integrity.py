import fnmatch
import tomllib
import unittest
from pathlib import Path


class PackagingIntegrityTest(unittest.TestCase):
    def test_public_package_metadata_is_release_ready(self):
        config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        project = config["project"]
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["license-files"], ["LICENSE"])
        license_text = Path("LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)
        self.assertIn("MechFerret contributors", license_text)
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("[MIT License](LICENSE)", readme)
        self.assertNotIn("License :: OSI Approved :: MIT License", project["classifiers"])
        self.assertIn("Topic :: Scientific/Engineering :: Artificial Intelligence", project["classifiers"])
        self.assertIn("Programming Language :: Python :: 3.11", project["classifiers"])
        self.assertIn("Programming Language :: Python :: 3.12", project["classifiers"])
        self.assertIn("Typing :: Typed", project["classifiers"])
        urls = project["urls"]
        self.assertTrue(urls["Homepage"].startswith("https://github.com/"))
        self.assertTrue(urls["Repository"].startswith("https://github.com/"))
        self.assertTrue(urls["Issues"].endswith("/issues"))

    def test_package_declares_pep561_typing_marker(self):
        config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        marker = Path("mechferret/py.typed")
        self.assertTrue(marker.is_file())
        self.assertEqual(marker.read_text(encoding="utf-8"), "\n")
        patterns = config["tool"]["setuptools"]["package-data"]["mechferret"]
        self.assertIn("py.typed", patterns)

    def test_citation_metadata_matches_project_metadata(self):
        config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        project = config["project"]
        citation = Path("CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("cff-version: 1.2.0", citation)
        self.assertIn('title: "MechFerret"', citation)
        self.assertIn(f'version: "{project["version"]}"', citation)
        self.assertIn(f'license: "{project["license"]}"', citation)
        self.assertIn(f'repository-code: "{project["urls"]["Repository"]}"', citation)
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("[CITATION.cff](CITATION.cff)", readme)

    def test_changelog_tracks_project_version_and_release_surface(self):
        config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        version = config["project"]["version"]
        changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn(f"## {version} - Unreleased", changelog)
        self.assertIn("Offline-first research loop", changelog)
        self.assertIn("CI gates", changelog)
        self.assertIn("Community health files", changelog)
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("[Changelog](CHANGELOG.md)", readme)

    def test_console_scripts_are_declared_and_smoked_in_ci(self):
        config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        scripts = config["project"]["scripts"]
        self.assertEqual(scripts["mechferret"], "mechferret.cli:main")
        self.assertEqual(scripts["mf"], "mechferret.cli:main")
        workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn('"$RUNNER_TEMP/wheel-venv/bin/mechferret" version --json | "$RUNNER_TEMP/wheel-venv/bin/python" -m json.tool', workflow)
        self.assertIn('"$RUNNER_TEMP/wheel-venv/bin/mf" commands --json | "$RUNNER_TEMP/wheel-venv/bin/python" -m json.tool', workflow)
        self.assertIn('"$RUNNER_TEMP/wheel-venv/bin/mf" commands --workflow --json | "$RUNNER_TEMP/wheel-venv/bin/python" -m json.tool', workflow)

    def test_ci_runs_release_hygiene_gates(self):
        workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("git diff --check", workflow)
        self.assertIn("term=$(printf '\\144\\145\\164\\145\\162\\155\\151\\156\\151\\163\\164\\151\\143')", workflow)
        self.assertIn('git grep -n -i "$term"', workflow)
        self.assertIn('placeholder_pattern="TODO: ""write|TODO: ""motivate|TODO: ""describe', workflow)
        self.assertIn('scaffold_pattern="structure-only local ""scaffold|local mode writes ""only"', workflow)
        self.assertIn('"$placeholder_pattern|$scaffold_pattern"', workflow)

    def test_ci_validates_generated_json_smoke_files(self):
        workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("python -m mechferret doctor --json > doctor.json", workflow)
        self.assertIn("python -m json.tool doctor.json > /dev/null", workflow)
        self.assertIn("python -m mechferret commands --workflow --json > workflows.json", workflow)
        self.assertIn("python -m json.tool workflows.json > /dev/null", workflow)
        self.assertIn("python -m mechferret quickstart --mode ci --json > quickstart.json", workflow)
        self.assertIn("python -m json.tool quickstart.json > /dev/null", workflow)
        self.assertIn("python -m mechferret selftest --json > selftest.json", workflow)
        self.assertIn("python -m json.tool selftest.json > /dev/null", workflow)
        self.assertIn('python -m mechferret support --report "$RUNNER_TEMP/support-selftest.json" --json > support.json', workflow)
        self.assertIn("python -m json.tool support.json > /dev/null", workflow)
        self.assertIn('python -m json.tool "$RUNNER_TEMP/support-selftest.json" > /dev/null', workflow)
        self.assertIn("python -m json.tool audit.json > /dev/null", workflow)

    def test_github_intake_templates_cover_core_workflows(self):
        bug = Path(".github/ISSUE_TEMPLATE/bug_report.yml").read_text(encoding="utf-8")
        feature = Path(".github/ISSUE_TEMPLATE/feature_request.yml").read_text(encoding="utf-8")
        config = Path(".github/ISSUE_TEMPLATE/config.yml").read_text(encoding="utf-8")
        pull_request = Path(".github/pull_request_template.md").read_text(encoding="utf-8")

        self.assertIn("doctor --strict", bug)
        self.assertIn("make support", bug)
        self.assertIn("runs/selftest/selftest.json", bug)
        self.assertIn("Diagnostics", bug)
        self.assertIn("Reproduction", bug)
        self.assertIn("Offline path", feature)
        self.assertIn("Artifacts and verification", feature)
        self.assertIn("CONTRIBUTING.md", config)
        self.assertIn("make check", pull_request)
        self.assertIn("make support", pull_request)
        self.assertIn("python3 -m unittest discover -s tests -q", pull_request)
        self.assertIn("python3 -m mechferret selftest --json", pull_request)
        self.assertIn("runs/selftest/selftest.json", pull_request)
        self.assertIn("CLI JSON contracts", pull_request)
        self.assertIn("Runtime assets are packageable", pull_request)

    def test_security_policy_covers_secret_and_artifact_risks(self):
        security = Path("SECURITY.md").read_text(encoding="utf-8")
        self.assertIn("Reporting a Vulnerability", security)
        self.assertIn("do not post secrets", security)
        self.assertIn("path traversal", security)
        self.assertIn("artifact tampering", security)
        self.assertIn("verify-bundle", security)

    def test_code_of_conduct_covers_public_collaboration_norms(self):
        conduct = Path("CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
        self.assertIn("Expected Behavior", conduct)
        self.assertIn("Unacceptable Behavior", conduct)
        self.assertIn("Protect private data", conduct)
        self.assertIn("Security Policy", conduct)

    def test_support_policy_routes_common_user_needs(self):
        support = Path("SUPPORT.md").read_text(encoding="utf-8")
        self.assertIn("Before Opening an Issue", support)
        self.assertIn("make support", support)
        self.assertIn("python3 -m mechferret support", support)
        self.assertIn("python3 -m mechferret doctor --strict", support)
        self.assertIn("python3 -m mechferret selftest --report runs/selftest/selftest.json", support)
        self.assertIn("python3 -m mechferret status --json", support)
        self.assertIn("credential values and environment values are omitted", support)
        self.assertIn("Where to Ask", support)
        self.assertIn("SECURITY.md", support)

    def test_gitignore_covers_generated_release_and_local_state(self):
        ignored = set(Path(".gitignore").read_text(encoding="utf-8").splitlines())
        for pattern in (
            ".mechferret/",
            "MECHFERRET.md",
            "runs/",
            "build/",
            "dist/",
            "*.egg-info/",
            "*.whl",
            ".venv/",
            "venv/",
            "__pycache__/",
            "*.py[cod]",
            ".coverage",
            "htmlcov/",
            ".env",
        ):
            self.assertIn(pattern, ignored)

    def test_editorconfig_covers_project_text_file_conventions(self):
        config = Path(".editorconfig").read_text(encoding="utf-8")
        self.assertIn("root = true", config)
        self.assertIn("end_of_line = lf", config)
        self.assertIn("insert_final_newline = true", config)
        self.assertIn("trim_trailing_whitespace = true", config)
        self.assertIn("[*.py]", config)
        self.assertIn("indent_size = 4", config)
        self.assertIn("[*.{md,yml,yaml,toml,json,cff}]", config)
        self.assertIn("indent_size = 2", config)
        self.assertIn("[Makefile]", config)
        self.assertIn("indent_style = tab", config)

    def test_makefile_exposes_one_command_release_loop(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")
        self.assertIn(".DEFAULT_GOAL := help", makefile)
        for target in (
            "help:",
            "docs:",
            "docs-check:",
            "workflows:",
            "quickstart:",
            "selftest:",
            "support:",
            "test:",
            "compile:",
            "doctor:",
            "workflows-json:",
            "quickstart-json:",
            "selftest-json:",
            "support-json:",
            "diff-check:",
            "language-scan:",
            "placeholder-scan:",
            "clean-bytecode:",
            "check:",
            "wheel:",
            "clean:",
        ):
            self.assertIn(target, makefile)
        self.assertIn("python3 -m mechferret commands --markdown --out docs/CLI.md", makefile)
        self.assertIn("python3 -m mechferret commands --examples --markdown --out docs/CLI_EXAMPLES.md", makefile)
        self.assertIn("python3 -m mechferret commands --workflow", makefile)
        self.assertIn("tests.test_docs_integrity.DocsIntegrityTest.test_cli_reference_is_generated_from_parser", makefile)
        self.assertIn("tests.test_docs_integrity.DocsIntegrityTest.test_cli_examples_reference_is_generated_from_parser", makefile)
        self.assertIn("python3 -m mechferret quickstart --run", makefile)
        self.assertIn("python3 -m mechferret selftest", makefile)
        self.assertIn("python3 -m mechferret support", makefile)
        self.assertIn("python3 -m unittest discover -s tests -q", makefile)
        self.assertIn("python3 -m compileall -q mechferret tests", makefile)
        self.assertIn("python3 -m mechferret doctor --strict", makefile)
        self.assertIn("python3 -m mechferret commands --workflow --json", makefile)
        self.assertIn("python3 -m mechferret quickstart --mode ci --json", makefile)
        self.assertIn("python3 -m mechferret selftest --json", makefile)
        self.assertIn("python3 -m mechferret support --report /tmp/mechferret-support.json --json", makefile)
        self.assertIn("python3 -m json.tool /tmp/mechferret-support.json", makefile)
        self.assertIn("git diff --check", makefile)
        self.assertIn("term=$$(printf '\\144\\145\\164\\145\\162\\155\\151\\156\\151\\163\\164\\151\\143')", makefile)
        self.assertIn("TODO: ''write|TODO: ''motivate|TODO: ''describe", makefile)
        self.assertIn("structure-only local ''scaffold|local mode writes ''only", makefile)
        self.assertIn("check: docs-check test compile doctor workflows-json quickstart-json selftest-json support-json diff-check language-scan placeholder-scan clean-bytecode", makefile)

    def test_contributing_points_to_makefile_release_loop(self):
        contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
        self.assertIn("make docs", contributing)
        self.assertIn("make quickstart", contributing)
        self.assertIn("make workflows", contributing)
        self.assertIn("make selftest", contributing)
        self.assertIn("make support", contributing)
        self.assertIn("make check", contributing)
        self.assertIn("make wheel", contributing)
        self.assertIn("docs/CLI.md", contributing)
        self.assertIn("docs/CLI_EXAMPLES.md", contributing)
        self.assertIn("python3 -m mechferret selftest --json", contributing)
        self.assertIn("python3 -m mechferret support --report /tmp/mechferret-support.json --json", contributing)
        self.assertIn("python3 -m unittest discover -s tests -q", contributing)
        self.assertIn("python3 -m pip wheel . -w /tmp/mechferret-wheels --no-deps", contributing)

    def test_runtime_assets_are_declared_as_package_data(self):
        config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        patterns = config["tool"]["setuptools"]["package-data"]["mechferret"]
        assets = _runtime_assets()
        self.assertGreaterEqual(len(assets), 15)
        missing = [
            asset
            for asset in assets
            if not any(fnmatch.fnmatch(asset, pattern) for pattern in patterns)
        ]
        self.assertEqual(missing, [])

    def test_package_data_patterns_match_existing_assets(self):
        config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        patterns = config["tool"]["setuptools"]["package-data"]["mechferret"]
        package_files = [
            path.relative_to("mechferret").as_posix()
            for path in Path("mechferret").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
        dead_patterns = [
            pattern
            for pattern in patterns
            if not any(fnmatch.fnmatch(path, pattern) for path in package_files)
        ]
        self.assertEqual(dead_patterns, [])

    def test_ci_wheel_smoke_checks_representative_runtime_assets(self):
        workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
        for asset in (
            "mechferret/py.typed",
            "mechferret/seed_corpus/hackathon_brief.md",
            "mechferret/skills/ioi-circuit.json",
            "mechferret/templates/openvla_sae/README.md",
            "mechferret/templates/openvla_sae/configs/phase1.yaml",
            "mechferret/templates/openvla_sae/paper/outline.md",
            "mechferret/templates/openvla_sae/scripts/phase1_commands.sh",
            "mechferret/templates/openvla_sae/src/simple_topk_sae.py",
        ):
            self.assertIn(asset, workflow)
        self.assertIn("License-Expression: MIT", workflow)
        self.assertIn("License-File: LICENSE", workflow)
        self.assertIn("Classifier: Typing :: Typed", workflow)
        self.assertIn("dist-info/licenses/LICENSE", workflow)


def _runtime_assets() -> list[str]:
    roots = [
        Path("mechferret/seed_corpus"),
        Path("mechferret/skills"),
        Path("mechferret/templates"),
    ]
    assets = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                assets.append(path.relative_to("mechferret").as_posix())
    return sorted(assets)


if __name__ == "__main__":
    unittest.main()
