.DEFAULT_GOAL := help

.PHONY: help docs docs-check workflows next quickstart selftest support test compile doctor workflows-json next-json quickstart-json selftest-json support-json diff-check language-scan placeholder-scan clean-bytecode check wheel clean

help:
	@printf '%s\n' 'MechFerret contributor commands:'
	@printf '%s\n' '  make docs             Regenerate CLI reference docs'
	@printf '%s\n' '  make workflows        List runnable workflow recipes'
	@printf '%s\n' '  make next             Print the next recommended project actions'
	@printf '%s\n' '  make quickstart       Create a local demo dossier'
	@printf '%s\n' '  make selftest         Run fast local readiness checks'
	@printf '%s\n' '  make support          Write a shareable self-test report'
	@printf '%s\n' '  make test             Run the unit suite'
	@printf '%s\n' '  make check            Run local release gates'
	@printf '%s\n' '  make wheel            Build a no-dependency wheel into /tmp/mechferret-wheels'
	@printf '%s\n' '  make clean            Remove local generated caches and build outputs'

docs:
	python3 -m mechferret commands --markdown --out docs/CLI.md
	python3 -m mechferret commands --examples --markdown --out docs/CLI_EXAMPLES.md

docs-check:
	python3 -m unittest tests.test_docs_integrity.DocsIntegrityTest.test_cli_reference_is_generated_from_parser tests.test_docs_integrity.DocsIntegrityTest.test_cli_examples_reference_is_generated_from_parser -q

workflows:
	python3 -m mechferret commands --workflow

next:
	python3 -m mechferret next

quickstart:
	python3 -m mechferret quickstart --run

selftest:
	python3 -m mechferret selftest

support:
	python3 -m mechferret support

test:
	python3 -m unittest discover -s tests -q

compile:
	python3 -m compileall -q mechferret tests

doctor:
	python3 -m mechferret doctor --strict

workflows-json:
	python3 -m mechferret commands --workflow --json | python3 -m json.tool > /dev/null

next-json:
	python3 -m mechferret next --json | python3 -m json.tool > /dev/null

quickstart-json:
	python3 -m mechferret quickstart --mode ci --json | python3 -m json.tool > /dev/null

selftest-json:
	python3 -m mechferret selftest --json | python3 -m json.tool > /dev/null

support-json:
	python3 -m mechferret support --report /tmp/mechferret-support.json --json | python3 -m json.tool > /dev/null
	python3 -m json.tool /tmp/mechferret-support.json > /dev/null

diff-check:
	git diff --check

language-scan:
	@term=$$(printf '\144\145\164\145\162\155\151\156\151\163\164\151\143'); \
	if rg -n -i "$$term" --glob '!__pycache__/**' --glob '!.git/**' .; then exit 20; fi

placeholder-scan:
	@placeholder_pattern='TODO: ''write|TODO: ''motivate|TODO: ''describe|TODO: ''report|TODO: ''cite|TODO: ''list|TODO: ''summarize'; \
	scaffold_pattern='structure-only local ''scaffold|local mode writes ''only'; \
	if rg -n "$$placeholder_pattern|$$scaffold_pattern" mechferret tests README.md docs projects/openvla_sae .github SECURITY.md CONTRIBUTING.md CODE_OF_CONDUCT.md SUPPORT.md CITATION.cff CHANGELOG.md .editorconfig Makefile; then exit 21; fi

clean-bytecode:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

check: docs-check test compile doctor workflows-json next-json quickstart-json selftest-json support-json diff-check language-scan placeholder-scan clean-bytecode

wheel:
	python3 -m pip wheel . -w /tmp/mechferret-wheels --no-deps

clean:
	rm -rf build dist *.egg-info .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
