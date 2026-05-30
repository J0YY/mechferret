from __future__ import annotations

from pathlib import Path

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None


if modal is not None:
    image = modal.Image.debian_slim(python_version="3.12").pip_install("openai")
    app = modal.App("mechferret-autoresearch", image=image)

    @app.function(timeout=900, secrets=[modal.Secret.from_name("openai-api-key")])
    def run_remote(question: str, corpus: dict[str, str], max_rounds: int = 3) -> dict:
        from mechferret.controller import MechFerret

        work = Path("/tmp/mechferret")
        source_dir = work / "corpus"
        source_dir.mkdir(parents=True, exist_ok=True)
        for name, text in corpus.items():
            (source_dir / name).write_text(text, encoding="utf-8")
        engine = MechFerret(work / "memory.sqlite")
        run = engine.run(
            question,
            source_paths=[str(source_dir)],
            out_dir=work / "run",
            max_rounds=max_rounds,
            use_openai=True,
            include_memory=False,
        )
        return run.to_dict()

    @app.local_entrypoint()
    def main() -> None:
        corpus = {
            "hackathon.md": "# Demo\nAutoresearch systems need planning, retrieval, synthesis, and evaluation.",
            "reliability.md": "# Reliability\nCitation tracking and replayable traces make long-horizon agents debuggable.",
        }
        result = run_remote.remote("What should an autoresearch agent optimize for?", corpus)
        print(result["metrics"])

