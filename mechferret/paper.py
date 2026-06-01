from __future__ import annotations

import shutil
import subprocess
import json
import math
from pathlib import Path
from typing import Any

from .audit import latest_run_json, load_run_artifact
from .config import configured_api_key, configured_model, load_config
from .models import ResearchRun
from .provenance import refresh_run_manifest
from .text import compact_text


TECTONIC_TIMEOUT_SECONDS = 60


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _text(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _strings(value: Any) -> list[str]:
    return [text for item in _items(value) if (text := _text(item).strip())]


def _number(value: Any, default: float = 0.0) -> float:
    if type(value) is bool:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _positive_int(value: Any, default: int) -> int:
    if type(value) is bool:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _flag(value: Any) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _field(row: Any, name: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _rows(run: ResearchRun, name: str) -> list[Any]:
    return _items(_field(run, name, []))


def _path(value: Any) -> Path | None:
    if isinstance(value, (str, Path)):
        text = str(value)
        return Path(text) if text else None
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {_text(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) or value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else 0.0
    try:
        return str(value)
    except Exception:
        return ""


def _json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(_json_safe(value), allow_nan=False, **kwargs)


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_paper_from_artifact(
    run_json: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    compile_pdf: bool = False,
    compile_timeout: int = TECTONIC_TIMEOUT_SECONDS,
    provider: str = "auto",
    model: str | None = None,
) -> dict[str, Any]:
    target = _path(run_json) if run_json else latest_run_json()
    if target is None:
        raise FileNotFoundError("no run artifact found (expected runs/**/run.json)")
    run = load_run_artifact(target)
    out = _path(out_dir) if out_dir is not None else None
    if out is None:
        out = target.parent / "paper"
    out.mkdir(parents=True, exist_ok=True)
    tex = out / "main.tex"
    mode = "scaffold"
    latex = ""
    diagnostics: dict[str, str] = {}
    provider_name = _text(provider).strip().lower() or "auto"
    model_name = _text(model).strip() or None
    if provider_name != "local":
        latex = draft_latex_with_model(run, provider=provider_name, model=model_name, diagnostics=diagnostics)
        mode = "model" if latex else "scaffold"
    if "\\documentclass" not in latex:
        latex = latex_from_run(run)
    tex.write_text(latex, encoding="utf-8")
    result: dict[str, Any] = {"ok": True, "run_json": str(target), "tex": str(tex), "mode": mode}
    if diagnostics.get("note"):
        result["note"] = diagnostics["note"]
    if _flag(compile_pdf):
        result.update(compile_tex(tex, timeout=compile_timeout))
    _record_paper_artifacts(target, result)
    return result


def latex_from_run(run: ResearchRun) -> str:
    """Write an evidence-bound local LaTeX draft.

    Local mode may render fields already present in the run ledger, but it must
    not synthesize new claims, citations, controls, or conclusions. With a
    configured provider, model-authored text can expand this structure from the
    same evidence.
    """

    title = _tex_title(_field(run, "question") or "Mechanistic Interpretability Findings")
    run_id = _tex(_field(run, "run_id"))
    created = _tex(_field(run, "created_at"))
    question = _tex(_field(run, "question") or "Not recorded.")
    plan = _field(run, "plan", {})
    plan_strategy = _tex(_field(plan, "strategy") or _field(run, "answer") or "Not recorded.")
    metrics = _mapping(_field(run, "metrics", {}))
    sources = _rows(run, "sources")
    evidence = _rows(run, "evidence")
    claims = _paper_claim_rows(run)
    experiments = _paper_experiment_rows(run)
    discoveries = _paper_discovery_rows(run)
    gaps = _strings(_field(run, "gaps", []))
    contradictions = _rows(run, "contradictions")
    artifacts = _mapping(_field(run, "artifacts", {}))
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[margin=1in]{geometry}",
            "\\usepackage{booktabs}",
            "\\usepackage{hyperref}",
            "\\usepackage{array}",
            f"\\title{{{title}}}",
            "\\author{MechFerret}",
            "\\date{}",
            "\\begin{document}",
            "\\maketitle",
            "\\begin{abstract}",
            (
                "\\noindent This manuscript is populated from run "
                f"\\texttt{{{run_id or 'unknown'}}}. It preserves the recorded research question, "
                "experiment ledger, evidence ledger, and open gaps without adding claims outside the run artifact."
            ),
            "\\end{abstract}",
            _section(
                "Introduction",
                "\n".join(
                    [
                        f"\\paragraph{{Question.}} {question}",
                        f"\\paragraph{{Run.}} \\texttt{{{run_id or 'unknown'}}} generated at {created or 'unknown time'}.",
                        (
                            "\\paragraph{Ledger scope.} "
                            f"The run records {len(sources)} sources, {len(evidence)} evidence chunks, "
                            f"{len(claims)} claims, {len(experiments)} experiment rows, and {len(discoveries)} discovery rows."
                        ),
                    ]
                ),
            ),
            _section(
                "Method",
                "\n".join(
                    [
                        f"\\paragraph{{Recorded strategy.}} {plan_strategy}",
                        _plan_table(run),
                    ]
                ),
            ),
            _section(
                "Results",
                "\n".join(
                    [
                        _discovery_table(discoveries),
                        _claim_table(claims),
                    ]
                ),
            ),
            _section("Experiment Ledger", _experiment_table(experiments)),
            _section(
                "Evidence Ledger",
                "\n".join(
                    [
                        _evidence_table(evidence),
                        _artifact_table(artifacts),
                    ]
                ),
            ),
            _section(
                "Limitations",
                "\n".join(
                    [
                        _gap_list(gaps),
                        _contradiction_table(contradictions),
                        _metric_line(metrics),
                    ]
                ),
            ),
            _section(
                "Conclusion",
                "\\noindent The paper should only advance claims represented in the tables above; unresolved gaps remain part of the result.",
            ),
            "\\end{document}",
            "",
        ]
    )


def _paper_claim_rows(run: ResearchRun) -> list[Any]:
    return sorted(
        [row for row in _rows(run, "claims") if _text(_field(row, "id")) or _text(_field(row, "text"))],
        key=lambda row: _number(_field(row, "confidence")),
        reverse=True,
    )[:12]


def _paper_experiment_rows(run: ResearchRun) -> list[Any]:
    return sorted(
        [row for row in _rows(run, "experiments") if _text(_field(row, "id")) or _text(_field(row, "probe"))],
        key=lambda row: abs(_number(_field(row, "effect_size"))),
        reverse=True,
    )[:16]


def _paper_discovery_rows(run: ResearchRun) -> list[Any]:
    return sorted(
        [row for row in _rows(run, "discoveries") if _text(_field(row, "id")) or _text(_field(row, "statement"))],
        key=lambda row: _number(_field(row, "confidence")),
        reverse=True,
    )[:10]


def _plan_table(run: ResearchRun) -> str:
    plan = _field(run, "plan", {})
    steps = _items(_field(plan, "steps", []))
    rows = []
    for index, step in enumerate(steps[:12], start=1):
        rows.append([
            str(index),
            _short(_field(step, "intent") or _field(step, "id") or "step"),
            _short(_field(step, "question") or _field(run, "question")),
            _short(_field(step, "status") or "recorded"),
        ])
    if not rows:
        return "\\noindent No structured plan steps were recorded."
    return _latex_table(["Step", "Intent", "Question", "Status"], rows, widths=["0.08", "0.22", "0.50", "0.14"])


def _discovery_table(discoveries: list[Any]) -> str:
    if not discoveries:
        return "\\noindent No discovery rows were present in the run ledger."
    rows = [
        [
            _field(row, "id"),
            _short(_field(row, "statement"), 150),
            _score(_field(row, "confidence")),
            _score(_field(row, "effect_size")),
            _score(_field(row, "reproducibility")),
            ", ".join(_strings(_field(row, "supporting_experiments", []))) or "none",
        ]
        for row in discoveries
    ]
    return _latex_table(
        ["ID", "Statement", "Conf.", "Effect", "Reprod.", "Support"],
        rows,
        widths=["0.09", "0.42", "0.08", "0.08", "0.08", "0.17"],
    )


def _claim_table(claims: list[Any]) -> str:
    if not claims:
        return "\\noindent No claim rows were present in the run ledger."
    rows = [
        [
            _field(row, "id"),
            _short(_field(row, "text"), 160),
            _score(_field(row, "confidence")),
            _score(_field(row, "support_score")),
            ", ".join(_strings(_field(row, "citations", []))) or "none",
        ]
        for row in claims
    ]
    return _latex_table(
        ["ID", "Claim", "Conf.", "Support", "Citations"],
        rows,
        widths=["0.09", "0.50", "0.08", "0.09", "0.16"],
    )


def _experiment_table(experiments: list[Any]) -> str:
    if not experiments:
        return "\\noindent No experiment rows were present in the run ledger."
    rows = [
        [
            _field(row, "id"),
            _short(_field(row, "probe"), 46),
            _short(_target_label(_field(row, "target", {})), 90),
            _short(_field(row, "status"), 36),
            _score(_field(row, "effect_size")),
            _score(_field(row, "baseline")),
            _yes_no(_field(row, "reproduced")),
        ]
        for row in experiments
    ]
    return _latex_table(
        ["ID", "Probe", "Target", "Status", "Effect", "Base", "Reprod."],
        rows,
        widths=["0.08", "0.20", "0.25", "0.12", "0.08", "0.08", "0.10"],
    )


def _evidence_table(evidence: list[Any]) -> str:
    rows = [
        [
            _field(row, "id"),
            _field(row, "source_id"),
            _short(_field(row, "title") or "Untitled", 80),
            _short(_field(row, "text"), 180),
        ]
        for row in evidence[:12]
        if _text(_field(row, "id")) or _text(_field(row, "text"))
    ]
    if not rows:
        return "\\noindent No evidence chunk rows were present in the run ledger."
    return _latex_table(
        ["ID", "Source", "Title", "Excerpt"],
        rows,
        widths=["0.08", "0.10", "0.22", "0.50"],
    )


def _artifact_table(artifacts: dict[str, Any]) -> str:
    rows = [
        [_short(label, 60), _short(path, 130)]
        for label, path in sorted(artifacts.items(), key=lambda item: _text(item[0]))
        if _text(label).strip() and _text(path).strip()
    ][:16]
    if not rows:
        return "\\noindent No artifact paths were recorded."
    return _latex_table(["Artifact", "Path"], rows, widths=["0.25", "0.65"])


def _gap_list(gaps: list[str]) -> str:
    if not gaps:
        return "\\noindent No explicit gap rows were recorded."
    return "\\begin{itemize}\n" + "\n".join(f"\\item {_tex(_short(gap, 180))}" for gap in gaps[:10]) + "\n\\end{itemize}"


def _contradiction_table(contradictions: list[Any]) -> str:
    rows = [
        [
            _field(row, "id"),
            _short(_field(row, "claim_a"), 55),
            _short(_field(row, "claim_b"), 55),
            _short(_field(row, "reason"), 120),
            _score(_field(row, "severity")),
        ]
        for row in contradictions[:10]
        if _text(_field(row, "id")) or _text(_field(row, "reason"))
    ]
    if not rows:
        return "\\noindent No contradiction rows were recorded."
    return _latex_table(
        ["ID", "Claim A", "Claim B", "Reason", "Severity"],
        rows,
        widths=["0.08", "0.18", "0.18", "0.42", "0.08"],
    )


def _metric_line(metrics: dict[str, Any]) -> str:
    keys = ["readiness_score", "source_diversity", "citation_density", "plan_coverage", "contradiction_pressure"]
    parts = [f"{key.replace('_', ' ')}={_score(metrics.get(key))}" for key in keys if key in metrics]
    if not parts:
        return "\\noindent No readiness metrics were recorded."
    return "\\noindent Recorded readiness metrics: " + _tex("; ".join(parts)) + "."


def _latex_table(headers: list[str], rows: list[list[Any]], *, widths: list[str] | None = None) -> str:
    if not rows:
        return ""
    if widths and len(widths) == len(headers):
        spec = "@{}" + "".join(f"p{{{width}\\linewidth}}" for width in widths) + "@{}"
    else:
        spec = "@{}" + "l" * len(headers) + "@{}"
    rendered = [
        "\\begin{center}",
        "\\small",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(_tex(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [_tex(_short(cell, 220)) for cell in row[: len(headers)]]
        cells.extend([""] * (len(headers) - len(cells)))
        rendered.append(" & ".join(cells) + " \\\\")
    rendered.extend(["\\bottomrule", "\\end{tabular}", "\\end{center}"])
    return "\n".join(rendered)


def _short(value: Any, limit: int = 120) -> str:
    text = " ".join(_text(value).split())
    if not text:
        return ""
    return compact_text(text, max(20, limit))


def _score(value: Any) -> str:
    return f"{_number(value):.2f}"


def _yes_no(value: Any) -> str:
    return "yes" if _flag(value) else "no"


def _target_label(value: Any) -> str:
    payload = _json_safe(value)
    if isinstance(payload, dict) and payload:
        return _json_dumps(payload, sort_keys=True)
    return _text(payload)


def compile_tex(tex: str | Path, *, timeout: int = TECTONIC_TIMEOUT_SECONDS) -> dict[str, Any]:
    path = _path(tex)
    if path is None:
        return {"pdf": "", "compiled": False, "note": "invalid TeX path"}
    if not shutil.which("tectonic"):
        return {"pdf": "", "compiled": False, "note": "tectonic is not installed"}
    timeout_seconds = _positive_int(timeout, TECTONIC_TIMEOUT_SECONDS)
    try:
        proc = subprocess.run(
            ["tectonic", path.name],
            cwd=str(path.parent),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "pdf": "",
            "compiled": False,
            "note": f"tectonic timed out after {timeout_seconds}s",
            "stderr": _text(exc.stderr)[-2000:],
        }
    pdf = path.with_suffix(".pdf")
    return {
        "pdf": str(pdf) if proc.returncode == 0 and pdf.exists() else "",
        "compiled": proc.returncode == 0 and pdf.exists(),
        "stderr": proc.stderr[-2000:],
    }


def _record_paper_artifacts(run_json: Path, result: dict[str, Any]) -> None:
    payload = _load_payload(run_json)
    artifacts = _mapping(payload.get("artifacts"))
    payload["artifacts"] = artifacts
    artifacts["paper"] = _text(result.get("tex"))
    if result.get("pdf"):
        artifacts["pdf"] = _text(result.get("pdf"))
    result["artifacts"] = {"paper": _text(result.get("tex")), **({"pdf": _text(result.get("pdf"))} if result.get("pdf") else {})}
    run_json.write_text(_json_dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    refresh_run_manifest(run_json)


def _record_review_artifact(paper_tex: Path, review_path: Path, result: dict[str, Any]) -> None:
    result["artifacts"] = {"review": str(review_path)}
    run_json = _run_json_for_paper(paper_tex)
    if run_json is None:
        return
    payload = _load_payload(run_json)
    artifacts = _mapping(payload.get("artifacts"))
    payload["artifacts"] = artifacts
    artifacts["review"] = str(review_path)
    run_json.write_text(_json_dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    refresh_run_manifest(run_json)


def _run_json_for_paper(paper_tex: Path) -> Path | None:
    path = paper_tex.resolve()
    if path.name == "main.tex" and path.parent.name == "paper":
        candidate = path.parent.parent / "run.json"
        if candidate.exists():
            return candidate
    return None


def print_paper_result(result: dict[str, Any]) -> None:
    print(f"Paper: {_text(result.get('tex'))}")
    print(f"Source run: {_text(result.get('run_json'))}")
    print(f"Mode: {result.get('mode', 'unknown')}")
    if result.get("note"):
        print(f"Note: {result['note']}")
    if "compiled" in result:
        if result["compiled"]:
            print(f"PDF: {result['pdf']}")
        else:
            print(f"PDF: not built ({result.get('note') or 'tectonic failed'})")


def review_paper(
    paper_tex: str | Path | None = None,
    *,
    provider: str = "auto",
    model: str | None = None,
    out_dir: str | Path | None = None,
    runs_root: str | Path = "runs",
    selection: str = "latest",
) -> dict[str, Any]:
    provider_name = _text(provider).strip().lower() or "auto"
    model_name = _text(model).strip() or None
    selection_name = _text(selection).strip().lower() or "latest"
    path = _resolve_paper_path(paper_tex, runs_root=runs_root, selection=selection_name)
    if path is None or not path.exists():
        result: dict[str, Any] = {
            "ok": False,
            "error": "paper artifact not found",
            "path": str(path) if path else "",
            "review": "",
            "runs_root": str(_path(runs_root) or Path("runs")),
            "selection": selection_name,
            "next_actions": ["Run `mechferret paper <run.json>` first, then review the run-bound paper."],
        }
        if paper_tex:
            result["requested_path"] = _text(paper_tex)
        else:
            result.update(_missing_paper_resolution(runs_root=runs_root, selection=selection_name))
        return result
    selected, selected_model, key = _paper_provider(provider_name, model_name)
    if not selected or not key:
        return {
            "ok": False,
            "path": str(path),
            "review": "",
            "next_actions": ["Run `mechferret login openai` or `mechferret login anthropic`, or pass --provider/--model."],
        }
    prompt = _review_prompt(path.read_text(encoding="utf-8", errors="ignore"))
    try:
        if selected == "anthropic":
            review = _call_anthropic(selected_model, key, prompt)
        elif selected == "openai":
            review = _call_openai(selected_model, key, prompt)
        else:
            review = ""
    except Exception as exc:  # noqa: BLE001 - provider failures should be actionable
        return {
            "ok": False,
            "path": str(path),
            "provider": selected,
            "model": selected_model,
            "review": "",
            "next_actions": [f"{selected} review failed: {_text(exc)[:180]}"],
        }
    review = _text(review).strip()
    result: dict[str, Any] = {
        "ok": bool(review),
        "path": str(path),
        "provider": selected,
        "model": selected_model,
        "review": review,
        "next_actions": [] if review else ["Provider returned an empty review; retry or use a different model."],
    }
    if review:
        target_dir = _path(out_dir) if out_dir is not None else None
        if target_dir is None:
            target_dir = path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        review_path = target_dir / "review.md"
        review_path.write_text(review + "\n", encoding="utf-8")
        result["review_path"] = str(review_path)
        _record_review_artifact(path, review_path, result)
    return result


def print_review_result(result: dict[str, Any]) -> None:
    print(f"Paper: {result.get('path', '') or 'missing'}")
    if result.get("provider"):
        print(f"Reviewer: {result['provider']} / {result.get('model', '')}")
    if result.get("review_path"):
        print(f"Review: {result['review_path']}")
    if result.get("review"):
        print()
        print(result["review"])
    if result.get("next_actions"):
        print("Next actions:")
        for action in result["next_actions"]:
            print(f"  - {action}")


def draft_latex_with_model(
    run: ResearchRun,
    *,
    provider: str = "auto",
    model: str | None = None,
    diagnostics: dict[str, str] | None = None,
) -> str:
    selected, selected_model, key = _paper_provider(provider, model)
    if not selected or not key:
        if diagnostics is not None:
            diagnostics["note"] = "no configured provider; wrote scaffold"
        return ""
    prompt = _paper_prompt(run)
    try:
        if selected == "anthropic":
            return _strip_code_fences(_call_anthropic(selected_model, key, prompt))
        if selected == "openai":
            return _strip_code_fences(_call_openai(selected_model, key, prompt))
    except Exception as exc:  # noqa: BLE001 - paper generation falls back to scaffold
        if diagnostics is not None:
            diagnostics["note"] = f"{selected} draft failed: {_text(exc)[:160]}; wrote scaffold"
        return ""
    return ""


def _resolve_paper_path(paper_tex: str | Path | None, *, runs_root: str | Path = "runs", selection: str = "latest") -> Path | None:
    if paper_tex:
        return _path(paper_tex)
    from .ops import resolve_artifact

    root = _path(runs_root) or Path("runs")
    resolved = resolve_artifact("paper", runs_root=root, selection=_text(selection).strip().lower() or "latest")
    if resolved.get("exists") and resolved.get("path"):
        return _path(resolved["path"])
    return None


def _missing_paper_resolution(*, runs_root: str | Path = "runs", selection: str = "latest") -> dict[str, Any]:
    from .ops import resolve_artifact

    root = _path(runs_root) or Path("runs")
    resolved = resolve_artifact("paper", runs_root=root, selection=_text(selection).strip().lower() or "latest")
    return {
        "target": resolved.get("target", "paper"),
        "reason": resolved.get("reason", ""),
        "selected_run": resolved.get("selected_run", ""),
        "next_actions": resolved.get("next_actions")
        or ["Run `mechferret paper <run.json>` first, then review the run-bound paper."],
    }


def _review_prompt(latex: str) -> str:
    latex = _text(latex)
    return (
        "You are a senior mechanistic-interpretability reviewer. Review this LaTeX paper rigorously. "
        "Give integer scores from 1-10 for Soundness, Novelty, Clarity, Significance, and Overall. "
        "Then give a recommendation: Accept, Borderline, or Reject. "
        "List exactly three strengths, exactly three weaknesses, required experiments/controls, and the top revision priority. "
        "Do not invent evidence that is not in the paper. Plain text only.\n\n"
        f"{latex[:24000]}"
    )


def _section(title: str, body: str) -> str:
    return f"\\section{{{_tex(title)}}}\n{body}"


def _tex_title(text: Any) -> str:
    cleaned = _text(text).strip().rstrip("?")
    if len(cleaned) > 90:
        cleaned = compact_text(cleaned, 90)
    return _tex(cleaned[:1].upper() + cleaned[1:])


def _tex(text: Any) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in _text(text))


def _paper_provider(provider: str, model: str | None) -> tuple[str, str, str]:
    config = load_config()
    provider = _text(provider).strip().lower() or "auto"
    selected = _text(config.default_provider).strip().lower() if provider == "auto" else provider
    if selected not in {"anthropic", "openai"}:
        return "", "", ""
    key = configured_api_key(selected, config)
    if not key:
        return "", "", ""
    selected_model = configured_model(selected, config, _text(model).strip() or None)
    if not selected_model:
        return "", "", ""
    return selected, selected_model, key


def _paper_prompt(run: ResearchRun) -> str:
    claims = sorted(_rows(run, "claims"), key=lambda item: _number(_field(item, "confidence")), reverse=True)[:14]
    discoveries = sorted(_rows(run, "discoveries"), key=lambda item: _number(_field(item, "confidence")), reverse=True)[:10]
    ran_experiments = [
        row for row in sorted(_rows(run, "experiments"), key=lambda item: abs(_number(_field(item, "effect_size"))), reverse=True)
        if _text(_field(row, "status")) == "ran"
    ][:24]
    evidence = {
        "run_id": _text(_field(run, "run_id")),
        "question": _text(_field(run, "question")),
        "metrics": _mapping(_field(run, "metrics", {})),
        "gaps": _strings(_field(run, "gaps", [])),
        "claims": [
            {
                "id": _text(_field(c, "id")),
                "text": _text(_field(c, "text")),
                "citations": _strings(_field(c, "citations", [])),
                "confidence": _number(_field(c, "confidence")),
                "flags": _strings(_field(c, "quality_flags", [])),
            }
            for c in claims
            if _text(_field(c, "id")) or _text(_field(c, "text"))
        ],
        "discoveries": [
            {
                "id": _text(_field(d, "id")),
                "statement": _text(_field(d, "statement")),
                "confidence": _number(_field(d, "confidence")),
                "effect_size": _number(_field(d, "effect_size")),
                "reproducibility": _number(_field(d, "reproducibility")),
                "novelty": _number(_field(d, "novelty")),
                "supporting_experiments": _strings(_field(d, "supporting_experiments", [])),
            }
            for d in discoveries
            if _text(_field(d, "id")) or _text(_field(d, "statement"))
        ],
        "experiments": [
            {
                "id": _text(_field(e, "id")),
                "probe": _text(_field(e, "probe")),
                "target": _mapping(_field(e, "target", {})),
                "effect_size": _number(_field(e, "effect_size")),
                "baseline": _number(_field(e, "baseline")),
                "per_seed": [_number(item) for item in _items(_field(e, "per_seed", []))],
                "significant": _flag(_field(e, "significant")),
                "reproduced": _flag(_field(e, "reproduced")),
                "backend": _text(_field(e, "backend_used")),
            }
            for e in ran_experiments
        ],
    }
    scaffold = latex_from_run(run)
    return (
        "Write the paper prose for this mechanistic-interpretability dossier as complete LaTeX.\n"
        "Keep the same high-level structure as the scaffold: title, abstract, Introduction, Method, "
        "Results, Experiment Ledger, Evidence Ledger, Limitations, Conclusion.\n"
        "Do not invent experiments, citations, effect sizes, seeds, baselines, or claims. If evidence is weak, say so. "
        "Use only article/booktabs/hyperref/array/geometry-compatible LaTeX. Output ONLY the LaTeX source.\n\n"
        "RUN EVIDENCE JSON:\n"
        f"{_json_dumps(evidence, indent=2, sort_keys=True)[:18000]}\n\n"
        "STRUCTURE SCAFFOLD:\n"
        f"{scaffold[:12000]}"
    )


def _call_anthropic(model: str, key: str, prompt: str) -> str:
    from .agent import _extract_anthropic_content, _extract_provider_text, _http_post

    data = _http_post(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        },
        {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    content, error = _extract_anthropic_content(data)
    if error:
        raise RuntimeError(f"provider response envelope: {error}")
    return _extract_provider_text(content)


def _call_openai(model: str, key: str, prompt: str) -> str:
    from .agent import _extract_openai_message, _extract_provider_text, _http_post

    data = _http_post(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "You write rigorous, evidence-grounded LaTeX papers."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": 4096,
        },
        {"authorization": f"Bearer {key}", "content-type": "application/json"},
    )
    message, error = _extract_openai_message(data)
    if error:
        raise RuntimeError(f"provider response envelope: {error}")
    return _extract_provider_text(message.get("content"))


def _strip_code_fences(text: str) -> str:
    import re

    match = re.search(r"```(?:latex|tex)?\n(.*?)```", text, re.S)
    return (match.group(1) if match else text).strip()
