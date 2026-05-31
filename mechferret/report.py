from __future__ import annotations

import html
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .agents import Synthesizer
from .models import ResearchRun
from .provenance import write_run_manifest
from .text import compact_text, domain


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


def _flag(value: Any) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _number(value: Any, default: float = 0.0) -> float:
    if type(value) is bool:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _number_label(value: Any, digits: int = 2) -> str:
    return f"{_number(value):.{digits}f}"


def _ratio(value: Any) -> float:
    return max(0.0, min(1.0, _number(value)))


def _field(row: Any, name: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _rows(run: ResearchRun, name: str) -> list[Any]:
    return _items(_field(run, name, []))


def _record_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return {}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
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


def _display(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value, sort_keys=True)
    return _text(value)


def _jsonable(value: Any) -> str:
    return _json_dumps(value, sort_keys=True)


def write_artifacts(run: ResearchRun, out_dir: str | Path) -> dict[str, str]:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "run.json"
    md_path = path / "report.md"
    html_path = path / "report.html"
    graph_path = path / "graph.json"
    evals_path = path / "evals.json"
    manifest_path = path / "manifest.json"
    artifacts = {
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
        "graph": str(graph_path),
        "evals": str(evals_path),
        "trace": str(path / "trace.jsonl"),
        "manifest": str(manifest_path),
    }
    has_sidecars = bool(_rows(run, "experiments") or _rows(run, "discoveries") or _rows(run, "hypotheses"))
    if has_sidecars:
        experiments_path = path / "experiments.json"
        discoveries_path = path / "discoveries.json"
        artifacts["experiments"] = str(experiments_path)
        artifacts["discoveries"] = str(discoveries_path)

    if not isinstance(_field(run, "artifacts", None), dict):
        run.artifacts = {}
    run.artifacts.update(artifacts)
    json_path.write_text(_json_dumps(run.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown_report(run), encoding="utf-8")
    html_path.write_text(html_report(run), encoding="utf-8")
    graph_path.write_text(_json_dumps(claim_graph(run), indent=2, sort_keys=True), encoding="utf-8")
    evals_path.write_text(_json_dumps(run_evals(run), indent=2, sort_keys=True), encoding="utf-8")
    if has_sidecars:
        experiments_path.write_text(
            _json_dumps([_record_dict(e) for e in _rows(run, "experiments")], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        discoveries_path.write_text(
            _json_dumps(
                {
                    "run_id": _text(_field(run, "run_id")),
                    "question": _text(_field(run, "question")),
                    "discoveries": [_record_dict(d) for d in _rows(run, "discoveries")],
                    "hypotheses": [_record_dict(h) for h in _rows(run, "hypotheses")],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    write_run_manifest(run, path)
    json_path.write_text(_json_dumps(run.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def markdown_report(run: ResearchRun) -> str:
    labels = Synthesizer().citation_labels(_rows(run, "evidence"))
    synthesis_source = synthesis_source_label(run)
    metrics = _mapping(_field(run, "metrics", {}))
    run_id = _text(_field(run, "run_id"))
    lines = [
        f"# MechFerret Research Dossier",
        "",
        f"**Question:** {_text(_field(run, 'question'))}",
        f"**Run:** {run_id}",
        f"**Readiness score:** {_number_label(metrics.get('readiness_score', 0))}",
        f"**Synthesis source:** {synthesis_source}",
        "",
        "## Synthesis",
        "",
        _text(_field(run, "answer")),
        "",
    ]
    discoveries = _rows(run, "discoveries")
    if discoveries:
        lines.extend(["## Confirmed Mechanisms", ""])
        for d in discoveries:
            supporting = _strings(_field(d, "supporting_experiments", []))
            lines.append(
                f"- **{_text(_field(d, 'statement'))}** (confidence={_number_label(_field(d, 'confidence'))}, "
                f"effect={_number_label(_field(d, 'effect_size'))}, "
                f"reproducibility={_number_label(_field(d, 'reproducibility'))}, "
                f"novelty={_number_label(_field(d, 'novelty'))}, experiments={len(supporting)})"
            )
        lines.append("")
    hypotheses = _rows(run, "hypotheses")
    if hypotheses:
        lines.extend(["## Hypotheses", ""])
        for h in hypotheses:
            lines.append(
                f"- [{_text(_field(h, 'status'))}] `{_text(_field(h, 'id'))}` "
                f"{_text(_field(h, 'statement'))} (confidence={_number_label(_field(h, 'confidence'))})"
            )
        lines.append("")
    experiments = _rows(run, "experiments")
    if experiments:
        ran = [e for e in experiments if _text(_field(e, "status")) == "ran"]
        lines.extend(["## Experiment Ledger", "", f"{len(ran)} experiments ran. Significant + reproducible shown:", ""])
        for e in sorted(ran, key=lambda x: abs(_number(_field(x, "effect_size"))), reverse=True):
            if not (_flag(_field(e, "significant")) and _flag(_field(e, "reproduced"))):
                continue
            lines.append(
                f"- `{_text(_field(e, 'probe'))}` {_jsonable(_field(e, 'target', {}))} "
                f"effect={_number(_field(e, 'effect_size')):+.3f} "
                f"control={_number(_field(e, 'baseline')):+.3f} "
                f"seeds={_jsonable(_field(e, 'per_seed', []))} backend={_text(_field(e, 'backend_used'))}"
            )
        lines.append("")
    lines.extend(["## Metrics", ""])
    for key, value in sorted(metrics.items(), key=lambda item: _text(item[0])):
        lines.append(f"- **{_text(key)}:** {_display(value)}")
    lines.extend(["", "## Claims", ""])
    for claim in _rows(run, "claims"):
        cites = ", ".join(labels.get(cid, cid) for cid in _strings(_field(claim, "citations", [])))
        flags_list = _strings(_field(claim, "quality_flags", []))
        flags = f" flags={','.join(flags_list)}" if flags_list else ""
        lines.append(
            f"- `{_text(_field(claim, 'id'))}` {_text(_field(claim, 'text'))} [{cites}] "
            f"confidence={_number_label(_field(claim, 'confidence'))}{flags}"
        )
    lines.extend(["", "## Evidence Ledger", ""])
    for chunk in _rows(run, "evidence"):
        chunk_id = _text(_field(chunk, "id"))
        label = labels.get(chunk_id, chunk_id)
        url = _text(_field(chunk, "url")) or "local"
        lines.append(
            f"- **{label}** {_text(_field(chunk, 'title'))} ({url}) "
            f"score={_number_label(_field(chunk, 'score'))}: {compact_text(_field(chunk, 'text'), 260)}"
        )
    contradictions = _rows(run, "contradictions")
    if contradictions:
        lines.extend(["", "## Contradictions", ""])
        for contradiction in contradictions:
            lines.append(
                f"- `{_text(_field(contradiction, 'id'))}` {_text(_field(contradiction, 'claim_a'))} "
                f"vs {_text(_field(contradiction, 'claim_b'))}: {_text(_field(contradiction, 'reason'))} "
                f"severity={_number_label(_field(contradiction, 'severity'))}"
            )
    gaps = _strings(_field(run, "gaps", []))
    if gaps:
        lines.extend(["", "## Gaps", ""])
        for gap in gaps:
            lines.append(f"- {gap}")
    return "\n".join(lines) + "\n"


def html_report(run: ResearchRun) -> str:
    labels = Synthesizer().citation_labels(_rows(run, "evidence"))
    metrics = _mapping(_field(run, "metrics", {}))
    readiness = _number(metrics.get("readiness_score", 0.0))
    synthesis_source = synthesis_source_label(run)
    claim_cards = "\n".join(
        f"""
        <article class="claim">
          <div class="claim-head"><code>{html.escape(_text(_field(claim, 'id')))}</code><span>{_number_label(_field(claim, 'confidence'))}</span></div>
          <p>{html.escape(_text(_field(claim, 'text')))}</p>
          <div class="meta">Citations: {html.escape(", ".join(labels.get(cid, cid) for cid in _strings(_field(claim, 'citations', []))))}</div>
          <div class="meta">Flags: {html.escape(", ".join(_strings(_field(claim, 'quality_flags', []))) or "none")}</div>
        </article>
        """
        for claim in _rows(run, "claims")
    )
    evidence_rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(labels.get(_text(_field(chunk, 'id')), _text(_field(chunk, 'id'))))}</td>
          <td>{html.escape(_text(_field(chunk, 'title')))}</td>
          <td>{html.escape(domain(_field(chunk, 'url')))}</td>
          <td>{_number_label(_field(chunk, 'score'))}</td>
          <td>{html.escape(compact_text(_field(chunk, 'text'), 240))}</td>
        </tr>
        """
        for chunk in _rows(run, "evidence")
    )
    metric_tiles = "\n".join(
        f'<div class="metric"><span>{html.escape(_text(key).replace("_", " "))}</span><strong>{html.escape(_display(value))}</strong></div>'
        for key, value in sorted(metrics.items(), key=lambda item: _text(item[0]))
    )
    gaps = "".join(f"<li>{html.escape(gap)}</li>" for gap in _strings(_field(run, "gaps", []))) or "<li>No major gaps flagged.</li>"
    contradictions = "".join(
        f"<li><code>{html.escape(_text(_field(c, 'claim_a')))}</code> vs <code>{html.escape(_text(_field(c, 'claim_b')))}</code>: "
        f"{html.escape(_text(_field(c, 'reason')))} ({_number_label(_field(c, 'severity'))})</li>"
        for c in _rows(run, "contradictions")
    ) or "<li>No contradiction pairs detected.</li>"
    plan_steps = "".join(
        f"<li><strong>{html.escape(_text(_field(step, 'intent')))}</strong>: {html.escape(_text(_field(step, 'question')))}</li>"
        for step in _items(_field(_field(run, "plan", {}), "steps", []))
    )
    discovery_section = ""
    discoveries = _rows(run, "discoveries")
    if discoveries:
        cards = "\n".join(
            f"""
            <article class="claim">
              <div class="claim-head"><code>{html.escape(_text(_field(d, 'id')))}</code><span>{_number_label(_field(d, 'confidence'))}</span></div>
              <p>{html.escape(_text(_field(d, 'statement')))}</p>
              <div class="meta">effect {_number(_field(d, 'effect_size')):+.2f} &middot; reproducibility {_number_label(_field(d, 'reproducibility'))}
                &middot; novelty {_number_label(_field(d, 'novelty'))} &middot; {len(_strings(_field(d, 'supporting_experiments', [])))} experiments</div>
            </article>
            """
            for d in discoveries
        )
        discovery_section = f'<section><h2>Confirmed Mechanisms</h2><div class="claims">{cards}</div></section>'
    experiment_section = ""
    ran = [e for e in _rows(run, "experiments") if _text(_field(e, "status")) == "ran"]
    if ran:
        rows = "\n".join(
            f"""
            <tr>
              <td><code>{html.escape(_text(_field(e, 'probe')))}</code></td>
              <td>{html.escape(_jsonable(_field(e, 'target', {})))}</td>
              <td>{_number(_field(e, 'effect_size')):+.3f}</td>
              <td>{_number(_field(e, 'baseline')):+.3f}</td>
              <td>{'yes' if _flag(_field(e, 'significant')) else 'no'}</td>
              <td>{'yes' if _flag(_field(e, 'reproduced')) else 'no'}</td>
              <td>{html.escape(_text(_field(e, 'backend_used')))}</td>
            </tr>
            """
            for e in sorted(ran, key=lambda x: abs(_number(_field(x, "effect_size"))), reverse=True)
            if _flag(_field(e, "significant")) and _flag(_field(e, "reproduced"))
        )
        experiment_section = f"""<section><h2>Experiment Ledger ({len(ran)} ran)</h2>
          <table>
            <thead><tr><th>Probe</th><th>Target</th><th>Effect</th><th>Control</th><th>Sig.</th><th>Repro.</th><th>Backend</th></tr></thead>
            <tbody>{rows}</tbody>
          </table></section>"""
    payload = _json_dumps(run.to_dict(), sort_keys=True).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MechFerret Dossier</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #5f6c7b;
      --line: #d8dee6;
      --paper: #fbfcfd;
      --accent: #0f766e;
      --warn: #9a3412;
      --panel: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 52px);
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 46px); letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 21px; letter-spacing: 0; }}
    .question {{ max-width: 980px; color: var(--muted); font-size: 18px; }}
    .score {{
      margin-top: 18px;
      width: min(560px, 100%);
      height: 14px;
      border: 1px solid var(--line);
      background: #edf2f7;
    }}
    .score > div {{ width: {max(2, _ratio(readiness) * 100):.1f}%; height: 100%; background: var(--accent); }}
    main {{ padding: 24px clamp(18px, 4vw, 52px) 52px; }}
    section {{ margin: 0 auto 28px; max-width: 1180px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
    .metric, .claim {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; text-transform: uppercase; }}
    .metric strong {{ font-size: 24px; }}
    .claims {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 12px; }}
    .claim-head {{ display: flex; justify-content: space-between; gap: 10px; color: var(--accent); }}
    .meta {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
    .synthesis-source {{ color: var(--muted); font-size: 13px; margin: -6px 0 10px; }}
    .synthesis {{
      white-space: pre-wrap;
      background: #ffffff;
      border-left: 4px solid var(--accent);
      padding: 16px;
    }}
    table {{ width: 100%; border-collapse: collapse; background: #ffffff; }}
    th, td {{ border: 1px solid var(--line); padding: 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    ul {{ padding-left: 22px; }}
  </style>
</head>
<body>
  <header>
    <h1>MechFerret Research Dossier</h1>
    <div class="question">{html.escape(_text(_field(run, "question")))}</div>
    <div class="score" title="Readiness score"><div></div></div>
    <p class="meta">Run {html.escape(_text(_field(run, "run_id")))} at {html.escape(_text(_field(run, "created_at")))}. Readiness {readiness:.2f}.</p>
  </header>
  <main>
    <section><h2>Synthesis</h2><p class="synthesis-source">{html.escape(synthesis_source)}</p><div class="synthesis">{html.escape(_text(_field(run, "answer")))}</div></section>
    {discovery_section}
    {experiment_section}
    <section><h2>Metrics</h2><div class="metrics">{metric_tiles}</div></section>
    <section><h2>Plan</h2><ul>{plan_steps}</ul></section>
    <section><h2>Claims</h2><div class="claims">{claim_cards}</div></section>
    <section><h2>Gaps</h2><ul>{gaps}</ul></section>
    <section><h2>Contradictions</h2><ul>{contradictions}</ul></section>
    <section><h2>Evidence Ledger</h2>
      <table>
        <thead><tr><th>ID</th><th>Title</th><th>Domain</th><th>Score</th><th>Excerpt</th></tr></thead>
        <tbody>{evidence_rows}</tbody>
      </table>
    </section>
  </main>
  <script id="run-json" type="application/json">{payload}</script>
</body>
</html>
"""


def synthesis_source_label(run: ResearchRun) -> str:
    provenance = _mapping(_field(run, "provenance", {}))
    author = _text(provenance.get("answer_author", ""))
    if author == "provider_model":
        provider = _text(provenance.get("answer_provider") or provenance.get("provider_requested") or "provider")
        model = _text(provenance.get("answer_model") or provenance.get("llm_model") or provenance.get("model") or "configured model")
        return f"model-authored synthesis ({provider}/{model})"
    if author == "experiment_ledger_synthesizer":
        return "local experiment-ledger synthesis; use --provider openai or --provider anthropic for model-authored final prose"
    if author == "local_extractive_synthesizer":
        return "local extractive synthesis; use --provider openai or --provider anthropic for model-authored final prose"
    return "legacy run; synthesis author was not recorded"


def claim_graph(run: ResearchRun) -> dict:
    nodes = []
    edges = []
    for source in _rows(run, "sources"):
        source_id = _text(_field(source, "id")).strip()
        if not source_id:
            continue
        nodes.append({
            "id": source_id,
            "type": "source",
            "label": _text(_field(source, "title")),
            "url": _text(_field(source, "url")),
            "kind": _text(_field(source, "kind", "document")) or "document",
        })
    for chunk in _rows(run, "evidence"):
        chunk_id = _text(_field(chunk, "id")).strip()
        if not chunk_id:
            continue
        source_id = _text(_field(chunk, "source_id")).strip()
        nodes.append({"id": chunk_id, "type": "evidence", "label": _text(_field(chunk, "title")), "score": _number(_field(chunk, "score"))})
        if source_id:
            edges.append({"from": source_id, "to": chunk_id, "type": "contains"})
    for claim in _rows(run, "claims"):
        claim_id = _text(_field(claim, "id")).strip()
        if not claim_id:
            continue
        nodes.append(
            {
                "id": claim_id,
                "type": "claim",
                "label": compact_text(_field(claim, "text"), 120),
                "confidence": _number(_field(claim, "confidence")),
                "support_score": _number(_field(claim, "support_score")),
                "flags": _strings(_field(claim, "quality_flags", [])),
            }
        )
        for citation in _strings(_field(claim, "citations", [])):
            edges.append({"from": citation, "to": claim_id, "type": "supports"})
    for contradiction in _rows(run, "contradictions"):
        claim_a = _text(_field(contradiction, "claim_a")).strip()
        claim_b = _text(_field(contradiction, "claim_b")).strip()
        if not claim_a or not claim_b:
            continue
        edges.append(
            {
                "from": claim_a,
                "to": claim_b,
                "type": "contradicts",
                "severity": _number(_field(contradiction, "severity")),
                "reason": _text(_field(contradiction, "reason")),
            }
        )
    for hypothesis in _rows(run, "hypotheses"):
        hypothesis_id = _text(_field(hypothesis, "id")).strip()
        if not hypothesis_id:
            continue
        nodes.append(
            {
                "id": hypothesis_id,
                "type": "hypothesis",
                "label": compact_text(_field(hypothesis, "statement"), 120),
                "status": _text(_field(hypothesis, "status")),
                "confidence": _number(_field(hypothesis, "confidence")),
            }
        )
    for discovery in _rows(run, "discoveries"):
        discovery_id = _text(_field(discovery, "id")).strip()
        if not discovery_id:
            continue
        nodes.append(
            {
                "id": discovery_id,
                "type": "discovery",
                "label": compact_text(_field(discovery, "statement"), 120),
                "confidence": _number(_field(discovery, "confidence")),
                "novelty": _number(_field(discovery, "novelty")),
                "effect_size": _number(_field(discovery, "effect_size")),
            }
        )
        hypothesis_id = _text(_field(discovery, "hypothesis_id")).strip()
        if hypothesis_id:
            edges.append({"from": hypothesis_id, "to": discovery_id, "type": "confirmed_by"})
        for claim_id in _strings(_field(discovery, "claim_ids", [])):
            edges.append({"from": discovery_id, "to": claim_id, "type": "asserts"})
    return {"run_id": _text(_field(run, "run_id")), "question": _text(_field(run, "question")), "nodes": nodes, "edges": edges}


def run_evals(run: ResearchRun) -> dict:
    metrics = _mapping(_field(run, "metrics", {}))
    claims = _rows(run, "claims")
    experiments = _rows(run, "experiments")
    discoveries = _rows(run, "discoveries")
    checks = [
        {
            "name": "extracts_at_least_five_claims",
            "passed": len(claims) >= 5,
            "observed": len(claims),
            "threshold": 5,
        },
        {
            "name": "uses_at_least_three_sources",
            "passed": _number(metrics.get("source_diversity", 0)) >= 3,
            "observed": metrics.get("source_diversity", 0),
            "threshold": 3,
        },
        {
            "name": "citations_per_claim",
            "passed": _number(metrics.get("citation_density", 0)) >= 0.85,
            "observed": metrics.get("citation_density", 0),
            "threshold": 0.85,
        },
        {
            "name": "plan_coverage",
            "passed": _number(metrics.get("plan_coverage", 0)) >= 0.7,
            "observed": metrics.get("plan_coverage", 0),
            "threshold": 0.7,
        },
        {
            "name": "contradiction_pressure_bounded",
            "passed": _number(metrics.get("contradiction_pressure", 0)) <= 1.0,
            "observed": metrics.get("contradiction_pressure", 0),
            "threshold": 1.0,
        },
    ]
    if _text(_field(run, "mode", "literature")) == "discovery" or experiments or discoveries:
        ran = [e for e in experiments if _text(_field(e, "status")) == "ran"]
        controlled = [e for e in ran if _field(e, "baseline", None) is not None]
        reproduced_sig = [e for e in ran if _flag(_field(e, "significant")) and _flag(_field(e, "reproduced"))]
        triangulated = all(
            _number(_field(d, "reproducibility")) > 0 and len(_strings(_field(d, "supporting_experiments", []))) >= 2
            for d in discoveries
        ) if discoveries else False
        checks.extend([
            {
                "name": "has_confirmed_mechanism",
                "passed": len(discoveries) >= 1,
                "observed": len(discoveries),
                "threshold": 1,
            },
            {
                "name": "every_experiment_has_control",
                "passed": len(controlled) == len(ran) and len(ran) > 0,
                "observed": f"{len(controlled)}/{len(ran)}",
                "threshold": "all",
            },
            {
                "name": "significant_effects_reproduce",
                "passed": _number(metrics.get("reproducibility_rate", 0)) >= 0.8,
                "observed": metrics.get("reproducibility_rate", 0),
                "threshold": 0.8,
            },
            {
                "name": "discoveries_are_triangulated",
                "passed": triangulated,
                "observed": triangulated,
                "threshold": True,
            },
        ])
    return {
        "run_id": _text(_field(run, "run_id")),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "readiness_score": metrics.get("readiness_score", 0),
    }
