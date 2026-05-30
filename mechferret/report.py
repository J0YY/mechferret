from __future__ import annotations

import html
import json
from pathlib import Path

from .agents import Synthesizer
from .models import ResearchRun
from .text import compact_text, domain


def write_artifacts(run: ResearchRun, out_dir: str | Path) -> dict[str, str]:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "run.json"
    md_path = path / "report.md"
    html_path = path / "report.html"
    graph_path = path / "graph.json"
    evals_path = path / "evals.json"
    json_path.write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown_report(run), encoding="utf-8")
    html_path.write_text(html_report(run), encoding="utf-8")
    graph_path.write_text(json.dumps(claim_graph(run), indent=2, sort_keys=True), encoding="utf-8")
    evals_path.write_text(json.dumps(run_evals(run), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
        "graph": str(graph_path),
        "evals": str(evals_path),
        "trace": str(path / "trace.jsonl"),
    }


def markdown_report(run: ResearchRun) -> str:
    labels = Synthesizer().citation_labels(run.evidence)
    lines = [
        f"# MechFerret Research Dossier",
        "",
        f"**Question:** {run.question}",
        f"**Run:** {run.run_id}",
        f"**Readiness score:** {run.metrics.get('readiness_score', 0):.2f}",
        "",
        "## Synthesis",
        "",
        run.answer,
        "",
        "## Metrics",
        "",
    ]
    for key, value in sorted(run.metrics.items()):
        lines.append(f"- **{key}:** {value}")
    lines.extend(["", "## Claims", ""])
    for claim in run.claims:
        cites = ", ".join(labels.get(cid, cid) for cid in claim.citations)
        flags = f" flags={','.join(claim.quality_flags)}" if claim.quality_flags else ""
        lines.append(f"- `{claim.id}` {claim.text} [{cites}] confidence={claim.confidence:.2f}{flags}")
    lines.extend(["", "## Evidence Ledger", ""])
    for chunk in run.evidence:
        label = labels.get(chunk.id, chunk.id)
        lines.append(f"- **{label}** {chunk.title} ({chunk.url or 'local'}) score={chunk.score:.2f}: {compact_text(chunk.text, 260)}")
    if run.contradictions:
        lines.extend(["", "## Contradictions", ""])
        for contradiction in run.contradictions:
            lines.append(
                f"- `{contradiction.id}` {contradiction.claim_a} vs {contradiction.claim_b}: "
                f"{contradiction.reason} severity={contradiction.severity:.2f}"
            )
    if run.gaps:
        lines.extend(["", "## Gaps", ""])
        for gap in run.gaps:
            lines.append(f"- {gap}")
    return "\n".join(lines) + "\n"


def html_report(run: ResearchRun) -> str:
    labels = Synthesizer().citation_labels(run.evidence)
    readiness = run.metrics.get("readiness_score", 0.0)
    claim_cards = "\n".join(
        f"""
        <article class="claim">
          <div class="claim-head"><code>{html.escape(claim.id)}</code><span>{claim.confidence:.2f}</span></div>
          <p>{html.escape(claim.text)}</p>
          <div class="meta">Citations: {html.escape(", ".join(labels.get(cid, cid) for cid in claim.citations))}</div>
          <div class="meta">Flags: {html.escape(", ".join(claim.quality_flags) or "none")}</div>
        </article>
        """
        for claim in run.claims
    )
    evidence_rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(labels.get(chunk.id, chunk.id))}</td>
          <td>{html.escape(chunk.title)}</td>
          <td>{html.escape(domain(chunk.url))}</td>
          <td>{chunk.score:.2f}</td>
          <td>{html.escape(compact_text(chunk.text, 240))}</td>
        </tr>
        """
        for chunk in run.evidence
    )
    metric_tiles = "\n".join(
        f'<div class="metric"><span>{html.escape(key.replace("_", " "))}</span><strong>{value}</strong></div>'
        for key, value in sorted(run.metrics.items())
    )
    gaps = "".join(f"<li>{html.escape(gap)}</li>" for gap in run.gaps) or "<li>No major gaps flagged.</li>"
    contradictions = "".join(
        f"<li><code>{html.escape(c.claim_a)}</code> vs <code>{html.escape(c.claim_b)}</code>: "
        f"{html.escape(c.reason)} ({c.severity:.2f})</li>"
        for c in run.contradictions
    ) or "<li>No contradiction pairs detected.</li>"
    plan_steps = "".join(
        f"<li><strong>{html.escape(step.intent)}</strong>: {html.escape(step.question)}</li>"
        for step in run.plan.steps
    )
    payload = html.escape(json.dumps(run.to_dict(), sort_keys=True))
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
    .score > div {{ width: {max(2, readiness * 100):.1f}%; height: 100%; background: var(--accent); }}
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
    <div class="question">{html.escape(run.question)}</div>
    <div class="score" title="Readiness score"><div></div></div>
    <p class="meta">Run {html.escape(run.run_id)} at {html.escape(run.created_at)}. Readiness {readiness:.2f}.</p>
  </header>
  <main>
    <section><h2>Synthesis</h2><div class="synthesis">{html.escape(run.answer)}</div></section>
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


def claim_graph(run: ResearchRun) -> dict:
    nodes = []
    edges = []
    for source in run.sources:
        nodes.append({"id": source.id, "type": "source", "label": source.title, "url": source.url, "kind": source.kind})
    for chunk in run.evidence:
        nodes.append({"id": chunk.id, "type": "evidence", "label": chunk.title, "score": chunk.score})
        edges.append({"from": chunk.source_id, "to": chunk.id, "type": "contains"})
    for claim in run.claims:
        nodes.append(
            {
                "id": claim.id,
                "type": "claim",
                "label": compact_text(claim.text, 120),
                "confidence": claim.confidence,
                "support_score": claim.support_score,
                "flags": claim.quality_flags,
            }
        )
        for citation in claim.citations:
            edges.append({"from": citation, "to": claim.id, "type": "supports"})
    for contradiction in run.contradictions:
        edges.append(
            {
                "from": contradiction.claim_a,
                "to": contradiction.claim_b,
                "type": "contradicts",
                "severity": contradiction.severity,
                "reason": contradiction.reason,
            }
        )
    return {"run_id": run.run_id, "question": run.question, "nodes": nodes, "edges": edges}


def run_evals(run: ResearchRun) -> dict:
    checks = [
        {
            "name": "extracts_at_least_five_claims",
            "passed": len(run.claims) >= 5,
            "observed": len(run.claims),
            "threshold": 5,
        },
        {
            "name": "uses_at_least_three_sources",
            "passed": run.metrics.get("source_diversity", 0) >= 3,
            "observed": run.metrics.get("source_diversity", 0),
            "threshold": 3,
        },
        {
            "name": "citations_per_claim",
            "passed": run.metrics.get("citation_density", 0) >= 0.85,
            "observed": run.metrics.get("citation_density", 0),
            "threshold": 0.85,
        },
        {
            "name": "plan_coverage",
            "passed": run.metrics.get("plan_coverage", 0) >= 0.7,
            "observed": run.metrics.get("plan_coverage", 0),
            "threshold": 0.7,
        },
        {
            "name": "contradiction_pressure_bounded",
            "passed": run.metrics.get("contradiction_pressure", 0) <= 1.0,
            "observed": run.metrics.get("contradiction_pressure", 0),
            "threshold": 1.0,
        },
    ]
    return {
        "run_id": run.run_id,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "readiness_score": run.metrics.get("readiness_score", 0),
    }
