import json
import math
import os
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


class AgentStackTest(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.mkdtemp()
        os.chdir(self._tmp)

    def tearDown(self):
        os.chdir(self._cwd)

    def test_sessions_roundtrip_and_list(self):
        from mechferret import sessions

        sessions.save_session("s1", "anthropic", "claude-opus-4-8", [{"role": "user", "content": "hi"}], {"usd": 0.02, "input": 9, "output": 3})
        loaded = sessions.load_session("s1")
        self.assertEqual(loaded["model"], "claude-opus-4-8")
        metas = sessions.list_sessions()
        self.assertEqual(metas[0].id, "s1")
        self.assertEqual(metas[0].turns, 1)
        with self.assertRaises(KeyError):
            sessions.load_session("nope")

    def test_list_sessions_normalizes_malformed_limits(self):
        from mechferret import sessions

        sessions.save_session("s1", "anthropic", "claude-opus-4-8", [{"role": "user", "content": "hi"}], {"usd": 0.02})
        sessions.save_session("s2", "openai", "gpt-5", [{"role": "user", "content": "hi"}], {"usd": 0.03})

        self.assertEqual(sessions.list_sessions(limit=0), [])
        self.assertEqual(sessions.list_sessions(limit=-5), [])
        self.assertEqual(len(sessions.list_sessions(limit="bad")), 2)

    def test_sessions_reject_path_like_ids(self):
        from mechferret import sessions

        for bad_id in ("../escape", "nested/session", "", "x" * 129):
            with self.subTest(session_id=bad_id):
                with self.assertRaises(ValueError):
                    sessions.load_session(bad_id)
                with self.assertRaises(ValueError):
                    sessions.save_session(bad_id, "anthropic", "model", [], {})

    def test_load_session_rejects_non_object_json(self):
        from mechferret import agent, sessions

        sessions.SESSIONS_DIR.mkdir(parents=True)
        (sessions.SESSIONS_DIR / "bad-shape.json").write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

        with self.assertRaises(ValueError):
            sessions.load_session("bad-shape")

        a = agent.Agent()
        a.provider, a.model, a._key = "anthropic", "claude-opus-4-8", "x"
        with self.assertRaises(ValueError):
            a.load_session("bad-shape")

    def test_load_session_reports_corrupt_json_cleanly(self):
        from mechferret import sessions

        sessions.SESSIONS_DIR.mkdir(parents=True)
        (sessions.SESSIONS_DIR / "bad-json.json").write_text("{", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "could not read JSON"):
            sessions.load_session("bad-json")

    def test_save_session_normalizes_non_json_payloads(self):
        from mechferret import sessions

        path = sessions.save_session(
            "odd",
            "anthropic",
            "claude-opus-4-8",
            [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_result", "path": Path("runs/demo"), "blob": b"abc"}],
                }
            ],
            {"usd": math.nan, "nested": {1: Path("artifact"), "items": ("a", Path("b")), "bad": math.inf}},
        )

        loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["messages"][0]["content"][0]["path"], "runs/demo")
        self.assertEqual(loaded["messages"][0]["content"][0]["blob"], "b'abc'")
        self.assertIsNone(loaded["cost"]["usd"])
        self.assertEqual(loaded["cost"]["nested"]["1"], "artifact")
        self.assertEqual(loaded["cost"]["nested"]["items"], ["a", "b"])
        self.assertIsNone(loaded["cost"]["nested"]["bad"])
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def test_session_listing_tolerates_corrupt_metadata(self):
        from mechferret import sessions

        sessions.SESSIONS_DIR.mkdir(parents=True)
        (sessions.SESSIONS_DIR / "bad-shape.json").write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        (sessions.SESSIONS_DIR / "bad-meta.json").write_text(
            json.dumps(
                {
                    "id": "../bad",
                    "provider": [],
                    "model": {"name": "x"},
                    "messages": "not a list",
                    "cost": {"usd": "Infinity"},
                    "updated_at": [],
                }
            ),
            encoding="utf-8",
        )
        sessions.save_session(
            "good",
            "anthropic",
            "claude-opus-4-8",
            [{"role": "user", "content": "hi"}],
            {"usd": 0.25},
        )

        metas = sessions.list_sessions(limit=10)
        by_id = {meta.id: meta for meta in metas}

        self.assertIn("good", by_id)
        self.assertIn("bad-meta", by_id)
        self.assertNotIn("bad-shape", by_id)
        self.assertEqual(by_id["bad-meta"].turns, 0)
        self.assertEqual(by_id["bad-meta"].usd, 0.0)
        self.assertEqual(by_id["bad-meta"].provider, "")
        self.assertEqual(by_id["bad-meta"].model, "")

    def test_session_listing_skips_invalid_fallback_ids_and_non_directory_store(self):
        from mechferret import sessions

        sessions.SESSIONS_DIR.mkdir(parents=True)
        (sessions.SESSIONS_DIR / f"{'x' * 129}.json").write_text(
            json.dumps({"id": "../bad", "messages": [{"role": "user"}]}),
            encoding="utf-8",
        )
        sessions.save_session("good", "anthropic", "claude-opus-4-8", [], {})

        self.assertEqual([meta.id for meta in sessions.list_sessions(limit=10)], ["good"])

        for path in sessions.SESSIONS_DIR.iterdir():
            path.unlink()
        sessions.SESSIONS_DIR.rmdir()
        sessions.SESSIONS_DIR.parent.mkdir(exist_ok=True)
        sessions.SESSIONS_DIR.write_text("not a directory", encoding="utf-8")

        self.assertEqual(sessions.list_sessions(), [])

    def test_agent_load_session_sanitizes_corrupt_payload(self):
        from mechferret import agent, sessions

        sessions.save_session(
            "corrupt",
            "",
            "",
            [{"role": "user", "content": "keep"}, "drop", []],
            {"usd": "bad", "input": "also bad", "output": 3},
        )

        a = agent.Agent()
        a.provider, a.model, a._key = "anthropic", "claude-opus-4-8", "x"
        a.load_session("corrupt")

        self.assertEqual(a.session_id, "corrupt")
        self.assertEqual(a.provider, "anthropic")
        self.assertEqual(a.model, "claude-opus-4-8")
        self.assertEqual(a.messages, [{"role": "user", "content": "keep"}])
        self.assertEqual(a.cost.usd, 0.0)
        self.assertEqual(a.cost.input_tokens, 0)
        self.assertEqual(a.cost.output_tokens, 3)

    def test_agent_load_session_refreshes_stale_system_prompt_and_scaffold(self):
        from mechferret import agent, sessions

        stale_system = (
            "NEVER end a turn flat. Always finish with NEXT STEP and "
            "press " + "enter to proceed."
        )
        stale_reply = (
            "The minimal experiment should be:\n"
            "1. Target behavior: duplicate-token/name-mover behavior in GPT-2 small.\n"
            "2. Start modules: heads 5.0, 5.2, 5.6, 5.11, 4.8, 6.8, 4.11, 4.3.\n"
            "Next: " + "press " + "enter to proceed."
        )
        sessions.save_session(
            "stale",
            "openai",
            "gpt-test",
            [
                {"role": "system", "content": stale_system},
                {"role": "user", "content": "keep going"},
                {"role": "assistant", "content": stale_reply},
            ],
            {},
        )

        a = agent.Agent()
        a.provider, a.model, a._key = "openai", "gpt-test", "x"
        a.load_session("stale")

        self.assertEqual(a.messages[0]["role"], "system")
        self.assertNotIn("NEVER end a turn flat", a.messages[0]["content"])
        self.assertNotIn("press " + "enter", a.messages[0]["content"].lower())
        self.assertEqual(a.messages[1], {"role": "user", "content": "keep going"})
        self.assertIn("which model and behavior/task", a.messages[2]["content"])
        self.assertNotIn("GPT-2", a.messages[2]["content"])
        self.assertNotIn("5.0", a.messages[2]["content"])

    def test_agent_load_session_rejects_embedded_bad_id_and_provider(self):
        from mechferret import agent, sessions

        sessions.SESSIONS_DIR.mkdir(parents=True)
        (sessions.SESSIONS_DIR / "bad-provider.json").write_text(
            json.dumps(
                {
                    "id": "../bad",
                    "provider": "local",
                    "model": "should-not-load",
                    "messages": [{"role": "user", "content": "keep"}],
                    "cost": {"usd": 1, "input": 2, "output": 3},
                }
            ),
            encoding="utf-8",
        )

        a = agent.Agent()
        a.provider, a.model, a._key = "anthropic", "claude-opus-4-8", "x"
        a.load_session("bad-provider")

        self.assertEqual(a.session_id, "bad-provider")
        self.assertEqual(a.provider, "anthropic")
        self.assertEqual(a.model, "claude-opus-4-8")
        self.assertEqual(a.messages, [{"role": "user", "content": "keep"}])
        self.assertEqual(a.cost.input_tokens, 2)

    def test_persist_failure_is_traced_without_blocking_send(self):
        from mechferret import agent, sessions

        def fake_post(url, payload, headers):
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "reply"}]}

        def fail_save(*args, **kwargs):
            raise OSError("disk full")

        original_post = agent._http_post
        original_save = sessions.save_session
        agent._http_post = fake_post
        sessions.save_session = fail_save
        try:
            a = agent.Agent()
            a.provider, a.model, a._key = "anthropic", "claude-opus-4-8", "x"
            reply = a.send("hello")
        finally:
            agent._http_post = original_post
            sessions.save_session = original_save

        self.assertEqual(reply, "reply")
        trace_path = Path(".mechferret/trace.jsonl")
        records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        persist_failures = [record for record in records if record.get("name") == "session_persist_failed"]
        self.assertEqual(len(persist_failures), 1)
        self.assertIn("OSError: disk full", persist_failures[0]["attributes"]["error"])

    def test_trace_recorder_normalizes_attrs_and_write_failures(self):
        from mechferret.tracing import TraceRecorder

        tracer = TraceRecorder("run", ".mechferret")
        tracer.event("rich_attrs", path=Path("runs/demo"), bad=math.inf, nested={1: (Path("artifact"), math.nan)})

        record = json.loads(Path(".mechferret/trace.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(record["attributes"]["path"], "runs/demo")
        self.assertIsNone(record["attributes"]["bad"])
        self.assertEqual(record["attributes"]["nested"]["1"][0], "artifact")
        self.assertIsNone(record["attributes"]["nested"]["1"][1])
        with Path(".mechferret/trace.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

        tracer.path = Path("missing") / "trace.jsonl"
        tracer.event("write_failure_is_best_effort", path=Path("no-parent"))

    def test_trace_recorder_initialization_is_best_effort(self):
        from mechferret import agent
        from mechferret.tracing import TraceRecorder

        Path(".mechferret").write_text("not a directory", encoding="utf-8")

        tracer = TraceRecorder("run", ".mechferret")
        tracer.event("disabled_local_trace")
        self.assertIsNone(tracer.path)

        a = agent.Agent()
        self.assertIsNone(a.tracer.path)

    def test_mechanisms_record_and_recall(self):
        from mechferret.memory import ResearchMemory

        mem = ResearchMemory(".mechferret/memory.sqlite")
        try:
            n = mem.record_mechanisms("gpt2", [{"statement": "head 5.5 is a name mover", "effect_size": 1.5, "reproducibility": 0.67, "novelty": 0.6}])
            self.assertEqual(n, 1)
            rows = mem.recent_mechanisms()
            self.assertEqual(rows[0]["statement"], "head 5.5 is a name mover")
        finally:
            mem.close()

    def test_compaction_keeps_tail_and_summary(self):
        from mechferret import agent

        original = agent._http_post
        agent._http_post = lambda url, p, h: {"content": [{"type": "text", "text": "SUMMARY: head 5.5 name-mover effect 1.5 seeds 0-2"}]}
        try:
            a = agent.Agent()
            a.provider, a.model, a._key = "anthropic", "claude-opus-4-8", "x"
            a.messages = [{"role": "user", "content": f"m{i}"} for i in range(12)]
            summary = a.compact()
            self.assertIn("SUMMARY", summary)
            self.assertEqual(len(a.messages), 1 + agent.COMPACT_KEEP_LAST)
            self.assertTrue(a.messages[0]["content"].startswith("[Summary"))
        finally:
            agent._http_post = original

    def test_automatic_compaction_failure_does_not_block_send(self):
        from mechferret import agent

        calls = {"n": 0}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            if payload.get("system") == agent.COMPACT_SYSTEM:
                raise RuntimeError("summary provider unavailable")
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "normal reply"}]}

        original_post = agent._http_post
        original_threshold = agent.COMPACT_CHAR_THRESHOLD
        agent._http_post = fake_post
        agent.COMPACT_CHAR_THRESHOLD = 1
        try:
            a = agent.Agent()
            a.provider, a.model, a._key = "anthropic", "claude-opus-4-8", "x"
            a.messages = [{"role": "user", "content": "old context " * 50} for _ in range(6)]
            reply = a.send("continue")
        finally:
            agent._http_post = original_post
            agent.COMPACT_CHAR_THRESHOLD = original_threshold

        self.assertEqual(reply, "normal reply")
        self.assertEqual(calls["n"], 2)
        self.assertTrue(any(message.get("content") == "continue" for message in a.messages))

    def test_parallel_readonly_and_plan_denial(self):
        from mechferret import agent

        a = agent.Agent()
        a.provider, a.model, a._key = "anthropic", "claude-opus-4-8", "x"
        res = a._run_tool_calls([("1", "list_skills", {}), ("2", "environment_status", {})])
        self.assertEqual(sorted(res), ["1", "2"])
        self.assertTrue(all(res.values()))
        a.permission_mode = "plan"
        res2 = a._run_tool_calls([("a", "write_file", {"path": "x", "content": "y"})])
        denied = json.loads(res2["a"])
        self.assertFalse(denied["ok"])
        self.assertTrue(denied["denied"])
        self.assertIn("tool_permission", denied["failed_checks"])

        malformed = json.loads(a._dispatch("read_file", []))
        self.assertFalse(malformed["ok"])
        self.assertIn("tool_arguments", malformed["failed_checks"])

        a.abort.set()
        aborted = json.loads(a._run_tool_calls([("b", "write_file", {"path": "x", "content": "y"})])["b"])
        self.assertFalse(aborted["ok"])
        self.assertTrue(aborted["aborted"])
        self.assertIn("tool_aborted", aborted["failed_checks"])

    def test_mcp_no_server_is_safe(self):
        from mechferret import mcp, tools

        self.assertEqual(mcp.status()["configured"], [])
        self.assertEqual(mcp.tool_specs(), [])
        payload = json.loads(tools.run_tool("mcp__x__y", {}))
        self.assertIn("error", payload)
        self.assertFalse(payload["ok"])
        self.assertIn("mcp_tool_call", payload["failed_checks"])

    def test_mcp_config_filters_malformed_servers(self):
        from mechferret import mcp

        mcp.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        mcp.CONFIG_PATH.write_text(
            json.dumps(
                {
                    "servers": {
                        "good": {"command": " python ", "args": ["-m", 123], "env": {"TOKEN": "x", "BAD": 1}},
                        "../bad": {"command": "python"},
                        "bad-command": {"command": ["python"]},
                        "bad-row": "not an object",
                    }
                }
            ),
            encoding="utf-8",
        )
        try:
            servers = mcp.load_servers()
            self.assertEqual(len(servers), 1)
            self.assertEqual(servers[0].name, "good")
            self.assertEqual(servers[0].command, "python")
            self.assertEqual(servers[0].args, ["-m"])
            self.assertEqual(servers[0].env, {"TOKEN": "x"})

            with self.assertRaises(ValueError):
                mcp.add_server("../bad", "python")
            with self.assertRaises(ValueError):
                mcp.add_server("good2", "")

            mcp.CONFIG_PATH.write_text("{", encoding="utf-8")
            path = mcp.add_server("new", " python ", ["-m", 3])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["servers"]["new"], {"command": "python", "args": ["-m"]})
        finally:
            mcp.reset()

    def test_command_registry_matches_handlers(self):
        from mechferret import commands, repl

        # every REPL-handled bare word should appear in the grouped help registry
        names = " ".join(c.name for _title, cmds in commands.SECTIONS for c in cmds)
        for handled in (
            "login", "model", "plan", "cost", "compact", "resume", "memory",
            "tool-results", "export", "init", "btw", "queue", "cancel", "goal", "why", "arch", "paper",
            "audit", "bundle", "verify-bundle", "sae", "quickstart", "status", "next",
            "runs", "open", "version", "commands", "completion", "api",
        ):
            self.assertIn(handled, names)
        self.assertIn("run_research", names)
        self.assertIn("commands --workflow first_run", names)
        self.assertIn("tool-results", repl.KNOWN_COMMANDS)
        self.assertIn("verify-bundle", repl.KNOWN_COMMANDS)
        self.assertIn("completion", repl.KNOWN_COMMANDS)
        self.assertIn("api", repl.KNOWN_COMMANDS)
        self.assertIn("tool-results", commands.REPL_HANDLED)
        self.assertIn("verify-bundle", commands.REPL_HANDLED)
        self.assertEqual(repl.KNOWN_COMMANDS, commands.COMMAND_WORDS)
        self.assertTrue(commands.CLI_FALLBACK <= repl.KNOWN_COMMANDS)

        out = StringIO()
        with redirect_stdout(out):
            repl._print_help()
        rendered_help = out.getvalue()
        self.assertIn("/btw <text>", rendered_help)
        self.assertIn("/queue", rendered_help)
        self.assertIn("/queue add <text>", rendered_help)
        self.assertIn("/queue show <id|latest|active|running|side|next>", rendered_help)
        self.assertIn("/queue retry <id|latest|running|side|next>", rendered_help)
        self.assertIn("/queue apply <id|side|latest|all>", rendered_help)
        self.assertIn("/queue edit <id|latest|next> <text>", rendered_help)
        self.assertIn("/queue move <id|latest|next> first|last|before|after", rendered_help)
        self.assertIn("/queue cancel <id|latest|next|all>", rendered_help)
        self.assertIn("/queue clear [queued|saved|all]", rendered_help)
        self.assertIn("/queue pause", rendered_help)
        self.assertIn("/queue resume", rendered_help)
        self.assertIn("/queue restore [id|latest|running|side|next|all]", rendered_help)
        self.assertIn("/queue wait [seconds]", rendered_help)
        self.assertIn("/queue join <id|latest|active|running|side|next> [seconds]", rendered_help)
        self.assertIn("/cancel <id|latest|next|all>", rendered_help)
        self.assertIn("/commands --workflow first_run", rendered_help)
        self.assertIn("show a runnable workflow recipe", rendered_help)

    def test_repl_chat_job_runner_queues_prompts_and_btw(self):
        from mechferret import repl

        calls = []

        def fake_chat(agent, session, text, *, background=False):
            calls.append((text, background))
            session.step = f"handled {len(calls)}"
            return f"reply {len(calls)}"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue.json"))
            first = runner.submit("first prompt")
            side = runner.submit_side(repl._btw_prompt("side question"))
            self.assertTrue(runner.wait_idle(timeout=2))
            repl._print_queue(runner)
            runner.stop(wait=True)

        self.assertEqual(first.status, "done")
        self.assertEqual(side.status, "done")
        self.assertEqual(first.reply, "reply 1")
        self.assertEqual(side.reply, "reply 2")
        self.assertEqual([background for _text, background in calls], [True, True])
        self.assertIn("first prompt", calls[0][0])
        self.assertIn("Side request entered with /btw", calls[1][0])
        rendered = out.getvalue()
        self.assertIn("queued #1", rendered)
        self.assertIn("side #2", rendered)
        self.assertIn("use /queue show #1", rendered)
        self.assertIn("use /queue show #2", rendered)
        self.assertNotIn("queue empty", rendered)
        self.assertIn("ready   #2 btw: side question", rendered)
        self.assertIn("/queue apply side", rendered)
        self.assertIn("done     #1 prompt: first prompt", rendered)
        self.assertNotIn("done     #2 btw: side question", rendered)

    def test_repl_chat_job_runner_prints_show_hint_for_errors(self):
        from mechferret import repl

        def fake_chat(agent, session, text, *, background=False):
            raise RuntimeError("boom")

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-error-hint.json"))
            try:
                job = runner.submit("broken prompt")
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.stop(wait=True)

        self.assertEqual(job.status, "error")
        self.assertEqual(job.error, "boom")
        rendered = out.getvalue()
        self.assertIn("error in queued #1: boom", rendered)
        self.assertIn("use /queue show #1", rendered)

    def test_repl_chat_job_runner_treats_missing_background_reply_as_error(self):
        from mechferret import repl

        def fake_chat(agent, session, text, *, background=False):
            return None

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-none-reply.json"))
            try:
                queued = runner.submit("missing queued reply")
                side = runner.submit_side(repl._btw_prompt("missing side reply"))
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.stop(wait=True)

        self.assertEqual(queued.status, "error")
        self.assertEqual(side.status, "error")
        self.assertEqual(queued.error, "no reply produced")
        self.assertEqual(side.error, "no reply produced")
        rendered = out.getvalue()
        self.assertIn("error in queued #1: no reply produced", rendered)
        self.assertIn("error in side #2: no reply produced", rendered)
        self.assertIn("use /queue show #1", rendered)
        self.assertIn("use /queue show #2", rendered)

    def test_repl_background_print_redisplays_readline_input_on_tty(self):
        import sys
        from unittest.mock import patch

        from mechferret import repl

        class TtyBuffer(StringIO):
            def isatty(self):
                return True

        class FakeReadline:
            def __init__(self) -> None:
                self.redisplays = 0

            def redisplay(self) -> None:
                self.redisplays += 1

        stdin = TtyBuffer()
        stdout = TtyBuffer()
        fake_readline = FakeReadline()
        with (
            patch.object(repl, "readline", fake_readline),
            patch.object(sys, "stdin", stdin),
            patch.object(sys, "stdout", stdout),
        ):
            repl._print_background("  finished #1")

        self.assertEqual(stdout.getvalue(), "\r\033[2K  finished #1\n")
        self.assertEqual(fake_readline.redisplays, 1)

    def test_repl_background_print_keeps_redirected_output_plain(self):
        from mechferret import repl

        out = StringIO()
        with redirect_stdout(out):
            repl._print_background("  finished #1")

        self.assertEqual(out.getvalue(), "  finished #1\n")

    def test_repl_background_chat_uses_background_output_for_missing_model(self):
        from unittest.mock import patch

        from mechferret import repl

        class Agent:
            configured = False

        emitted = []
        out = StringIO()
        with patch.object(repl, "_print_background", emitted.append), redirect_stdout(out):
            reply = repl._chat(Agent(), repl.Session(), "queued prompt", background=True)

        self.assertIsNone(reply)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(len(emitted), 1)
        self.assertIn("queued prompt needs a model", emitted[0])

    def test_repl_background_chat_uses_background_output_for_reply_and_cost(self):
        from unittest.mock import patch

        from mechferret import repl

        class Cost:
            def format_total(self):
                return "$0.0000"

        class Agent:
            configured = True
            cost = Cost()

            def send(self, text):
                return f"reply to {text}"

        emitted = []
        out = StringIO()
        with patch.object(repl, "_print_background", emitted.append), redirect_stdout(out):
            reply = repl._chat(Agent(), repl.Session(), "queued prompt", background=True)

        self.assertEqual(reply, "reply to queued prompt")
        self.assertEqual(out.getvalue(), "")
        self.assertTrue(any("reply to queued prompt" in line for line in emitted))
        self.assertTrue(any("$0.0000" in line for line in emitted))

    def test_repl_background_chat_uses_background_output_for_errors(self):
        from unittest.mock import patch

        from mechferret import repl

        class Cost:
            def format_total(self):
                return "$0.0000"

        class Agent:
            configured = True
            cost = Cost()

            def send(self, text):
                raise RuntimeError("401 authentication failed")

        emitted = []
        out = StringIO()
        with patch.object(repl, "_print_background", emitted.append), redirect_stdout(out):
            reply = repl._chat(Agent(), repl.Session(), "queued prompt", background=True)

        self.assertIsNone(reply)
        self.assertEqual(out.getvalue(), "")
        self.assertTrue(any("401 authentication failed" in line for line in emitted))
        self.assertTrue(any("API key may be invalid" in line for line in emitted))

    def test_repl_print_queued_shows_position_and_controls(self):
        from mechferret import repl

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=Path("queue-print.json"))
            try:
                runner.pause()
                first = runner.submit("first")
                second = runner.submit("second")

                repl._print_queued(first, runner)
                repl._print_queued(second, runner)
            finally:
                runner.resume()
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("queued #1 (position 1/2)", rendered)
        self.assertIn("queued #2 (position 2/2)", rendered)
        self.assertIn("/queue edit #1 <prompt>", rendered)
        self.assertIn("/queue move #2 first", rendered)
        self.assertIn("/queue cancel #2", rendered)

    def test_repl_input_prompt_summarizes_live_work(self):
        from mechferret import repl

        release = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            self.assertTrue(release.wait(timeout=2))
            return text

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-prompt.json"))
            try:
                self.assertEqual(repl._input_prompt(runner), "❯ ")
                runner.pause()
                self.assertIn("paused", repl._input_prompt(runner))
                runner.resume()
                active = runner.submit("active prompt")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                queued = runner.submit("queued prompt")
                side = runner.submit_side(repl._btw_prompt("side question"))
                runner.pause()
                prompt = repl._input_prompt(runner)
            finally:
                runner.resume()
                release.set()
                runner.stop(wait=True)

        self.assertEqual(active.id, 1)
        self.assertEqual(queued.id, 2)
        self.assertEqual(side.id, 3)
        self.assertIn("paused", prompt)
        self.assertIn("run#1", prompt)
        self.assertIn("btw:1", prompt)
        self.assertIn("q:1", prompt)

    def test_repl_input_prompt_shows_saved_queue_state(self):
        from mechferret import repl

        queue_path = Path("queue-prompt-saved.json")
        repl._save_queue_jobs(queue_path, [repl.PromptJob(id=7, text="saved prompt")])

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=queue_path)
            try:
                prompt = repl._input_prompt(runner)
            finally:
                runner.stop(wait=True)

        self.assertIn("saved:1", prompt)

    def test_repl_input_prompt_shows_ready_side_replies(self):
        from mechferret import repl

        def fake_chat(agent, session, text, *, background=False):
            return "side answer"

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-prompt-side-ready.json"))
            try:
                side = runner.submit_side(repl._btw_prompt("side question"))
                self.assertTrue(runner.wait_idle(timeout=2))
                prompt = repl._input_prompt(runner)
                side.applied = True
                applied_prompt = repl._input_prompt(runner)
            finally:
                runner.stop(wait=True)

        self.assertIn("btw-ready:1", prompt)
        self.assertNotIn("btw-ready", applied_prompt)

    def test_repl_status_bar_shows_ready_side_replies(self):
        from mechferret import repl

        class Agent:
            configured = False
            model = "offline"
            permission_mode = "auto"

        def fake_chat(agent, session, text, *, background=False):
            return "side answer"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-status-side-ready.json"))
            try:
                runner.submit_side(repl._btw_prompt("side question"))
                self.assertTrue(runner.wait_idle(timeout=2))
                repl._print_status_and_bar(Agent(), repl.Session(), runner)
            finally:
                runner.stop(wait=True)

        self.assertIn("btw-ready:1", out.getvalue())

    def test_repl_queue_view_shows_ready_side_replies(self):
        from mechferret import repl

        def fake_chat(agent, session, text, *, background=False):
            return "side answer"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-view-side-ready.json"))
            try:
                runner.submit_side(repl._btw_prompt("side question"))
                self.assertTrue(runner.wait_idle(timeout=2))
                repl._print_queue(runner)
            finally:
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertNotIn("queue empty", rendered)
        self.assertIn("ready   #1 btw: side question", rendered)
        self.assertIn("/queue apply side", rendered)
        self.assertNotIn("done     #1 btw: side question", rendered)

    def test_repl_queue_view_suggests_apply_all_for_multiple_ready_sides(self):
        from mechferret import repl

        def fake_chat(agent, session, text, *, background=False):
            return "side answer"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-view-side-ready-all.json"))
            try:
                runner.submit_side(repl._btw_prompt("first side"))
                runner.submit_side(repl._btw_prompt("second side"))
                self.assertTrue(runner.wait_idle(timeout=2))
                repl._print_queue(runner)
            finally:
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("ready   #1 btw: first side", rendered)
        self.assertIn("ready   #2 btw: second side", rendered)
        self.assertIn("/queue apply all", rendered)
        self.assertNotIn("/queue apply side", rendered)

    def test_repl_queue_view_does_not_duplicate_live_queued_jobs_as_saved(self):
        from mechferret import repl

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=Path("queue-no-saved-dupes.json"))
            try:
                runner.pause()
                runner.submit("one")
                runner.submit("two")
                repl._print_queue(runner)
            finally:
                runner.resume()
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("queued  #1", rendered)
        self.assertIn("queued  #2", rendered)
        self.assertNotIn("saved   #1", rendered)
        self.assertNotIn("saved   #2", rendered)

    def test_repl_queue_add_explicitly_enqueues_prompt_text(self):
        from mechferret import repl

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=Path("queue-add.json"))
            try:
                runner.pause()
                job = repl._queue_add(runner, "/paper draft from latest run")
                self.assertIsNotNone(job)
                assert job is not None
                self.assertEqual(job.status, "queued")
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job.text, "/paper draft from latest run")
        rendered = out.getvalue()
        self.assertIn("queued #1 (position 1/1)", rendered)
        self.assertIn("/queue edit #1 <prompt>", rendered)

    def test_repl_prompt_text_unwraps_single_outer_shell_quote(self):
        from mechferret import repl

        self.assertEqual(repl._prompt_text_after_words('/queue add "run /paper next"', 2), "run /paper next")
        self.assertEqual(repl._prompt_text_after_words("/queue edit #1 'new prompt'", 3), "new prompt")
        self.assertEqual(repl._prompt_text_after_words('/queue add say "hi" exactly', 2), 'say "hi" exactly')
        self.assertEqual(repl._prompt_text_after_words('/queue add "unfinished prompt', 2), '"unfinished prompt')

    def test_repl_queue_edit_accepts_quoted_replacement_prompt(self):
        from mechferret import repl

        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-edit-quoted.json"))
            try:
                runner.pause()
                job = runner.submit("old prompt")

                text = repl._prompt_text_after_words(f"/queue edit #{job.id} 'new prompt'", 3)
                repl._queue_edit(runner, [str(job.id)], text)
                self.assertEqual(job.text, "new prompt")

                runner.resume()
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertEqual(started, ["new prompt"])
        self.assertIn("edited #1", out.getvalue())

    def test_repl_btw_queue_views_show_user_prompt_not_internal_prefix(self):
        from mechferret import repl

        release = threading.Event()
        calls = []

        def fake_chat(agent, session, text, *, background=False):
            calls.append(text)
            self.assertTrue(release.wait(timeout=2))
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("btw-display.json"))
            try:
                side = runner.submit_side(repl._btw_prompt("side question"))
                deadline = time.monotonic() + 2
                while not runner.side_active() and time.monotonic() < deadline:
                    time.sleep(0.01)
                repl._print_queue(runner)
                repl._queue_show(runner, [str(side.id)])
            finally:
                release.set()
                runner.wait_idle(timeout=2)
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("side question", rendered)
        self.assertNotIn("Side request entered with /btw", rendered)
        self.assertIn("Side request entered with /btw", calls[0])

    def test_repl_btw_runs_while_main_prompt_is_active(self):
        from mechferret import repl

        release_main = threading.Event()
        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            if text == "main":
                self.assertTrue(release_main.wait(timeout=2))
            return text

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("btw-live.json"))
            try:
                main = runner.submit("main")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                side = runner.submit_side(repl._btw_prompt("side question"))
                deadline = time.monotonic() + 2
                while side.status == "running" and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(side.status, "done")
                self.assertEqual(main.status, "running")
                release_main.set()
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                release_main.set()
                runner.stop(wait=True)

        self.assertEqual(len(started), 2)
        self.assertEqual(started[0], "main")
        self.assertIn("Side request entered with /btw", started[1])

    def test_repl_queue_apply_promotes_finished_btw_to_main_context(self):
        from mechferret import repl

        class MainAgent:
            provider = "openai"
            messages: list[dict[str, object]]

            def __init__(self) -> None:
                self.messages = []

        def fake_chat(agent, session, text, *, background=False):
            return "side answer"

        agent = MainAgent()
        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(agent, repl.Session(), chat_fn=fake_chat, queue_path=Path("btw-apply.json"))
            try:
                side = runner.submit_side(repl._btw_prompt("side question"))
                self.assertTrue(runner.wait_idle(timeout=2))
                repl._queue_apply(runner, ["side"])
                repl._queue_apply(runner, [str(side.id)])
                repl._queue_show(runner, [str(side.id)])
            finally:
                runner.stop(wait=True)

        self.assertTrue(side.applied)
        self.assertEqual([message["role"] for message in agent.messages], ["system", "user", "assistant"])
        self.assertIn("Applied /btw side question #1", str(agent.messages[1]["content"]))
        self.assertIn("side question", str(agent.messages[1]["content"]))
        self.assertEqual(agent.messages[2]["content"], "side answer")
        rendered = out.getvalue()
        self.assertIn("use /queue apply #1", rendered)
        self.assertIn("applied side #1 to the main conversation", rendered)
        self.assertIn("side #1 is already applied", rendered)
        self.assertIn("applied to main conversation", rendered)

    def test_repl_queue_apply_waits_for_main_prompt_to_finish(self):
        from mechferret import repl

        class MainAgent:
            provider = "anthropic"

            def __init__(self) -> None:
                self.messages = []

        release_main = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            if text == "main":
                self.assertTrue(release_main.wait(timeout=2))
            return "reply"

        agent = MainAgent()
        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(agent, repl.Session(), chat_fn=fake_chat, queue_path=Path("btw-apply-busy.json"))
            try:
                runner.submit("main")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                side = runner.submit_side(repl._btw_prompt("side question"))
                deadline = time.monotonic() + 2
                while side.status == "running" and time.monotonic() < deadline:
                    time.sleep(0.01)
                repl._queue_apply(runner, ["side"])
            finally:
                release_main.set()
                runner.stop(wait=True)

        self.assertFalse(side.applied)
        self.assertEqual(agent.messages, [])
        self.assertIn("wait for the active prompt before applying side replies", out.getvalue())

    def test_repl_queue_apply_latest_targets_latest_side_not_latest_main(self):
        from mechferret import repl

        class MainAgent:
            provider = "anthropic"

            def __init__(self) -> None:
                self.messages = []

        calls = []

        def fake_chat(agent, session, text, *, background=False):
            calls.append(text)
            return f"reply for {text}"

        agent = MainAgent()
        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(agent, repl.Session(), chat_fn=fake_chat, queue_path=Path("btw-apply-latest.json"))
            try:
                side = runner.submit_side(repl._btw_prompt("side question"))
                self.assertTrue(runner.wait_idle(timeout=2))
                main = runner.submit("newer main")
                self.assertTrue(runner.wait_idle(timeout=2))

                self.assertGreater(main.id, side.id)
                repl._queue_apply(runner, ["latest"])
            finally:
                runner.stop(wait=True)

        self.assertTrue(side.applied)
        self.assertIn("applied side #1 to the main conversation", out.getvalue())
        self.assertEqual(agent.messages[-1]["content"], "reply for " + repl._btw_prompt("side question"))
        self.assertEqual(calls, [repl._btw_prompt("side question"), "newer main"])

    def test_repl_queue_apply_side_prefers_ready_side_over_running_side(self):
        from mechferret import repl

        class MainAgent:
            provider = "anthropic"

            def __init__(self) -> None:
                self.messages = []

        release_second = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            if "second side" in text:
                self.assertTrue(release_second.wait(timeout=2))
            return f"reply for {text}"

        agent = MainAgent()
        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(agent, repl.Session(), chat_fn=fake_chat, queue_path=Path("btw-apply-ready-side.json"))
            try:
                first = runner.submit_side(repl._btw_prompt("first side"))
                deadline = time.monotonic() + 2
                while first.status == "running" and time.monotonic() < deadline:
                    time.sleep(0.01)
                second = runner.submit_side(repl._btw_prompt("second side"))
                deadline = time.monotonic() + 2
                while not runner.side_active() and time.monotonic() < deadline:
                    time.sleep(0.01)

                repl._queue_apply(runner, ["side"])
            finally:
                release_second.set()
                runner.stop(wait=True)

        self.assertTrue(first.applied)
        self.assertFalse(second.applied)
        self.assertIn("applied side #1 to the main conversation", out.getvalue())

    def test_repl_queue_apply_all_promotes_ready_sides_in_order(self):
        from mechferret import repl

        class MainAgent:
            provider = "anthropic"

            def __init__(self) -> None:
                self.messages = []

        def fake_chat(agent, session, text, *, background=False):
            return f"reply for {text}"

        agent = MainAgent()
        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(agent, repl.Session(), chat_fn=fake_chat, queue_path=Path("btw-apply-all.json"))
            try:
                first = runner.submit_side(repl._btw_prompt("first side"))
                second = runner.submit_side(repl._btw_prompt("second side"))
                self.assertTrue(runner.wait_idle(timeout=2))

                repl._queue_apply(runner, ["all"])
            finally:
                runner.stop(wait=True)

        self.assertTrue(first.applied)
        self.assertTrue(second.applied)
        self.assertIn("applied side replies #1, #2", out.getvalue())
        self.assertEqual([message["role"] for message in agent.messages], ["user", "assistant", "user", "assistant"])
        self.assertIn("first side", str(agent.messages[0]["content"]))
        self.assertIn("second side", str(agent.messages[2]["content"]))

    def test_repl_queue_apply_all_reports_empty_ready_set(self):
        from mechferret import repl

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: "reply", queue_path=Path("btw-apply-all-empty.json"))
            try:
                repl._queue_apply(runner, ["all"])
            finally:
                runner.stop(wait=True)

        self.assertIn("no ready side replies to apply", out.getvalue())

    def test_repl_queue_apply_all_waits_for_main_prompt_to_finish(self):
        from mechferret import repl

        class MainAgent:
            provider = "anthropic"

            def __init__(self) -> None:
                self.messages = []

        release_main = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            if text == "main":
                self.assertTrue(release_main.wait(timeout=2))
            return "reply"

        agent = MainAgent()
        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(agent, repl.Session(), chat_fn=fake_chat, queue_path=Path("btw-apply-all-busy.json"))
            try:
                runner.submit("main")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                side = runner.submit_side(repl._btw_prompt("side question"))
                deadline = time.monotonic() + 2
                while side.status == "running" and time.monotonic() < deadline:
                    time.sleep(0.01)

                repl._queue_apply(runner, ["all"])
            finally:
                release_main.set()
                runner.stop(wait=True)

        self.assertFalse(side.applied)
        self.assertEqual(agent.messages, [])
        self.assertIn("wait for the active prompt before applying side replies", out.getvalue())

    def test_repl_queue_wait_waits_for_main_and_side_jobs(self):
        from mechferret import repl

        release = threading.Event()
        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            self.assertTrue(release.wait(timeout=2))
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-wait.json"))
            try:
                runner.submit("main")
                runner.submit_side(repl._btw_prompt("side"))
                releaser = threading.Thread(target=lambda: (time.sleep(0.05), release.set()))
                releaser.start()
                repl._queue_wait(runner, ["2"])
                releaser.join(timeout=2)
            finally:
                release.set()
                runner.stop(wait=True)

        self.assertFalse(runner.is_busy())
        self.assertEqual(len(started), 2)
        rendered = out.getvalue()
        self.assertIn("waiting for active or side work", rendered)
        self.assertIn("queue idle", rendered)

    def test_repl_queue_join_waits_for_one_job_and_shows_result_hint(self):
        from mechferret import repl

        release = threading.Event()
        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            self.assertTrue(release.wait(timeout=2))
            return f"reply for {text}"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-join.json"))
            try:
                job = runner.submit("main")
                releaser = threading.Thread(target=lambda: (time.sleep(0.05), release.set()))
                releaser.start()
                repl._queue_join(runner, [str(job.id), "2"])
                releaser.join(timeout=2)
            finally:
                release.set()
                runner.stop(wait=True)

        self.assertEqual(started, ["main"])
        self.assertEqual(job.status, "done")
        rendered = out.getvalue()
        self.assertIn("waiting for job #1", rendered)
        self.assertIn("job #1 done", rendered)
        self.assertIn("use /queue show #1", rendered)

    def test_repl_queue_join_active_latches_resolved_job_until_done(self):
        from mechferret import repl

        release = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            self.assertTrue(release.wait(timeout=2))
            return f"reply for {text}"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-join-active.json"))
            try:
                job = runner.submit("main")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                releaser = threading.Thread(target=lambda: (time.sleep(0.05), release.set()))
                releaser.start()
                repl._queue_join(runner, ["active", "2"])
                releaser.join(timeout=2)
            finally:
                release.set()
                runner.stop(wait=True)

        self.assertEqual(job.status, "done")
        rendered = out.getvalue()
        self.assertIn("waiting for job #1", rendered)
        self.assertIn("job #1 done", rendered)

    def test_repl_queue_join_running_can_target_side_jobs(self):
        from mechferret import repl

        release = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            self.assertTrue(release.wait(timeout=2))
            return f"reply for {text}"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-join-running-side.json"))
            try:
                side = runner.submit_side(repl._btw_prompt("side question"))
                deadline = time.monotonic() + 2
                while not runner.side_active() and time.monotonic() < deadline:
                    time.sleep(0.01)
                releaser = threading.Thread(target=lambda: (time.sleep(0.05), release.set()))
                releaser.start()
                repl._queue_join(runner, ["running", "2"])
                releaser.join(timeout=2)
            finally:
                release.set()
                runner.stop(wait=True)

        self.assertEqual(side.status, "done")
        rendered = out.getvalue()
        self.assertIn("waiting for job #1", rendered)
        self.assertIn("job #1 done", rendered)

    def test_repl_queue_side_alias_targets_btw_while_main_is_active(self):
        from mechferret import repl

        release_main = threading.Event()
        release_side = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            if text == "main":
                self.assertTrue(release_main.wait(timeout=2))
            else:
                self.assertTrue(release_side.wait(timeout=2))
            return f"reply for {text}"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-side-alias.json"))
            try:
                main = runner.submit("main")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                side = runner.submit_side(repl._btw_prompt("side question"))
                deadline = time.monotonic() + 2
                while not runner.side_active() and time.monotonic() < deadline:
                    time.sleep(0.01)

                found_side, saved = runner.find_job("side")
                self.assertIs(found_side, side)
                self.assertFalse(saved)
                self.assertIs(runner.find_job("active")[0], main)
                repl._queue_show(runner, ["side"])
                releaser = threading.Thread(target=lambda: (time.sleep(0.05), release_side.set()))
                releaser.start()
                repl._queue_join(runner, ["side", "2"])
                releaser.join(timeout=2)
            finally:
                release_side.set()
                release_main.set()
                runner.stop(wait=True)

        self.assertEqual(side.status, "done")
        self.assertEqual(main.status, "done")
        rendered = out.getvalue()
        self.assertIn("job #2", rendered)
        self.assertIn("side question", rendered)
        self.assertIn("waiting for job #2", rendered)
        self.assertIn("job #2 done", rendered)

    def test_repl_queue_side_alias_targets_latest_finished_btw(self):
        from mechferret import repl

        calls = []

        def fake_chat(agent, session, text, *, background=False):
            calls.append(text)
            return f"reply for {text}"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-side-finished.json"))
            try:
                first = runner.submit_side(repl._btw_prompt("first side"))
                second = runner.submit_side(repl._btw_prompt("second side"))
                self.assertTrue(runner.wait_idle(timeout=2))
                found_side, saved = runner.find_job("side")
                self.assertIs(found_side, second)
                self.assertFalse(saved)
                repl._queue_show(runner, ["side"])
                repl._queue_retry(runner, ["btw"])
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.stop(wait=True)

        self.assertEqual(first.status, "done")
        self.assertEqual(second.status, "done")
        self.assertGreaterEqual(sum("second side" in call for call in calls), 2)
        rendered = out.getvalue()
        self.assertIn("job #2", rendered)
        self.assertIn("second side", rendered)
        self.assertIn("retried #2 as #3", rendered)

    def test_repl_queue_join_times_out_or_refuses_paused_jobs(self):
        from mechferret import repl

        release = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            self.assertTrue(release.wait(timeout=2))
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-join-timeout.json"))
            try:
                running = runner.submit("running")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                repl._queue_join(runner, [str(running.id), "0.05"])

                queued = runner.submit("queued")
                runner.pause()
                repl._queue_join(runner, [str(queued.id), "1"])
            finally:
                runner.resume()
                release.set()
                runner.wait_idle(timeout=2)
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("job #1 still running after timeout", rendered)
        self.assertIn("queue paused; use /queue resume", rendered)

    def test_repl_queue_join_restores_saved_job_and_waits(self):
        from mechferret import repl

        queue_path = Path("queue-join-saved.json")
        release = threading.Event()
        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            self.assertTrue(release.wait(timeout=2))
            return f"reply for {text}"

        repl._save_queue_jobs(queue_path, [repl.PromptJob(id=9, text="saved join prompt")])

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=queue_path)
            try:
                releaser = threading.Thread(target=lambda: (time.sleep(0.05), release.set()))
                releaser.start()
                repl._queue_join(runner, ["9", "2"])
                releaser.join(timeout=2)
            finally:
                release.set()
                runner.stop(wait=True)

        self.assertEqual(started, ["saved join prompt"])
        self.assertEqual(runner.saved(), [])
        rendered = out.getvalue()
        self.assertIn("restored #9", rendered)
        self.assertIn("waiting for job #9", rendered)
        self.assertIn("job #9 done", rendered)

    def test_repl_queue_wait_allows_running_work_while_paused_without_queued_jobs(self):
        from mechferret import repl

        release = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            self.assertTrue(release.wait(timeout=2))
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-wait-paused-running.json"))
            try:
                runner.submit_side(repl._btw_prompt("side question"))
                deadline = time.monotonic() + 2
                while not runner.side_active() and time.monotonic() < deadline:
                    time.sleep(0.01)
                runner.pause()
                releaser = threading.Thread(target=lambda: (time.sleep(0.05), release.set()))
                releaser.start()
                repl._queue_wait(runner, ["2"])
                releaser.join(timeout=2)
            finally:
                release.set()
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("waiting for active or side work", rendered)
        self.assertIn("queue idle", rendered)
        self.assertNotIn("use /queue resume", rendered)

    def test_repl_queue_show_renders_prompt_reply_and_saved_jobs(self):
        from mechferret import repl

        def fake_chat(agent, session, text, *, background=False):
            return f"reply for {text}"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-show.json"))
            try:
                job = runner.submit("full prompt text")
                self.assertTrue(runner.wait_idle(timeout=2))
                repl._queue_show(runner, [str(job.id)])
                repl._save_queue_jobs(Path("queue-show.json"), [repl.PromptJob(id=9, text="saved prompt", kind="btw")])
                repl._queue_show(runner, ["9"])
            finally:
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("job #1", rendered)
        self.assertIn("full prompt text", rendered)
        self.assertIn("reply for full prompt text", rendered)
        self.assertIn("job #9 saved", rendered)
        self.assertIn("saved prompt", rendered)

    def test_repl_queue_saved_aliases_resolve_side_and_running_jobs(self):
        from mechferret import repl

        queue_path = Path("queue-saved-side-running.json")
        main_running = repl.PromptJob(id=3, text="main running prompt", status="running", created_at=100.0)
        side_old = repl.PromptJob(id=5, text=repl._btw_prompt("old side prompt"), kind="btw", status="queued", created_at=150.0)
        side_new = repl.PromptJob(id=8, text=repl._btw_prompt("new side prompt"), kind="btw", status="running", created_at=200.0)
        repl._save_queue_jobs(queue_path, [main_running, side_old, side_new])

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: "reply", queue_path=queue_path)
            try:
                side, side_saved = runner.find_job("side")
                running, running_saved = runner.find_job("running")
                self.assertIsNotNone(side)
                self.assertIsNotNone(running)
                self.assertEqual(side.id, side_new.id)
                self.assertEqual(running.id, side_new.id)
                self.assertTrue(side_saved)
                self.assertTrue(running_saved)

                repl._queue_show(runner, ["side"])
                repl._queue_retry(runner, ["btw"])
                self.assertTrue(runner.wait_idle(timeout=2))
                restored = runner.restore_saved("running")
                self.assertEqual([job.id for job in restored], [side_new.id])
            finally:
                runner.stop(wait=True)

        self.assertEqual([job.id for job in repl._load_saved_queue(queue_path)], [main_running.id, side_old.id])
        rendered = out.getvalue()
        self.assertIn("job #8 saved", rendered)
        self.assertIn("new side prompt", rendered)
        self.assertIn("retried #8 saved", rendered)

    def test_repl_queue_saved_aliases_resolve_latest_and_next_jobs(self):
        from mechferret import repl

        queue_path = Path("queue-saved-aliases.json")
        old = repl.PromptJob(id=3, text="older saved prompt", created_at=100.0)
        new = repl.PromptJob(id=8, text="newer saved prompt", created_at=200.0)
        repl._save_queue_jobs(queue_path, [old, new])

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: "reply", queue_path=queue_path)
            try:
                latest, latest_saved = runner.find_job("latest")
                next_job, next_saved = runner.find_job("next")
                self.assertIsNotNone(latest)
                self.assertIsNotNone(next_job)
                self.assertEqual(latest.id, new.id)
                self.assertEqual(next_job.id, old.id)
                self.assertTrue(latest_saved)
                self.assertTrue(next_saved)

                repl._queue_show(runner, ["latest"])
                repl._queue_retry(runner, ["latest"])
                self.assertTrue(runner.wait_idle(timeout=2))
                repl._queue_retry(runner, ["next"])
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("job #8 saved", rendered)
        self.assertIn("newer saved prompt", rendered)
        self.assertIn("retried #8 saved as #9", rendered)
        self.assertIn("retried #3 saved as #10", rendered)

    def test_repl_saved_queue_preserves_pending_statuses(self):
        from mechferret import repl

        queue_path = Path("queue-saved-status.json")
        repl._save_queue_jobs(queue_path, [
            repl.PromptJob(id=1, text="active saved prompt", status="running"),
            repl.PromptJob(id=2, text="queued saved prompt", status="queued"),
            repl.PromptJob(id=3, text="bad saved prompt", status="done"),
        ])

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: "reply", queue_path=queue_path)
            try:
                saved = runner.saved()
                self.assertEqual([job.status for job in saved], ["running", "queued", "queued"])
                repl._queue_show(runner, ["1"])
            finally:
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("job #1 saved", rendered)
        self.assertIn("running", rendered)

    def test_repl_queue_edit_updates_saved_queued_prompts(self):
        from mechferret import repl

        queue_path = Path("queue-saved-edit.json")
        repl._save_queue_jobs(queue_path, [
            repl.PromptJob(id=3, text="older saved prompt", created_at=100.0),
            repl.PromptJob(id=8, text="newer saved prompt", created_at=200.0),
            repl.PromptJob(id=9, text="running saved prompt", status="running", created_at=300.0),
        ])

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: "reply", queue_path=queue_path)
            try:
                repl._queue_edit(runner, ["next"], "updated older saved prompt")
                repl._queue_edit(runner, ["8"], "updated newer saved prompt")
                repl._queue_edit(runner, ["latest"], "should not update running prompt")
                repl._queue_show(runner, ["3"])
                repl._queue_show(runner, ["8"])
                repl._queue_show(runner, ["9"])
            finally:
                runner.stop(wait=True)

        saved = repl._load_saved_queue(queue_path)
        self.assertEqual([job.text for job in saved], [
            "updated older saved prompt",
            "updated newer saved prompt",
            "running saved prompt",
        ])
        rendered = out.getvalue()
        self.assertIn("edited #3", rendered)
        self.assertIn("edited #8", rendered)
        self.assertIn("job #9 is running; only queued prompts can be edited.", rendered)
        self.assertIn("updated older saved prompt", rendered)
        self.assertIn("updated newer saved prompt", rendered)

    def test_repl_queue_cancel_removes_saved_aliases(self):
        from mechferret import repl

        queue_path = Path("queue-saved-cancel.json")
        old = repl.PromptJob(id=3, text="older saved prompt", created_at=100.0)
        new = repl.PromptJob(id=8, text="newer saved prompt", created_at=200.0)
        repl._save_queue_jobs(queue_path, [old, new])

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: "reply", queue_path=queue_path)
            try:
                repl._queue_cancel(runner, ["next"])
                self.assertEqual([job.id for job in runner.saved()], [new.id])
                repl._queue_cancel(runner, ["latest"])
                self.assertEqual(runner.saved(), [])
            finally:
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("canceled #3", rendered)
        self.assertIn("canceled #8", rendered)

    def test_repl_queue_move_reorders_saved_queued_prompts(self):
        from mechferret import repl

        queue_path = Path("queue-saved-move.json")
        old = repl.PromptJob(id=3, text="older saved prompt", created_at=100.0)
        mid = repl.PromptJob(id=5, text="middle saved prompt", created_at=150.0)
        new = repl.PromptJob(id=8, text="newer saved prompt", created_at=200.0)
        running = repl.PromptJob(id=9, text="running saved prompt", status="running", created_at=300.0)
        repl._save_queue_jobs(queue_path, [old, mid, new, running])

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: "reply", queue_path=queue_path)
            try:
                repl._queue_move(runner, [str(new.id), "first"])
                self.assertEqual([job.id for job in runner.saved()], [new.id, old.id, mid.id, running.id])
                repl._queue_move(runner, ["next", "after", "5"])
                self.assertEqual([job.id for job in runner.saved()], [old.id, mid.id, new.id, running.id])
                repl._queue_move(runner, ["latest", "last"])
                self.assertEqual([job.id for job in runner.saved()], [old.id, mid.id, new.id, running.id])
                repl._queue_move(runner, [str(old.id), "last"])
                self.assertEqual([job.id for job in runner.saved()], [mid.id, new.id, old.id, running.id])
            finally:
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("moved #8 first", rendered)
        self.assertIn("moved #8 after #5", rendered)
        self.assertIn("job #9 is running; only queued prompts can be moved.", rendered)
        self.assertIn("moved #3 last", rendered)

    def test_repl_queue_latest_targets_most_recent_live_job(self):
        from mechferret import repl

        calls = []

        def fake_chat(agent, session, text, *, background=False):
            calls.append(text)
            return f"reply for {text}"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-latest.json"))
            try:
                runner.submit("first")
                latest = runner.submit("second")
                self.assertTrue(runner.wait_idle(timeout=2))

                found, saved = runner.find_job("latest")
                self.assertIs(found, latest)
                self.assertFalse(saved)
                repl._queue_show(runner, ["latest"])
                repl._queue_retry(runner, ["last"])
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.stop(wait=True)

        self.assertEqual(calls.count("second"), 2)
        rendered = out.getvalue()
        self.assertIn("job #2", rendered)
        self.assertIn("second", rendered)
        self.assertIn("retried #2 as #3", rendered)

    def test_repl_queue_usage_mentions_supported_aliases(self):
        from mechferret import repl

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=Path("queue-usage.json"))
            try:
                repl._queue_add(runner, "")
                repl._queue_show(runner, [])
                repl._queue_retry(runner, [])
                repl._queue_edit(runner, [], "")
                repl._queue_move(runner, [])
                repl._queue_join(runner, [])
                repl._queue_cancel(runner, [])
            finally:
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("/queue add <prompt>", rendered)
        self.assertIn("/queue show <job id|latest|active|running|side|next>", rendered)
        self.assertIn("/queue retry <job id|latest|running|side|next>", rendered)
        self.assertIn("/queue edit <job id|latest|next> <new prompt>", rendered)
        self.assertIn("/queue move <job id|latest|next>", rendered)
        self.assertIn("/queue join <job id|latest|active|running|side|next> [seconds]", rendered)
        self.assertIn("/queue cancel <job id|latest|next|all>", rendered)

    def test_repl_queue_latest_targets_live_mutations(self):
        from mechferret import repl

        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-latest-mutations.json"))
            try:
                runner.pause()
                first = runner.submit("first")
                second = runner.submit("second")

                repl._queue_edit(runner, ["latest"], "updated second")
                self.assertEqual(second.text, "updated second")
                repl._queue_move(runner, ["latest", "first"])
                self.assertEqual([job.id for job in runner.queued()], [second.id, first.id])
                repl._queue_cancel(runner, ["latest"])
                self.assertEqual(second.status, "canceled")

                runner.resume()
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertEqual(started, ["first"])
        rendered = out.getvalue()
        self.assertIn("edited #2", rendered)
        self.assertIn("moved #2 first", rendered)
        self.assertIn("canceled #2", rendered)

    def test_repl_queue_latest_stays_chronological_after_reorder(self):
        from mechferret import repl

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=Path("queue-latest-order.json"))
            try:
                runner.pause()
                first = runner.submit("first")
                second = runner.submit("second")
                third = runner.submit("third")

                repl._queue_move(runner, [str(third.id), "first"])
                self.assertEqual([job.id for job in runner.queued()], [third.id, first.id, second.id])
                self.assertIs(runner.find_job("next")[0], third)
                self.assertIs(runner.find_job("latest")[0], third)
                self.assertEqual([job.id for job in runner.recent(3)], [first.id, second.id, third.id])
            finally:
                runner.resume()
                runner.stop(wait=True)

    def test_repl_queue_active_and_next_targets_resolve_live_jobs(self):
        from mechferret import repl

        release = threading.Event()
        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            if text == "active prompt":
                self.assertTrue(release.wait(timeout=2))
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-active-next.json"))
            try:
                active = runner.submit("active prompt")
                queued = runner.submit("queued prompt")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)

                found_active, saved_active = runner.find_job("active")
                found_next, saved_next = runner.find_job("next")
                self.assertIs(found_active, active)
                self.assertFalse(saved_active)
                self.assertIs(found_next, queued)
                self.assertFalse(saved_next)

                repl._queue_show(runner, ["running"])
                repl._queue_edit(runner, ["next"], "updated queued")
                self.assertEqual(queued.text, "updated queued")
                repl._queue_cancel(runner, ["next"])
                self.assertEqual(queued.status, "canceled")
            finally:
                release.set()
                runner.wait_idle(timeout=2)
                runner.stop(wait=True)

        self.assertEqual(started, ["active prompt"])
        rendered = out.getvalue()
        self.assertIn("job #1", rendered)
        self.assertIn("active prompt", rendered)
        self.assertIn("edited #2", rendered)
        self.assertIn("canceled #2", rendered)

    def test_repl_queue_retry_requeues_main_side_and_saved_jobs(self):
        from mechferret import repl

        calls = []

        def fake_chat(agent, session, text, *, background=False):
            calls.append(text)
            return f"reply for {text}"

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-retry.json"))
            try:
                main = runner.submit("main retry prompt")
                side = runner.submit_side(repl._btw_prompt("side retry prompt"))
                self.assertTrue(runner.wait_idle(timeout=2))
                repl._queue_retry(runner, [str(main.id)])
                repl._queue_retry(runner, [str(side.id)])
                repl._save_queue_jobs(Path("queue-retry.json"), [repl.PromptJob(id=9, text="saved retry prompt", kind="prompt")])
                repl._queue_retry(runner, ["9"])
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.stop(wait=True)

        self.assertGreaterEqual(calls.count("main retry prompt"), 2)
        self.assertGreaterEqual(calls.count("saved retry prompt"), 1)
        self.assertGreaterEqual(sum("side retry prompt" in call for call in calls), 2)
        rendered = out.getvalue()
        self.assertIn("retried #1 as #3", rendered)
        self.assertIn("retried #2 as #4", rendered)
        self.assertIn("retried #9 saved", rendered)

    def test_repl_queue_retry_refuses_active_or_queued_jobs(self):
        from mechferret import repl

        started = []
        release = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            if text == "first":
                self.assertTrue(release.wait(timeout=2))
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-retry-active.json"))
            try:
                first = runner.submit("first")
                second = runner.submit("second")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertIsNotNone(runner.active())

                repl._queue_retry(runner, [str(first.id)])
                repl._queue_retry(runner, [str(second.id)])

                self.assertEqual([job.id for job in runner.recent(10)], [first.id, second.id])
            finally:
                release.set()
                runner.wait_idle(timeout=2)
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("job #1 is running", rendered)
        self.assertIn("job #2 is queued", rendered)
        self.assertEqual(started, ["first", "second"])

    def test_repl_queue_edit_updates_prompt_before_it_starts(self):
        from mechferret import repl

        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-edit.json"))
            try:
                runner.pause()
                job = runner.submit("old prompt")

                repl._queue_edit(runner, [str(job.id)], "new prompt with details")
                repl._queue_show(runner, [str(job.id)])

                runner.resume()
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertEqual(started, ["new prompt with details"])
        rendered = out.getvalue()
        self.assertIn("edited #1", rendered)
        self.assertIn("new prompt with details", rendered)
        self.assertNotIn("old prompt", rendered)

    def test_repl_queue_edit_refuses_running_or_finished_jobs(self):
        from mechferret import repl

        release = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            if text == "running prompt":
                self.assertTrue(release.wait(timeout=2))
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-edit-active.json"))
            try:
                running = runner.submit("running prompt")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertIsNotNone(runner.active())

                repl._queue_edit(runner, [str(running.id)], "changed running prompt")
                release.set()
                self.assertTrue(runner.wait_idle(timeout=2))
                repl._queue_edit(runner, [str(running.id)], "changed finished prompt")
            finally:
                release.set()
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("job #1 is running", rendered)
        self.assertIn("job #1 is done", rendered)
        self.assertEqual(running.text, "running prompt")

    def test_repl_queue_move_reorders_paused_prompts_before_running(self):
        from mechferret import repl

        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-move.json"))
            try:
                runner.pause()
                first = runner.submit("first")
                second = runner.submit("second")
                third = runner.submit("third")
                time.sleep(0.1)

                repl._queue_move(runner, [str(third.id), "first"])
                repl._queue_move(runner, [str(first.id), "after", str(second.id)])
                self.assertEqual([job.id for job in runner.queued()], [third.id, second.id, first.id])

                runner.resume()
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertEqual(started, ["third", "second", "first"])
        rendered = out.getvalue()
        self.assertIn("moved #3 first", rendered)
        self.assertIn("moved #1 after #2", rendered)

    def test_repl_queue_move_refuses_nonqueued_or_missing_anchor(self):
        from mechferret import repl

        release = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            if text == "running":
                self.assertTrue(release.wait(timeout=2))
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-move-active.json"))
            try:
                running = runner.submit("running")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertIsNotNone(runner.active())

                repl._queue_move(runner, [str(running.id), "first"])
                queued = runner.submit("queued")
                repl._queue_move(runner, [str(queued.id), "before", "999"])
            finally:
                release.set()
                runner.wait_idle(timeout=2)
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("job #1 is running", rendered)
        self.assertIn("no queued anchor matched '999'", rendered)

    def test_repl_queue_move_self_anchor_is_noop(self):
        from mechferret import repl

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=Path("queue-move-self.json"))
            try:
                runner.pause()
                first = runner.submit("first")
                second = runner.submit("second")

                repl._queue_move(runner, [str(first.id), "before", str(first.id)])
                repl._queue_move(runner, [str(second.id), "after", str(second.id)])

                self.assertEqual([job.id for job in runner.queued()], [first.id, second.id])
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertIn("job #1 is already in that spot", out.getvalue())
        self.assertIn("job #2 is already in that spot", out.getvalue())

    def test_repl_queue_pause_holds_prompts_until_resume(self):
        from mechferret import repl

        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-pause.json"))
            try:
                self.assertTrue(runner.pause())
                first = runner.submit("first")
                second = runner.submit("second")
                time.sleep(0.1)

                self.assertTrue(runner.paused())
                self.assertEqual(started, [])
                self.assertEqual([job.id for job in runner.queued()], [first.id, second.id])
                repl._print_queue(runner)
                repl._queue_wait(runner, ["1"])

                self.assertTrue(runner.resume())
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertEqual(started, ["first", "second"])
        rendered = out.getvalue()
        self.assertIn("queue paused", rendered)
        self.assertIn("queued  #1", rendered)
        self.assertIn("queued  #2", rendered)
        self.assertIn("use /queue resume", rendered)

    def test_repl_chat_job_runner_cancels_pending_prompts(self):
        from mechferret import repl

        started = []
        release = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            if text == "first":
                self.assertTrue(release.wait(timeout=2))
            return text

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue.json"))
            try:
                first = runner.submit("first")
                second = runner.submit("second")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                canceled = runner.cancel(str(second.id))
                self.assertEqual(canceled, [second])
                self.assertEqual(second.status, "canceled")
                self.assertEqual(runner.queued(), [])
                release.set()
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                release.set()
                runner.stop(wait=True)

        self.assertEqual(first.status, "done")
        self.assertEqual(second.status, "canceled")
        self.assertEqual(started, ["first"])

    def test_repl_queue_cancel_alias_cancels_pending_prompts(self):
        from mechferret import repl

        started = []
        release = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            if text == "first":
                self.assertTrue(release.wait(timeout=2))
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=Path("queue-cancel-alias.json"))
            try:
                runner.submit("first")
                second = runner.submit("second")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)

                repl._queue_cancel(runner, ["next"])
                release.set()
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                release.set()
                runner.stop(wait=True)

        self.assertEqual(second.status, "canceled")
        self.assertEqual(started, ["first"])
        self.assertIn("canceled #2", out.getvalue())

    def test_save_queue_jobs_serializes_concurrent_writes_to_same_path(self):
        from mechferret import repl

        queue_path = Path("queue-concurrent.json")
        thread_count = 12
        barrier = threading.Barrier(thread_count + 1)
        errors = []

        def writer(index):
            try:
                barrier.wait(timeout=2)
                repl._save_queue_jobs(queue_path, [repl.PromptJob(id=index + 1, text=f"prompt {index}")])
            except Exception as exc:  # noqa: BLE001 - test reports cross-thread failures
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(index,)) for index in range(thread_count)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=2)
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(errors, [])
        jobs = repl._load_saved_queue(queue_path)
        self.assertEqual(len(jobs), 1)
        self.assertTrue(jobs[0].text.startswith("prompt "))
        self.assertEqual(list(queue_path.parent.glob(f".{queue_path.name}.*.tmp")), [])

    def test_repl_queue_clear_scopes_live_and_saved_queue_state(self):
        from mechferret import repl

        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            return text

        queue_path = Path("queue-clear-scopes.json")
        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=queue_path)
            try:
                runner.pause()
                first = runner.submit("first")
                second = runner.submit("second")

                repl._queue_clear(runner, ["queued"])
                self.assertEqual([first.status, second.status], ["canceled", "canceled"])
                self.assertEqual(runner.saved(), [])

                repl._queue_clear(runner, ["all"])
            finally:
                runner.resume()
                runner.stop(wait=True)

            repl._save_queue_jobs(queue_path, [repl.PromptJob(id=9, text="saved prompt")])
            repl._queue_clear(runner, ["saved"])
            self.assertEqual(runner.saved(), [])

        self.assertEqual(started, [])
        rendered = out.getvalue()
        self.assertIn("canceled #1, #2", rendered)
        self.assertIn("cleared 1 saved queued prompt", rendered)
        self.assertIn("no queued prompts to cancel", rendered)

    def test_repl_queue_clear_saved_preserves_live_queue_persistence(self):
        from mechferret import repl

        queue_path = Path("queue-clear-saved-live.json")

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=queue_path)
            try:
                runner.pause()
                live = runner.submit("live queued prompt")
                repl._save_queue_jobs(queue_path, [
                    live,
                    repl.PromptJob(id=9, text="saved prompt"),
                ])
                repl._queue_clear(runner, ["saved"])
                persisted = runner.saved()
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertEqual([job.id for job in persisted], [live.id])
        self.assertEqual(persisted[0].text, "live queued prompt")
        self.assertIn("cleared 1 saved queued prompt", out.getvalue())

    def test_repl_queue_clear_saved_counts_id_collisions_as_saved(self):
        from mechferret import repl

        queue_path = Path("queue-clear-saved-collision.json")

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=queue_path)
            try:
                runner.pause()
                live = runner.submit("live queued prompt")
                repl._save_queue_jobs(queue_path, [
                    live,
                    repl.PromptJob(id=live.id, text="different saved prompt"),
                ])
                repl._queue_clear(runner, ["saved"])
                persisted = runner.saved()
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertEqual([job.text for job in persisted], ["live queued prompt"])
        self.assertIn("cleared 1 saved queued prompt", out.getvalue())

    def test_repl_saved_only_keeps_distinct_saved_id_collisions_visible(self):
        from mechferret import repl

        queue_path = Path("queue-saved-visible-collision.json")

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=queue_path)
            try:
                runner.pause()
                live = runner.submit("live queued prompt")
                repl._save_queue_jobs(queue_path, [
                    live,
                    repl.PromptJob(id=live.id, text="different saved prompt"),
                ])
                prompt = repl._input_prompt(runner)
                saved = runner.saved_only()
                repl._print_queue(runner)
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertEqual([job.text for job in saved], ["different saved prompt"])
        self.assertIn("saved:1", prompt)
        rendered = out.getvalue()
        self.assertIn("saved   #1", rendered)
        self.assertIn("different saved prompt", rendered)

    def test_repl_save_pending_preserves_distinct_saved_id_collisions(self):
        from mechferret import repl

        queue_path = Path("queue-save-pending-collision.json")

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=queue_path)
            try:
                runner.pause()
                live = runner.submit("live queued prompt")
                repl._save_queue_jobs(queue_path, [
                    live,
                    repl.PromptJob(id=live.id, text="different saved prompt"),
                ])
                runner._preserved_saved_ids.add(live.id)
                runner.save_pending()
                persisted = runner.saved()
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertEqual([job.text for job in persisted], ["live queued prompt", "different saved prompt"])

    def test_repl_restore_one_saved_preserves_distinct_saved_id_collisions(self):
        from mechferret import repl

        queue_path = Path("restore-one-saved-collision.json")

        repl._save_queue_jobs(queue_path, [
            repl.PromptJob(id=3, text="selected saved", created_at=1.0),
            repl.PromptJob(id=3, text="other saved with same id", created_at=2.0),
        ])

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: "reply", queue_path=queue_path)
            try:
                restored = runner.restore_saved("3")
                self.assertTrue(runner.wait_idle(timeout=2))
                saved = runner.saved()
            finally:
                runner.stop(wait=True)

        self.assertEqual([job.text for job in restored], ["selected saved"])
        self.assertEqual([job.text for job in saved], ["other saved with same id"])

    def test_repl_cancel_one_saved_preserves_distinct_saved_id_collisions(self):
        from mechferret import repl

        queue_path = Path("cancel-one-saved-collision.json")

        repl._save_queue_jobs(queue_path, [
            repl.PromptJob(id=3, text="selected saved", created_at=1.0),
            repl.PromptJob(id=3, text="other saved with same id", created_at=2.0),
        ])

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=queue_path)
            try:
                canceled = runner.cancel("3")
                saved = runner.saved()
            finally:
                runner.stop(wait=True)

        self.assertEqual([job.text for job in canceled], ["selected saved"])
        self.assertEqual([job.text for job in saved], ["other saved with same id"])

    def test_repl_join_saved_alias_preserves_distinct_saved_id_collision_target(self):
        from mechferret import repl

        queue_path = Path("join-saved-collision-alias.json")
        repl._save_queue_jobs(queue_path, [
            repl.PromptJob(id=3, text="older saved", created_at=1.0),
            repl.PromptJob(id=3, text="newer saved", created_at=2.0),
        ])

        def fake_chat(_agent, _session, text, *, background=False):
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=queue_path)
            try:
                repl._queue_join(runner, ["latest", "2"])
                saved = runner.saved()
            finally:
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("restored #3", rendered)
        self.assertIn("job #3 done", rendered)
        self.assertEqual([job.text for job in runner.recent()], ["newer saved"])
        self.assertEqual([job.text for job in saved], ["older saved"])

    def test_repl_chat_job_runner_saves_and_restores_pending_prompts(self):
        from mechferret import repl

        queue_path = Path("saved-queue.json")
        release = threading.Event()
        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            if text == "first":
                self.assertTrue(release.wait(timeout=2))
            return text

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=queue_path)
            try:
                runner.submit("first")
                second = runner.submit("second")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual([job.text for job in runner.saved()], ["second"])
                self.assertTrue(queue_path.exists())
                restored_runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=queue_path)
                restored = restored_runner.restore_saved()
                self.assertEqual([job.text for job in restored], [second.text])
                self.assertEqual([job.id for job in restored], [second.id])
                followup = restored_runner.submit("third")
                self.assertEqual(followup.id, second.id + 1)
                self.assertEqual(runner.cancel(str(second.id)), [second])
                release.set()
                self.assertTrue(restored_runner.wait_idle(timeout=2))
                self.assertTrue(runner.wait_idle(timeout=2))
                self.assertEqual(restored_runner.saved(), [])
            finally:
                release.set()
                runner.stop(wait=True)
                if "restored_runner" in locals():
                    restored_runner.stop(wait=True)

        self.assertEqual(sorted(started), ["first", "second", "third"])

    def test_repl_restore_saved_queue_skips_already_live_jobs(self):
        from mechferret import repl

        queue_path = Path("restore-skips-live.json")

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: None, queue_path=queue_path)
            try:
                runner.pause()
                live = runner.submit("live queued prompt")
                repl._save_queue_jobs(queue_path, [
                    live,
                    repl.PromptJob(id=9, text="saved prompt"),
                ])
                restored = runner.restore_saved("all")
                queued_texts = [job.text for job in runner.queued()]
                persisted = runner.saved()
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertEqual([job.text for job in restored], ["saved prompt"])
        self.assertEqual(queued_texts, ["live queued prompt", "saved prompt"])
        self.assertEqual([job.text for job in persisted], queued_texts)

    def test_repl_restore_saved_queue_ids_avoid_live_collisions(self):
        from mechferret import repl

        queue_path = Path("restore-id-collision.json")

        def fake_chat(agent, session, text, *, background=False):
            return text

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=queue_path)
            try:
                runner.pause()
                live = runner.submit("live")
                self.assertEqual(live.id, 1)
                repl._save_queue_jobs(queue_path, [
                    repl.PromptJob(id=1, text="saved one"),
                    repl.PromptJob(id=8, text="saved eight"),
                ])
                restored = runner.restore_saved()
                followup = runner.submit("followup")
                runner.resume()
                self.assertTrue(runner.wait_idle(timeout=2))
            finally:
                runner.resume()
                runner.stop(wait=True)

        self.assertEqual([job.text for job in restored], ["saved one", "saved eight"])
        self.assertEqual([job.id for job in restored], [2, 8])
        self.assertEqual(followup.id, 9)

    def test_repl_restore_saved_queue_can_target_one_prompt(self):
        from mechferret import repl

        queue_path = Path("restore-one-saved.json")
        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            return text

        repl._save_queue_jobs(queue_path, [
            repl.PromptJob(id=3, text="old saved", created_at=1.0),
            repl.PromptJob(id=8, text="new saved", created_at=2.0),
        ])

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=queue_path)
            try:
                restored = runner.restore_saved("next")
                self.assertTrue(runner.wait_idle(timeout=2))
                followup = runner.submit("fresh")
                self.assertTrue(runner.wait_idle(timeout=2))
                saved = runner.saved()
            finally:
                runner.stop(wait=True)

        self.assertEqual([job.text for job in restored], ["old saved"])
        self.assertEqual([job.id for job in restored], [3])
        self.assertEqual(followup.id, 9)
        self.assertEqual(started, ["old saved", "fresh"])
        self.assertEqual([job.text for job in saved], ["new saved"])

    def test_repl_restore_saved_queue_preserves_latest_order(self):
        from mechferret import repl

        queue_path = Path("restore-latest-order.json")

        repl._save_queue_jobs(queue_path, [
            repl.PromptJob(id=3, text="old saved", created_at=100.0),
            repl.PromptJob(id=8, text="new saved", created_at=200.0),
        ])

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=lambda *args, **kwargs: "reply", queue_path=queue_path)
            try:
                restored = runner.restore_saved("all")
                self.assertTrue(runner.wait_idle(timeout=2))
                latest, saved = runner.find_job("latest")
            finally:
                runner.stop(wait=True)

        self.assertEqual([job.created_at for job in restored], [100.0, 200.0])
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertFalse(saved)
        self.assertEqual(latest.id, 8)

    def test_repl_chat_job_runner_saves_active_prompt_on_fast_shutdown(self):
        from mechferret import repl

        queue_path = Path("active-queue.json")
        release = threading.Event()
        started = []

        def fake_chat(agent, session, text, *, background=False):
            started.append(text)
            self.assertTrue(release.wait(timeout=2))
            return text

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=queue_path)
            try:
                runner.submit("active prompt")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                runner.stop(wait=False)
                self.assertEqual([job.text for job in runner.saved()], ["active prompt"])
            finally:
                release.set()
                runner.stop(wait=True)

        self.assertEqual(started, ["active prompt"])

    def test_repl_stop_wait_joins_side_jobs_after_fast_shutdown(self):
        from mechferret import repl

        queue_path = Path("side-stop-queue.json")
        release = threading.Event()
        finished = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            self.assertTrue(release.wait(timeout=2))
            time.sleep(0.05)
            finished.set()
            return text

        with redirect_stdout(StringIO()):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=queue_path)
            try:
                side = runner.submit_side(repl._btw_prompt("side shutdown"))
                runner.stop(wait=False)
                release.set()
                runner.stop(wait=True)
            finally:
                release.set()
                runner.stop(wait=True)

        self.assertTrue(finished.is_set())
        self.assertEqual(side.status, "done")
        self.assertEqual(runner.saved(), [])

    def test_repl_busy_guard_blocks_agent_state_mutations(self):
        from mechferret import repl

        queue_path = Path("busy-guard-queue.json")
        release = threading.Event()

        def fake_chat(agent, session, text, *, background=False):
            self.assertTrue(release.wait(timeout=2))
            return text

        out = StringIO()
        with redirect_stdout(out):
            runner = repl.ChatJobRunner(object(), repl.Session(), chat_fn=fake_chat, queue_path=queue_path)
            try:
                runner.submit("active prompt")
                deadline = time.monotonic() + 2
                while runner.active() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(repl._guard_agent_idle(runner, "/model"))
                release.set()
                self.assertTrue(runner.wait_idle(timeout=2))
                self.assertTrue(repl._guard_agent_idle(runner, "/model"))
            finally:
                release.set()
                runner.stop(wait=True)

        rendered = out.getvalue()
        self.assertIn("/model waits for the active prompt", rendered)
        self.assertIn("use /btw", rendered)
        self.assertIn("/cancel <id|latest|next|all>", rendered)
        self.assertIn("running #1", rendered)

    def test_repl_btw_parsing_preserves_prompt_text(self):
        from mechferret import repl

        self.assertEqual(repl._line_after_command("/btw explain this -- with flags"), "explain this -- with flags")
        prompt = repl._btw_prompt("ask a clarifying question")
        self.assertIn("compact aside", prompt)
        self.assertTrue(prompt.endswith("ask a clarifying question"))

    def test_cli_command_index_primary_names_route_from_repl(self):
        from mechferret import commands
        from mechferret.cli import _command_index_payload, build_parser

        payload = _command_index_payload(build_parser())
        command_names = {row["name"] for row in payload["commands"]}
        # `repl` intentionally launches the interactive prompt and should not
        # recurse when typed inside an existing prompt.
        command_names.discard("repl")
        self.assertTrue(command_names <= commands.COMMAND_WORDS)

    def test_repl_shortcuts_do_not_shadow_argument_bearing_cli_commands(self):
        from mechferret import repl

        cli_fallbacks = [
            ("open", ["open", "bundle"]),
            ("demo", ["demo", "--out", "runs/custom"]),
            ("init", ["init", "--project-root", "subdir"]),
            ("status", ["status", "--json"]),
            ("status", ["status", "--project-root", "project"]),
            ("audit", ["audit", "--json"]),
            ("audit", ["audit", "--strict"]),
            ("paper", ["paper", "--help"]),
            ("quickstart", ["quickstart", "--mode", "ci"]),
            ("quickstart", ["quickstart", "--run"]),
            ("review-paper", ["review-paper", "--json"]),
            ("review-paper", ["review-paper", "--provider", "openai"]),
            ("review-paper", ["review-paper", "--help"]),
            ("verify", ["verify", "--json"]),
        ]

        self.assertTrue(repl._uses_repl_shortcut("open", ["open"]))
        self.assertFalse(repl._uses_repl_shortcut("open", ["open", "bundle"]))
        self.assertTrue(repl._uses_repl_shortcut("demo", ["demo"]))
        self.assertFalse(repl._uses_repl_shortcut("demo", ["demo", "--out", "runs/custom"]))
        self.assertTrue(repl._uses_repl_shortcut("init", ["init"]))
        self.assertFalse(repl._uses_repl_shortcut("init", ["init", "--project-root", "subdir"]))
        self.assertTrue(repl._uses_repl_shortcut("status", ["status", "--select", "best"]))
        self.assertFalse(repl._uses_repl_shortcut("status", ["status", "--json"]))
        self.assertFalse(repl._uses_repl_shortcut("status", ["status", "--project-root", "project"]))
        self.assertTrue(repl._uses_repl_shortcut("audit", ["audit", "--select", "best"]))
        self.assertFalse(repl._uses_repl_shortcut("audit", ["audit", "--json"]))
        self.assertFalse(repl._uses_repl_shortcut("audit", ["audit", "--strict"]))
        self.assertTrue(repl._uses_repl_shortcut("paper", ["paper", "--select", "best"]))
        self.assertFalse(repl._uses_repl_shortcut("paper", ["paper", "--help"]))
        self.assertTrue(repl._uses_repl_shortcut("quickstart", ["quickstart"]))
        self.assertTrue(repl._uses_repl_shortcut("quickstart", ["quickstart", "ci"]))
        self.assertFalse(repl._uses_repl_shortcut("quickstart", ["quickstart", "--mode", "ci"]))
        self.assertFalse(repl._uses_repl_shortcut("quickstart", ["quickstart", "--run"]))
        self.assertTrue(repl._uses_repl_shortcut("review-paper", ["review-paper", "--select", "best"]))
        for bare, tokens in cli_fallbacks:
            self.assertFalse(repl._uses_repl_shortcut(bare, tokens))
            self.assertIn(bare, repl.KNOWN_COMMANDS)

    def test_system_prompt_routes_research_and_discovery_tools(self):
        from mechferret import agent

        prompt = agent.BASE_SYSTEM_PROMPT
        self.assertIn("run_research", prompt)
        self.assertIn("run_discovery", prompt)
        self.assertIn("Use run_research for general literature/source-grounded research", prompt)
        self.assertIn("Use run_discovery only", prompt)
        self.assertIn("audit advisories", prompt)
        self.assertIn("ask one targeted clarifying question", prompt)
        self.assertNotIn("press " + "enter", prompt.lower())


if __name__ == "__main__":
    unittest.main()
