import json
import os
import tempfile
import unittest
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

    def test_parallel_readonly_and_plan_denial(self):
        from mechferret import agent

        a = agent.Agent()
        a.provider, a.model, a._key = "anthropic", "claude-opus-4-8", "x"
        res = a._run_tool_calls([("1", "list_skills", {}), ("2", "environment_status", {})])
        self.assertEqual(sorted(res), ["1", "2"])
        self.assertTrue(all(res.values()))
        a.permission_mode = "plan"
        res2 = a._run_tool_calls([("a", "write_file", {"path": "x", "content": "y"})])
        self.assertTrue(json.loads(res2["a"])["denied"])

    def test_mcp_no_server_is_safe(self):
        from mechferret import mcp, tools

        self.assertEqual(mcp.status()["configured"], [])
        self.assertEqual(mcp.tool_specs(), [])
        self.assertIn("error", tools.run_tool("mcp__x__y", {}))

    def test_command_registry_matches_handlers(self):
        from mechferret import commands

        # every REPL-handled bare word should appear in the grouped help registry
        names = " ".join(c.name for _title, cmds in commands.SECTIONS for c in cmds)
        for handled in ("login", "model", "plan", "cost", "compact", "resume", "memory", "export", "init", "goal", "why", "arch", "paper"):
            self.assertIn(handled, names)


if __name__ == "__main__":
    unittest.main()
