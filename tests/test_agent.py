import json
import os
import tempfile
import unittest
from pathlib import Path

from mechferret import agent


class AgentToolTest(unittest.TestCase):
    def test_tool_schemas_are_well_formed(self):
        from mechferret import tools

        names = {t["name"] for t in tools.TOOL_SPECS}
        self.assertEqual(names, set(tools.HANDLERS))
        # the full Claude-Code-style suite is present
        for expected in ("bash", "read_file", "write_file", "edit_file", "grep", "glob", "web_search", "web_fetch"):
            self.assertIn(expected, names)
        for tool in tools.TOOL_SPECS:
            self.assertIn("description", tool)
            self.assertEqual(tool["parameters"]["type"], "object")

    def test_run_discovery_tool_returns_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = agent._run_tool("run_discovery", {"skill": "ioi-circuit", "backend": "synthetic", "out_dir": str(Path(tmp) / "r")})
            payload = json.loads(out)
            self.assertGreaterEqual(len(payload["discoveries"]), 1)
            self.assertIn("rigor_score", payload["metrics"])

    def test_list_skills_tool(self):
        payload = json.loads(agent._run_tool("list_skills", {}))
        self.assertTrue(any(s["name"] == "ioi-circuit" for s in payload))

    def test_unknown_tool_is_reported(self):
        payload = json.loads(agent._run_tool("nope", {}))
        self.assertIn("error", payload)

    def test_permissions_and_cost(self):
        from mechferret import permissions
        from mechferret.costs import CostTracker

        self.assertEqual(permissions.decide("read_file", {}, read_only=True, permission_class="local", mode="auto").behavior, "allow")
        self.assertEqual(permissions.decide("write_file", {}, read_only=False, permission_class="write", mode="auto").behavior, "allow")
        self.assertEqual(permissions.decide("write_file", {}, read_only=False, permission_class="write", mode="plan").behavior, "ask")
        self.assertEqual(permissions.decide("bash", {"command": "rm -rf /tmp/x"}, read_only=False, permission_class="exec", mode="auto").behavior, "ask")
        c = CostTracker()
        c.add("claude-opus-4-8", {"input_tokens": 1_000_000, "output_tokens": 0})
        self.assertAlmostEqual(c.usd, 15.0, places=3)

    def test_dispatch_denies_in_plan_mode_without_confirm(self):
        a = agent.Agent()
        a.provider, a.model, a._key = "anthropic", "claude-opus-4-8", "x"
        a.permission_mode = "plan"
        out = json.loads(a._dispatch("write_file", {"path": "/tmp/should_not_exist_zzz", "content": "x"}))
        self.assertTrue(out["denied"])
        self.assertIn("write_file", a.denials)
        self.assertFalse(Path("/tmp/should_not_exist_zzz").exists())

    def test_large_output_persisted(self):
        from mechferret import tools

        big = "x = 1\n" * 5000
        out = tools._persist_if_large("bash", big)
        self.assertIn("saved to", out)
        self.assertLess(len(out), len(big))

    def test_anthropic_tool_loop_executes_tool_then_replies(self):
        calls = {"n": 0}

        def fake_post(url, payload, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                self.assertIn("tools", payload)
                return {
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "text", "text": "Let me list them."},
                        {"type": "tool_use", "id": "t1", "name": "list_skills", "input": {}},
                    ],
                }
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Here are the skills."}]}

        original = agent._http_post
        agent._http_post = fake_post
        fired = []
        try:
            a = agent.Agent(on_tool=lambda name, args: fired.append(name))
            a.provider, a.model, a._key = "anthropic", "claude-sonnet-4-6", "fake"
            reply = a.send("what skills exist?")
        finally:
            agent._http_post = original
        self.assertEqual(calls["n"], 2)
        self.assertEqual(fired, ["list_skills"])
        self.assertIn("Here are the skills", reply)

    def test_active_provider_empty_without_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.json"
            old = os.environ.get("MECHFERRET_CONFIG")
            old_keys = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
            os.environ["MECHFERRET_CONFIG"] = str(cfg)
            try:
                provider, model, key = agent.active_provider()
                self.assertEqual(provider, "")
                self.assertFalse(agent.is_configured())
            finally:
                if old is None:
                    os.environ.pop("MECHFERRET_CONFIG", None)
                else:
                    os.environ["MECHFERRET_CONFIG"] = old
                for k, v in old_keys.items():
                    if v is not None:
                        os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
