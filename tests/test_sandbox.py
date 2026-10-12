def test_sandbox_timeout():
    from agent.tools.sandbox import run_python

    r = run_python("while True: pass", timeout_s=1)
    assert "timeout" in r["error"]
