from science_agent.tools.sandbox import parse_result, run_python


def test_runs_scientific_stack_and_parses_result():
    code = (
        "import json, numpy as np, sympy as sp\n"
        "x = sp.symbols('x')\n"
        "val = float(sp.integrate(x**2, (x, 0, 3)))\n"
        "print('RESULT_JSON: ' + json.dumps({'value': val, 'mean': float(np.mean([1, 2, 3]))}))\n"
    )
    r = run_python(code, timeout=60)
    assert r.ok, r.stderr
    assert r.result == {"value": 9.0, "mean": 2.0}


def test_failure_reports_stderr():
    r = run_python("raise ValueError('boom')", timeout=30)
    assert not r.ok
    assert "boom" in r.summary()


def test_timeout():
    r = run_python("while True: pass", timeout=1)
    assert r.timed_out and not r.ok


def test_parse_result_uses_last_marker_and_wraps_scalars():
    assert parse_result("RESULT_JSON: {\"a\": 1}\nRESULT_JSON: 5") == {"value": 5}
    assert parse_result("nothing") is None


def test_non_ascii_output_round_trips():
    # Models like to print "≈", "µ" or "→"; on Windows the default cp1252 stdout crashed on these.
    code = "import json\nprint('C ≈ 9967 bit/s, µ → ∞')\nprint('RESULT_JSON: ' + json.dumps({'note': '≈'}, ensure_ascii=False))\n"
    r = run_python(code, timeout=30)
    assert r.ok, r.stderr
    assert "C ≈ 9967 bit/s, µ → ∞" in r.stdout
    assert r.result == {"note": "≈"}


def test_sandbox_runner_round_trip():
    import threading
    from http.server import HTTPServer

    from science_agent.tools import sandbox_server
    from science_agent.tools.sandbox import RemoteSandbox

    srv = HTTPServer(("127.0.0.1", 0), sandbox_server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        remote = RemoteSandbox(f"http://127.0.0.1:{srv.server_address[1]}")
        r = remote("import json\nprint('RESULT_JSON: ' + json.dumps({'v': 2 + 2}))", timeout=30)
        assert r.ok and r.result == {"v": 4}
        r = remote("raise SystemExit('nope')", timeout=30)
        assert not r.ok and "nope" in r.stderr
    finally:
        srv.shutdown()
        srv.server_close()


def test_sandbox_runner_over_unix_socket(tmp_path):
    import os
    import threading

    if os.name != "posix":
        return
    from science_agent.tools import sandbox_server
    from science_agent.tools.sandbox import RemoteSandbox

    path = str(tmp_path / "sandbox.sock")
    srv, _ = sandbox_server.make_server("", 0, path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert os.stat(path).st_mode & 0o777 == 0o666
        r = RemoteSandbox(f"unix://{path}")("import json\nprint('RESULT_JSON: ' + json.dumps({'v': 6 * 7}))", timeout=30)
        assert r.ok and r.result == {"v": 42}
    finally:
        srv.shutdown()
        srv.server_close()
    assert not os.path.exists(path)
    r = RemoteSandbox(f"unix://{path}")("print(1)", timeout=1)
    assert not r.ok and "unreachable" in r.summary()


def test_unreachable_runner_reports_a_failed_script():
    from science_agent.tools.sandbox import RemoteSandbox

    r = RemoteSandbox("http://127.0.0.1:9")("print(1)", timeout=1)
    assert not r.ok and "unreachable" in r.summary()


def test_background_processes_do_not_outlive_the_script(tmp_path):
    import os
    import sys

    if os.name != "posix":
        return
    marker = tmp_path / "survivor"
    child = f"import time, pathlib; time.sleep(1.5); pathlib.Path({str(marker)!r}).write_text('x')"
    code = f"import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', {child!r}])\nprint('started')\n"
    assert run_python(code, timeout=30).ok
    import time
    time.sleep(2.5)
    assert not marker.exists()
