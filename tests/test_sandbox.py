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
