def test_ci_gate_check_deliberately_fails():
    assert False, "deliberate failure to check the merge gate; do not merge"
