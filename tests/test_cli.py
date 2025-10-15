import subprocess
import sys
import os
import pytest


def _run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode, result.stdout, result.stderr


def test_cli_help():
    # Ensure CLI is installed and --help works
    code, out, err = _run_cmd([sys.executable, '-m', 'solweig_gpu.cli', '--help'])
    combined = out + err
    
    # Should succeed with code 0 and show help
    assert code == 0, f"CLI --help should exit with 0, got {code}"
    # Should contain help text with key arguments
    assert 'base_path' in combined or 'Base directory' in combined or 'usage' in combined.lower(), \
        f"Help output should mention base_path or usage. Got: {combined[:200]}"


@pytest.mark.integration
def test_cli_missing_required_args(tmp_path):
    # Calling without required args should fail
    code, out, err = _run_cmd([sys.executable, '-m', 'solweig_gpu.cli'])
    combined = out + err
    
    # Should fail with non-zero exit code
    assert code != 0, f"CLI without args should fail, got exit code {code}"
    # Should show error about required arguments
    assert 'required' in combined.lower() or 'base_path' in combined.lower() or 'error' in combined.lower(), \
        f"Error message should mention required args. Got: {combined[:200]}"


