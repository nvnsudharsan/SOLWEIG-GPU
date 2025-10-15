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
    assert code == 0
    assert 'thermal_comfort' in out or 'usage' in out.lower()


@pytest.mark.integration
def test_cli_missing_required_args(tmp_path):
    # Calling without required args should fail with non-zero exit and show message
    code, out, err = _run_cmd([sys.executable, '-m', 'solweig_gpu.cli'])
    assert code != 0
    # Look for a hint of argparse error or our message
    combined = out + err
    assert 'required' in combined.lower() or 'usage:' in combined.lower()


