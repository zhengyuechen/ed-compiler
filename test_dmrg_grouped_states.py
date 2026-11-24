"""Check the orthogonality penalty with both grouped products on a 10-spin cylinder."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def test_grouped_states():
    root = Path(__file__).resolve().parent
    check = root / "bench" / "dmrg_gpu" / "check_excited.py"
    env = os.environ.copy()
    paths = [str(check.parent / "fakecupy"), str(root)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    with tempfile.TemporaryDirectory() as tmp:
        for engine in ("gpu3g", "gpu3gr"):
            result = Path(tmp) / f"{engine}.json"
            subprocess.run([sys.executable, str(check), "2", "2", "0", "5", "0",
                            "64", "2", "4", engine, str(result)], env=env, check=True)
            data = json.loads(result.read_text())
            # The second energy is degenerate. Compare energies and orthogonality.
            assert data["exact"][1] - data["exact"][0] > 0.1
            for state, reference in zip(data["states"], data["exact"]):
                assert abs(state["E"] - reference) < 1e-8
                assert abs(state["E_engine"] - reference) < 1e-8
            assert data["states"][1]["overlaps"][0] < 1e-5


if __name__ == "__main__":
    test_grouped_states()
