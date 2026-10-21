from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from evals.dataset import GoldenProblem
from evals.runner import run_batch


def _problem(problem_id: str) -> GoldenProblem:
    return GoldenProblem(
        id=problem_id,
        statement=f"Implement solve_{problem_id}(x)",
        constraints="x is an integer",
        complexity="O(1)",
        function_name=f"solve_{problem_id}",
        cases={"base_case": {}, "edge_case": {}, "adversarial_case": {}},
    )


def test_runner_persists_manifest_and_repeat_index(tmp_path, monkeypatch):
    from evals import runner

    monkeypatch.setattr(runner, "load_dataset", lambda path: [_problem("1"), _problem("2")])
    monkeypatch.setattr(runner, "dataset_hash", lambda path: "digest")
    monkeypatch.setattr(runner, "_git_sha", lambda: "abc123")
    monkeypatch.setattr(
        runner,
        "get_config",
        lambda: SimpleNamespace(
            models={"heavy": SimpleNamespace(name="model-a")},
            gateway=SimpleNamespace(experiment="leetcodeagent"),
            evals=SimpleNamespace(
                dataset_name="leetcode-golden-v1",
                experiment="leetcodeagent-evals",
                baseline_run_id="",
            ),
        ),
    )

    def run(command, *, cwd, env, check, capture_output, text):
        assert "-m" in command and "cli" in command
        problems_file = Path(command[command.index("--problems-file") + 1])
        output_root = Path(command[command.index("--out-dir") + 1])
        records = [json.loads(line) for line in problems_file.read_text().splitlines()]
        output_dir = output_root / "batch_fake"
        output_dir.mkdir(parents=True)
        for record in records:
            (output_dir / f"{record['id']}.json").write_text(
                json.dumps({"run_id": record["id"], "status": "ok"})
            )
        (output_dir / "summary.json").write_text("{}")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(runner.subprocess, "run", run)
    batch = run_batch(
        dataset_path="unused.csv",
        subset="1,2",
        repeat=2,
        results_root=tmp_path / "results",
        no_mlflow=True,
    )

    manifest = json.loads((batch / "manifest.json").read_text())
    index = json.loads((batch / "run_index.json").read_text())
    assert manifest["dataset_hash"] == "digest"
    assert manifest["git_sha"] == "abc123"
    assert manifest["repeat"] == 2
    assert manifest["subset"] == "1,2"
    assert set(index) == {"1", "2"}
    assert len(index["1"]) == 2
    assert [item["status"] for item in index["1"]] == ["ok", "ok"]
    assert all(item["trace_id"] is None for item in index["1"])
