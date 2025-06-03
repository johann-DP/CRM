import sys
from pathlib import Path
from phase4bis import run_all_since_commit


def test_run_all_executes_scripts(monkeypatch, tmp_path):
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    scripts = [script_dir / f"s{i}.py" for i in range(3)]
    for p in scripts:
        p.write_text("print('ok')", encoding="utf-8")

    cfg_path = tmp_path / "cfg.yaml"
    outdir = tmp_path / "out"
    cfg_path.write_text(f"output_dir: {outdir}\n", encoding="utf-8")

    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        assert run_all_since_commit.OUTPUT_DIR == outdir
        return True

    monkeypatch.setattr(run_all_since_commit, "run", fake_run)
    monkeypatch.setattr(
        run_all_since_commit,
        "_scripts_in_dir",
        lambda folder: sorted(script_dir.glob("*.py")),
    )
    monkeypatch.setattr(run_all_since_commit, "_needs_config", lambda p: False)

    run_all_since_commit.main(["--config", str(cfg_path), "--jobs", "1"])

    assert outdir.exists()
    assert [Path(c[1]).name for c in calls] == [p.name for p in scripts]
