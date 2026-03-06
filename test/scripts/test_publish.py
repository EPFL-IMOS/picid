"""Tests for publish scripts."""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestBuildSubmissionMessage:
    def test_contains_reproduce_info_and_drive_link(self, tmp_path):
        (tmp_path / "REPRODUCE.md").write_text("# Reproduce\n\nContent here")
        (tmp_path / "run_metadata.yaml").write_text("git_commit: abc123\ngit_branch: main\n")
        r = subprocess.run(
            [
                sys.executable,
                "scripts/publish/build_submission_message.py",
                str(tmp_path),
                "--gdrive-link",
                "https://drive.google.com/drive/folders/TEST123",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert r.returncode == 0, r.stderr
        assert "Reproduce" in r.stdout
        assert "abc123" in r.stdout
        assert "https://drive.google.com" in r.stdout

    def test_works_without_drive_link(self, tmp_path):
        (tmp_path / "REPRODUCE.md").write_text("# Reproduce\n\nContent")
        r = subprocess.run(
            [sys.executable, "scripts/publish/build_submission_message.py", str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert r.returncode == 0, r.stderr
        assert "Reproduce" in r.stdout

    def test_output_to_file(self, tmp_path):
        (tmp_path / "REPRODUCE.md").write_text("# Reproduce\n\nContent")
        out_file = tmp_path / "submission.md"
        r = subprocess.run(
            [
                sys.executable,
                "scripts/publish/build_submission_message.py",
                str(tmp_path),
                "--output",
                str(out_file),
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert r.returncode == 0, r.stderr
        assert out_file.exists()
        assert "Reproduce" in out_file.read_text()

    def test_empty_run_dir_produces_empty_output(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "scripts/publish/build_submission_message.py", str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == ""
