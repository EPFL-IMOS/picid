"""Tests for publish scripts."""
import subprocess
import sys
import zipfile
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


class TestPushRunToGdrive:
    def test_raises_on_nonexistent_run_dir(self, tmp_path):
        from scripts.publish.push_run_to_gdrive import push_run_to_gdrive

        with pytest.raises(ValueError, match="does not exist"):
            push_run_to_gdrive(tmp_path / "nonexistent")

    def test_dry_run_returns_link_with_run_name(self, tmp_path):
        from scripts.publish.push_run_to_gdrive import push_run_to_gdrive

        run_dir = tmp_path / "my_run_2026-03-07"
        run_dir.mkdir()
        (run_dir / "REPRODUCE.md").write_text("# Reproduce")
        (run_dir / "config_resolved.yaml").write_text("experiment: test\n")

        link = push_run_to_gdrive(run_dir, dry_run=True)
        assert "drive.google.com" in link
        assert "my_run_2026-03-07" in link

    def test_dry_run_zips_key_files(self, tmp_path):
        from scripts.publish.push_run_to_gdrive import zip_run_dir

        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        (run_dir / "REPRODUCE.md").write_text("# Reproduce")
        (run_dir / "config_resolved.yaml").write_text("experiment: test\n")
        (run_dir / "run_metadata.yaml").write_text("git_commit: abc\n")
        (run_dir / "should_be_excluded.bin").write_bytes(b"\x00" * 100)

        out_zip = tmp_path / "out.zip"
        zip_run_dir(run_dir, out_zip)

        assert out_zip.exists()
        with zipfile.ZipFile(out_zip) as zf:
            names = zf.namelist()
        assert "REPRODUCE.md" in names
        assert "config_resolved.yaml" in names
        assert "run_metadata.yaml" in names
        assert "should_be_excluded.bin" not in names

    def test_cli_dry_run_prints_link(self, tmp_path):
        run_dir = tmp_path / "my_run_cli"
        run_dir.mkdir()
        (run_dir / "REPRODUCE.md").write_text("# Reproduce")

        r = subprocess.run(
            [sys.executable, "scripts/publish/push_run_to_gdrive.py", str(run_dir), "--dry-run"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert r.returncode == 0, r.stderr
        assert "drive.google.com" in r.stdout

    def test_cli_dry_run_print_message(self, tmp_path):
        run_dir = tmp_path / "my_run_msg"
        run_dir.mkdir()
        (run_dir / "REPRODUCE.md").write_text("# Reproduce\n\nRepro content")
        (run_dir / "run_metadata.yaml").write_text("git_commit: deadbeef\n")

        r = subprocess.run(
            [
                sys.executable,
                "scripts/publish/push_run_to_gdrive.py",
                str(run_dir),
                "--dry-run",
                "--print-message",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert r.returncode == 0, r.stderr
        assert "Reproduce" in r.stdout
        assert "deadbeef" in r.stdout
        assert "drive.google.com" in r.stdout
