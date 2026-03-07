#!/usr/bin/env python3
"""Push a run dir to Google Drive and return a shareable link.

Usage::

    uv run python scripts/publish/push_run_to_gdrive.py <run_dir> [--dry-run] [--print-message] [--config CONFIG]

Options:
    --dry-run        Zip files and return a fake link without uploading. Use this
                     until real Drive credentials are configured.
    --print-message  After upload/dry-run, print the full submission message
                     (calls build_submission_message with the returned link).
    --config PATH    Path to gdrive_config.yaml (default: scripts/publish/gdrive_config.yaml).

Real upload:
    1. Copy ``scripts/publish/gdrive_config.example.yaml`` to ``scripts/publish/gdrive_config.yaml``.
    2. Fill in ``gdrive_folder_id`` and ``credentials_path``.
    3. Run without ``--dry-run``.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Files included in the submission zip (everything else is excluded).
_ZIP_INCLUDE = frozenset(
    {"REPRODUCE.md", "config_resolved.yaml", "run_metadata.yaml", "uv.lock", "debug.log"}
)


def zip_run_dir(run_dir: Path, out_zip: Path) -> None:
    """Zip key files from a run dir into out_zip."""
    run_dir = Path(run_dir).resolve()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(run_dir.iterdir()):
            if f.is_file() and f.name in _ZIP_INCLUDE:
                zf.write(f, f.name)


def _upload_to_gdrive(zip_path: Path, folder_id: str, credentials_path: str) -> str:
    """Upload zip to Google Drive. Returns shareable link.

    TODO: implement with PyDrive2 or google-api-python-client once credentials
    are available. Example with PyDrive2::

        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile(credentials_path)
        drive = GoogleDrive(gauth)
        f = drive.CreateFile({"title": zip_path.name, "parents": [{"id": folder_id}]})
        f.SetContentFile(str(zip_path))
        f.Upload()
        f.InsertPermission({"type": "anyone", "value": "anyone", "role": "reader"})
        return f["alternateLink"]
    """
    raise NotImplementedError(
        "Real GDrive upload is not yet implemented. Run with --dry-run to test the workflow."
    )


def push_run_to_gdrive(
    run_dir: Path,
    config_path: Path | None = None,
    dry_run: bool = False,
) -> str:
    """Zip a run dir and push it to Google Drive (or dry-run).

    Parameters
    ----------
    run_dir : Path
        The run output directory to upload.
    config_path : Path | None
        Path to ``gdrive_config.yaml``. Defaults to ``scripts/publish/gdrive_config.yaml``.
    dry_run : bool
        If True, skip actual upload and return a fake link.

    Returns
    -------
    str
        Shareable Google Drive link (real or fake in dry-run mode).
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise ValueError(f"Run dir does not exist: {run_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / f"{run_dir.name}.zip"
        zip_run_dir(run_dir, zip_path)

        if dry_run:
            return f"https://drive.google.com/drive/folders/DRY_RUN_{run_dir.name}"

        cfg_path = config_path or (Path(__file__).parent / "gdrive_config.yaml")
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"GDrive config not found: {cfg_path}\n"
                "Copy scripts/publish/gdrive_config.example.yaml to gdrive_config.yaml and fill it in."
            )

        import yaml  # deferred so script is importable without PyYAML at module level

        cfg = yaml.safe_load(cfg_path.read_text())
        return _upload_to_gdrive(zip_path, cfg["gdrive_folder_id"], cfg.get("credentials_path", ""))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Push a run dir to Google Drive and optionally print a submission message.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=Path, help="Run output directory to upload")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Zip files but skip real upload; returns a fake Drive link",
    )
    p.add_argument(
        "--print-message",
        action="store_true",
        help="Print full submission message after upload",
    )
    p.add_argument("--config", type=Path, default=None, help="Path to gdrive_config.yaml")
    args = p.parse_args()

    link = push_run_to_gdrive(args.run_dir, config_path=args.config, dry_run=args.dry_run)
    print(f"Drive link: {link}")

    if args.print_message:
        # Import and call build_submission_message directly
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.publish.build_submission_message import build_submission_message

        msg = build_submission_message(args.run_dir, gdrive_link=link)
        if msg:
            print("\n" + "=" * 60 + "\n## Submission Message\n" + "=" * 60)
            print(msg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
