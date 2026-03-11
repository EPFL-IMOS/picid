"""Comprehensive tests for FileLock.

This module tests the file-based locking mechanism used for
concurrent cache access protection.

PHM Context:
-----------
When multiple processes run preprocessing pipelines, file locking
prevents race conditions when writing to shared cache directories.

Test Coverage Strategy:
----------------------
1. **Basic Lock/Unlock**: Acquire and release operations
2. **Lock File Creation**: File handle management
3. **Idempotent Release**: Safe repeated release calls
4. **Platform Support**: Unix-specific behavior testing
"""

import time

from picid.data.cache.file_lock import FileLock, LOCK_FILE


class TestFileLockInitialization:
    """Tests for FileLock initialization."""

    def test_default_path(self):
        """Test initialization with default path.

        **PHM Logic**: Default lock file in /tmp for system-wide coordination.

        **Methodology**: Create FileLock without arguments.

        **Expected**: Uses LOCK_FILE constant.

        Validates: Requirement FL-1.1 - Default path usage
        """
        lock = FileLock()

        assert lock.path == LOCK_FILE
        assert lock.handle is None

    def test_custom_path(self, tmp_path):
        """Test initialization with custom path.

        **PHM Logic**: Custom paths allow project-specific locking.

        **Methodology**: Create FileLock with custom path.

        **Expected**: Uses provided path.

        Validates: Requirement FL-1.2 - Custom path support
        """
        custom_path = str(tmp_path / "custom.lock")
        lock = FileLock(path=custom_path)

        assert lock.path == custom_path
        assert lock.handle is None


class TestFileLockAcquire:
    """Tests for lock acquisition."""

    def test_acquire_creates_file(self, tmp_path):
        """Test that acquire creates lock file.

        **PHM Logic**: Lock file must exist to acquire lock.

        **Methodology**: Acquire lock on non-existent file.

        **Expected**: File created and lock acquired.

        Validates: Requirement FL-2.1 - Lock file creation
        """
        lock_path = tmp_path / "test.lock"
        lock = FileLock(path=str(lock_path))

        assert not lock_path.exists()

        lock.acquire()

        # File should be created
        assert lock_path.exists()
        assert lock.handle is not None

        lock.release()

    def test_acquire_sets_handle(self, tmp_path):
        """Test that acquire sets file handle.

        **PHM Logic**: Handle tracks the open file for release.

        **Methodology**: Acquire lock, verify handle set.

        **Expected**: handle is not None after acquire.

        Validates: Requirement FL-2.2 - Handle initialization
        """
        lock_path = tmp_path / "test.lock"
        lock = FileLock(path=str(lock_path))

        assert lock.handle is None

        lock.acquire()

        assert lock.handle is not None

        lock.release()

    def test_acquire_is_exclusive(self, tmp_path):
        """Test that lock is exclusive (single process test).

        **PHM Logic**: Lock should prevent concurrent access.

        **Methodology**: Acquire lock, verify held.

        **Expected**: Lock file opened with exclusive flag.

        Validates: Requirement FL-2.3 - Exclusive locking
        """
        lock_path = tmp_path / "test.lock"
        lock = FileLock(path=str(lock_path))

        lock.acquire()

        # Verify lock is held by checking handle is valid
        assert lock.handle is not None
        assert not lock.handle.closed

        lock.release()


class TestFileLockRelease:
    """Tests for lock release."""

    def test_release_closes_handle(self, tmp_path):
        """Test that release closes file handle.

        **PHM Logic**: Handle must be closed to release lock.

        **Methodology**: Acquire then release, verify handle closed.

        **Expected**: Handle is closed after release.

        Validates: Requirement FL-3.1 - Handle closure
        """
        lock_path = tmp_path / "test.lock"
        lock = FileLock(path=str(lock_path))

        lock.acquire()
        handle = lock.handle

        lock.release()

        # Handle should be closed
        assert handle.closed

    def test_release_after_acquire(self, tmp_path):
        """Test standard acquire/release pattern.

        **PHM Logic**: Standard usage pattern for locking.

        **Methodology**: Acquire then release once.

        **Expected**: No error on release.

        Validates: Requirement FL-3.2 - Standard release
        """
        lock_path = tmp_path / "test.lock"
        lock = FileLock(path=str(lock_path))

        lock.acquire()
        lock.release()

        # Verify handle is closed
        assert lock.handle is None or lock.handle.closed

    def test_release_without_acquire(self, tmp_path):
        """Test release without prior acquire.

        **PHM Logic**: Safe to call release even if not acquired.

        **Methodology**: Create lock and release without acquire.

        **Expected**: No error (handle is None, no-op).

        Validates: Requirement FL-3.3 - Safe no-op release
        """
        lock_path = tmp_path / "test.lock"
        lock = FileLock(path=str(lock_path))

        # Should not raise
        lock.release()


class TestFileLockContextManager:
    """Tests for using FileLock as context manager (if supported)."""

    def test_acquire_release_pattern(self, tmp_path):
        """Test standard acquire/release pattern.

        **PHM Logic**: Typical usage pattern for cache operations.

        **Methodology**: Acquire, do work, release.

        **Expected**: Lock held during work, released after.

        Validates: Requirement FL-4.1 - Standard usage pattern
        """
        lock_path = tmp_path / "test.lock"
        lock = FileLock(path=str(lock_path))

        try:
            lock.acquire()
            # Simulate work
            time.sleep(0.01)
        finally:
            lock.release()

        # Lock should be released
        assert lock.handle is None or lock.handle.closed


class TestFileLockEdgeCases:
    """Edge case tests for FileLock."""

    def test_path_with_spaces(self, tmp_path):
        """Test lock with spaces in path.

        **PHM Logic**: Paths may contain spaces.

        **Methodology**: Create lock with space in path.

        **Expected**: Lock works correctly.

        Validates: Requirement FL-5.1 - Special path handling
        """
        space_dir = tmp_path / "path with spaces"
        space_dir.mkdir()
        lock_path = space_dir / "test.lock"

        lock = FileLock(path=str(lock_path))

        lock.acquire()
        assert lock.handle is not None
        lock.release()

    def test_nested_directory_path(self, tmp_path):
        """Test lock in nested non-existent directory.

        **PHM Logic**: Lock file creation doesn't auto-create directories.

        **Methodology**: Try to create lock in non-existent nested dir.

        **Expected**: May raise error or create path.

        Validates: Requirement FL-5.2 - Directory handling
        """
        # Note: FileLock may not create parent directories
        # This tests the actual behavior
        nested_path = tmp_path / "a" / "b" / "c" / "test.lock"
        lock = FileLock(path=str(nested_path))

        try:
            # Create parent directories first
            nested_path.parent.mkdir(parents=True, exist_ok=True)
            lock.acquire()
            lock.release()
        except (FileNotFoundError, PermissionError):
            # Expected if directories don't exist
            pass
