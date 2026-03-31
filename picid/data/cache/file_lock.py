import sys

SYSTEM_PLATFORM = sys.platform
if SYSTEM_PLATFORM == "win32":
    import msvcrt
else:
    import fcntl

import logging

logger = logging.getLogger(__name__)
LOCK_FILE = "/tmp/picid_preprocess.lock"


class FileLock:
    """
    FileLock class for managing exclusive file-based locks.

    This class provides a simple interface for acquiring and releasing exclusive locks
    using the filesystem. It's useful for coordinating access to shared resources
    across multiple processes.

    Parameters
    ----------
    path : str, optional
        The filesystem path where the lock file will be created.

    Attributes
    ----------
    path : str
        The filesystem path where the lock file will be created.
    handle : file object
        File handle used to maintain the lock. None if not acquired.

    Examples
    --------
    >>> lock = FileLock("/tmp/my_lock.lock")
    >>> lock.acquire()
    Lock acquired.
    >>> # Perform critical operations here
    >>> lock.release()
    Lock released.
    """

    def __init__(self, path=LOCK_FILE):
        """
        Initialize a FileLock instance.

        Parameters
        ----------
        path : str, optional
            The path to the lock file. Defaults to LOCK_FILE.
        """
        self.path = path
        self.handle = None

    def acquire(self):
        """
        Acquire an exclusive lock on the file.

        Creates or opens the lock file and applies an exclusive lock using fcntl.
        Prints a confirmation message upon successful acquisition.

        Note:
            This method blocks until the lock is available if another process holds it.
        """
        self.handle = open(self.path, "w")
        if SYSTEM_PLATFORM == "win32":
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_LOCK)
        else:
            fcntl.flock(self.handle, fcntl.LOCK_EX)
        logger.info("Lock acquired.")

    def release(self):
        """
        Release the exclusive lock and remove the lock file.

        Unlocks the file handle, closes it, and deletes the lock file from the filesystem.
        Prints a confirmation message upon successful release.

        Note:
            Safe to call even if lock was never acquired (checks if handle exists).
        """
        if self.handle:
            if SYSTEM_PLATFORM == "win32":
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 100)
            else:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            logger.info("Lock released.")
