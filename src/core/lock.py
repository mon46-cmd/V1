"""Cross-platform single-process lock.

Used by ``scripts/run_exec.py`` to refuse to start when another exec
loop is already running. Without it, two systemd restarts that race
each other can each fire ``ai.chat_watchlist`` (real USD) before the
file-based watchlist reuse guard has had time to write.

The lock is advisory (not enforced by the kernel beyond the lock
holder cooperating) and falls back to a stale-PID check on Windows
where ``fcntl`` is unavailable.
"""
from __future__ import annotations

import errno
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

log = logging.getLogger(__name__)


class LockBusy(RuntimeError):
    """Raised when another process already holds the lock."""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
            )
            if h == 0:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        os.kill(pid, 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM
    except Exception:  # noqa: BLE001
        return False


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Acquire an exclusive process lock at ``path``.

    Raises :class:`LockBusy` if another live process holds it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh: IO | None = None

    if sys.platform != "win32":
        # POSIX: use fcntl.flock on a long-lived file descriptor.
        import fcntl
        fh = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            fh.close()
            raise LockBusy(
                f"another process already holds {path} (errno={e.errno})"
            ) from e
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(str(os.getpid()))
            fh.flush()
            try:
                yield
            finally:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            fh.close()
            try:
                path.unlink()
            except OSError:
                pass
        return

    # Windows: stale-PID check + best-effort exclusive create.
    if path.exists():
        try:
            other = int(path.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            other = 0
        # Reject when the lock is held by ANY live process, including
        # ourselves (re-entrant acquisition is a bug).
        if other and _pid_alive(other):
            raise LockBusy(f"another process (pid={other}) holds {path}")
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_TRUNC | os.O_WRONLY)
    except OSError as e:
        raise LockBusy(f"cannot acquire {path}: {e}") from e
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
    finally:
        os.close(fd)
    try:
        yield
    finally:
        try:
            path.unlink()
        except OSError:
            pass
