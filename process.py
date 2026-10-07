"""Own native processes until exit, including interruption and parent shutdown."""

import atexit
import ctypes
from ctypes import wintypes
import os
import signal
import subprocess
import tempfile
import time


if os.name == "nt":
    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", ctypes.c_ulonglong * 6),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel.SetInformationJobObject.restype = wintypes.BOOL
    _kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel.CloseHandle.restype = wintypes.BOOL


class ProcessOwner:
    def __init__(self):
        self.process = None
        self.job = None
        if os.name == "nt":
            self.job = _kernel.CreateJobObjectW(None, None)
            if not self.job:
                raise ctypes.WinError(ctypes.get_last_error())
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not _kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                error = ctypes.WinError(ctypes.get_last_error())
                self.close()
                raise error

    def start(self, command, stdout, stderr, cwd):
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        self.process = subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            close_fds=True, **kwargs,
        )
        if self.job and not _kernel.AssignProcessToJobObject(self.job, int(self.process._handle)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def close(self):
        if self.job:
            _kernel.CloseHandle(self.job)
            self.job = None
        process = self.process
        if process is None:
            return
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        finally:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.process = None


def run_process(command, *, cwd, timeout, check_cancel, temp_dir=None):
    """Return stdout and a bounded diagnostic tail; never leave a live child."""
    check_cancel()
    with tempfile.TemporaryFile(dir=temp_dir) as output, tempfile.TemporaryFile(dir=temp_dir) as diagnostics:
        owner = ProcessOwner()
        atexit.register(owner.close)
        try:
            owner.start(command, output, diagnostics, cwd)
            deadline = time.monotonic() + timeout
            while owner.process.poll() is None:
                check_cancel()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Ninferencing: runtime exceeded {timeout:g} seconds. Increase timeout in Settings if needed.")
                time.sleep(0.05)
            check_cancel()
            code = owner.process.returncode
            output.seek(0)
            text = output.read().decode("utf-8", errors="replace")
            diagnostics.seek(0, os.SEEK_END)
            diagnostics.seek(max(0, diagnostics.tell() - 12288))
            tail = diagnostics.read().decode("utf-8", errors="replace").strip()
            return code, text, tail
        finally:
            try:
                owner.close()
            finally:
                atexit.unregister(owner.close)
