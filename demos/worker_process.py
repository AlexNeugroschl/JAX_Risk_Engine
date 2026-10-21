"""Windows job ownership for the Python benchmark workers and their descendants."""


class WindowsJob:
    """Create before suspended Popen, attach then resume, close after the worker ends.

    Nested jobs include descendants even when they create process groups of their own:
    https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
    No process is opened or terminated by a caller-supplied PID.
    """
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w

        class Basic(c.Structure):
            _fields_ = [("process_time", c.c_longlong), ("job_time", c.c_longlong),
                        ("flags", w.DWORD), ("min_working_set", c.c_size_t),
                        ("max_working_set", c.c_size_t), ("active_processes", w.DWORD),
                        ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class Extended(c.Structure):
            _fields_ = [("basic", Basic), ("io", c.c_ulonglong * 6),
                        ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                        ("peak_process_memory", c.c_size_t), ("peak_job_memory", c.c_size_t)]

        class ThreadEntry(c.Structure):
            _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("id", w.DWORD),
                        ("owner", w.DWORD), ("priority", w.LONG), ("delta", w.LONG), ("flags", w.DWORD)]

        self.api = c.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes, self.api.CreateJobObjectW.restype = (w.LPVOID, w.LPCWSTR), w.HANDLE
        self.api.SetInformationJobObject.argtypes = (w.HANDLE, c.c_int, w.LPVOID, w.DWORD)
        self.api.SetInformationJobObject.restype = w.BOOL
        self.api.AssignProcessToJobObject.argtypes, self.api.AssignProcessToJobObject.restype = (w.HANDLE, w.HANDLE), w.BOOL
        self.api.CloseHandle.argtypes, self.api.CloseHandle.restype = (w.HANDLE,), w.BOOL
        self.api.CreateToolhelp32Snapshot.argtypes, self.api.CreateToolhelp32Snapshot.restype = (w.DWORD, w.DWORD), w.HANDLE
        for name in ("Thread32First", "Thread32Next"):
            function = getattr(self.api, name)
            function.argtypes, function.restype = (w.HANDLE, c.POINTER(ThreadEntry)), w.BOOL
        self.api.OpenThread.argtypes, self.api.OpenThread.restype = (w.DWORD, w.BOOL, w.DWORD), w.HANDLE
        self.api.ResumeThread.argtypes, self.api.ResumeThread.restype = (w.HANDLE,), w.DWORD
        self.thread_entry_type = ThreadEntry
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error())
            self.close()
            raise error

    def attach(self, process):
        import ctypes as c
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise c.WinError(c.get_last_error())

    def resume(self, process):
        """Resume only the primary thread of our still-suspended worker.

        Popen closes the initial thread handle. The documented thread snapshot API recovers
        its ID without letting a venv launcher spawn a child before assignment to the job.
        """
        import ctypes as c
        snapshot = self.api.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())
        entry = self.thread_entry_type()
        entry.size = c.sizeof(entry)
        try:
            available = self.api.Thread32First(snapshot, c.byref(entry))
            while available:
                if entry.owner == process.pid:
                    thread = self.api.OpenThread(0x2, False, entry.id)  # THREAD_SUSPEND_RESUME
                    if not thread:
                        raise c.WinError(c.get_last_error())
                    try:
                        if self.api.ResumeThread(thread) != 1:
                            raise OSError("worker primary thread did not have exactly one suspension")
                    finally:
                        self.api.CloseHandle(thread)
                    return
                available = self.api.Thread32Next(snapshot, c.byref(entry))
            raise ProcessLookupError("suspended worker has no primary thread")
        finally:
            self.api.CloseHandle(snapshot)

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            self.api.CloseHandle(handle)
