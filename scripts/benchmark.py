"""Measure fresh-process first render and same-process repeats on the same local input."""

import argparse
import ctypes
import json
import os
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path


def peak_bytes():
    if os.name != "nt":
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage if platform.system() == "Darwin" else usage * 1024
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            *(
                (key, ctypes.c_size_t)
                for key in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ),
        ]

    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    proc = ctypes.windll.kernel32.GetCurrentProcess
    proc.restype = wintypes.HANDLE
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    if not fn(proc(), ctypes.byref(counter), counter.cb):
        raise ctypes.WinError()
    return counter.PeakWorkingSetSize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("examples/phase2/mixed/job.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("use at least two renders to measure the warm process")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        os.environ["HWSOL_DATA_DIR"] = str(root / "data")
        os.environ["MPLCONFIGDIR"] = str(root / "mpl")
        runs = []
        for index in range(args.repeat):
            start, cpu = time.perf_counter(), time.process_time()
            from hwsol.jobs.runtime import render_job

            result = render_job(args.manifest, root / str(index))
            runs.append(
                {
                    "cache": "warm" if index else "fresh",
                    "pages": result["page_count"],
                    "wall_s": round(time.perf_counter() - start, 4),
                    "cpu_s": round(time.process_time() - cpu, 4),
                    "peak_process_bytes": peak_bytes(),
                }
            )
    report = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "package": version("handwritten-solutions"),
        "fixture": args.manifest.name,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
