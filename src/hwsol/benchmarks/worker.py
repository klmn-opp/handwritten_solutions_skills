"""One case per fresh process so peak RSS is attributable to that case."""

import platform
import resource
import sys
import time
import traceback
from pathlib import Path

from hwsol.jobs.runtime import render_job, verify_job
from hwsol.profiles.io import atomic_json


def main():
    manifest, output, result = map(Path, sys.argv[1:4])
    start = time.perf_counter()
    try:
        value = render_job(manifest, output)
        verify_job(output, pixels=True)
    except Exception as exc:
        value = {
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    value.update(
        render_seconds=time.perf_counter() - start,
        peak_process_rss_bytes=rss if platform.system() == "Darwin" else rss * 1024,
        memory_scope="fresh worker total high-water RSS, includes interpreter/dependencies",
    )
    atomic_json(result, value)


if __name__ == "__main__":
    main()
