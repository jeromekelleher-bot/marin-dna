import json
from pathlib import Path
import time
while True:
    files = [Path("/sys/fs/cgroup") / k for k in ("cpu.stat","cpu.max","cpu.pressure","memory.current","memory.events","memory.pressure","cpuset.cpus.effective")]
    files += [Path("/proc/pressure/cpu"),Path("/proc/loadavg")]
    print(json.dumps({"epoch":time.time(), **{str(f):f.read_text().strip() for f in files if f.is_file()}}),flush=True)
    time.sleep(2)
