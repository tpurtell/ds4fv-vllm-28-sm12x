#!/usr/bin/env python3
"""Exercise local/SSH launcher dispatch without Docker, vLLM, or CUDA."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch-one-spark-exl3.sh"


def executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def run_launcher(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"launcher exited {result.returncode}\nstdout:\n{result.stdout}"
            f"\nstderr:\n{result.stderr}"
        )
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ds4fv-launch-smoke-") as tmp:
        root = Path(tmp)
        fake_bin = root / "bin"
        fake_bin.mkdir()
        trace = root / "trace"

        executable(fake_bin / "uname", "#!/bin/sh\necho aarch64\n")
        executable(fake_bin / "hostname", "#!/bin/sh\necho local-spark\n")
        executable(
            fake_bin / "docker",
            """#!/bin/sh
printf 'docker %s\\n' "$*" >> "$DS4FV_TEST_TRACE"
if [ "$1 $2" = "container inspect" ]; then
  exit 1
fi
exit 0
""",
        )
        executable(
            fake_bin / "ssh",
            """#!/bin/sh
printf 'ssh %s\\n' "$*" >> "$DS4FV_TEST_TRACE"
last=''
for argument do last=$argument; done
case "$last" in
  *'uname -m'*) echo aarch64 ;;
  *'command -v docker'*) echo /usr/bin/docker ;;
  *'docker container inspect'*) exit 1 ;;
esac
exit 0
""",
        )

        environment = os.environ.copy()
        for name in ("SPARK_HOST", "DS4FV_IMAGE", "CONTAINER_NAME"):
            environment.pop(name, None)
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "DS4FV_TEST_TRACE": str(trace),
                "HF_CACHE": str(root / "cache"),
            }
        )

        local = run_launcher(environment)
        local_trace = trace.read_text()
        assert "ssh " not in local_trace
        assert "docker run -d" in local_trace
        assert "ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.1" in local_trace
        assert "Started local-spark/ds4fv-exl3." in local.stdout
        assert "Follow startup: docker logs -f ds4fv-exl3" in local.stdout

        trace.write_text("")
        remote_environment = environment | {"SPARK_HOST": "kiwi"}
        remote = run_launcher(remote_environment)
        remote_trace = trace.read_text()
        assert "ssh -o BatchMode=yes kiwi" in remote_trace
        assert "docker run -d" in remote_trace
        assert "Started kiwi/ds4fv-exl3." in remote.stdout
        assert "Follow startup: ssh kiwi docker logs -f ds4fv-exl3" in remote.stdout

    print("one-Spark local/SSH launcher dispatch smoke passed")


if __name__ == "__main__":
    main()
