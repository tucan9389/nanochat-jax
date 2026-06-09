"""Markdown report generation for nanochat-jax public runs.

This mirrors upstream ``nanochat.report`` at the interface level while avoiding
PyTorch/runtime dependency additions. Individual scripts call
``get_report().log(...)`` and ``python -m nanochat_jax.report generate`` stitches
the sections together.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import platform
import re
import shutil
import socket
import subprocess
from collections.abc import Mapping
from typing import Any

from nanochat_jax.common import get_base_dir


EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]

SUMMARY_KEYS = [
    "CORE metric",
    "train bpb",
    "val bpb",
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "GSM8K",
    "HumanEval",
    "SpellingBee",
    "ChatCORE metric",
]


def run_command(cmd: str, *, timeout: float = 5.0) -> str | None:
    """Run a small shell command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return None
    if result.stdout.strip():
        return result.stdout.strip()
    if result.returncode == 0:
        return ""
    return None


def slugify(text: str) -> str:
    """Convert a report section title to an upstream-style markdown filename."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _format_value(value: Any) -> str:
    """Format values for stable markdown report output."""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int) and abs(value) >= 10_000:
        return f"{value:,}"
    if value is None:
        return "None"
    return str(value)


def _extract_timestamp(content: str, prefix: str) -> _dt.datetime | None:
    for line in content.splitlines():
        if not line.startswith(prefix):
            continue
        value = line.split(":", 1)[1].strip()
        try:
            return _dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


def _extract_metrics(section: str, keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    wanted = set(keys)
    for line in section.splitlines():
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        key = key.strip()
        if key in wanted:
            out[key] = value.strip()
    return out


def _get_git_info() -> dict[str, Any]:
    status = run_command("git status --porcelain")
    message = run_command("git log -1 --pretty=%B") or ""
    return {
        "branch": run_command("git rev-parse --abbrev-ref HEAD") or "unknown",
        "commit": run_command("git rev-parse --short HEAD") or "unknown",
        "dirty": bool(status) if status is not None else None,
        "message": message.splitlines()[0][:80] if message else "",
    }


def _get_system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "user": os.environ.get("USER", "unknown"),
        "working_dir": os.getcwd(),
        "nanochat_jax_base_dir": get_base_dir(),
    }
    try:
        import psutil  # type: ignore[import-not-found]

        info["cpu_count"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["memory_gb"] = psutil.virtual_memory().total / (1024**3)
    except Exception:
        info["cpu_count"] = os.cpu_count()
        info["cpu_count_logical"] = os.cpu_count()
        info["memory_gb"] = None
    return info


def _get_jax_info() -> dict[str, Any]:
    try:
        import jax
        import jaxlib

        devices = jax.devices()
        device_kinds = sorted({getattr(device, "device_kind", str(device)) for device in devices})
        platforms = sorted({getattr(device, "platform", "unknown") for device in devices})
        return {
            "available": True,
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "process_index": int(jax.process_index()),
            "process_count": int(jax.process_count()),
            "local_device_count": int(jax.local_device_count()),
            "device_count": int(jax.device_count()),
            "platforms": ", ".join(platforms) if platforms else "unknown",
            "device_kinds": ", ".join(device_kinds) if device_kinds else "unknown",
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _get_bloat_info() -> dict[str, int]:
    extensions = ["py", "md", "toml", "sh", "html", "svg"]
    patterns = " ".join(f"'*.{ext}'" for ext in extensions)
    files_output = run_command(f"git ls-files -- {patterns}") or ""
    files = [line for line in files_output.splitlines() if line]
    num_lines = 0
    num_chars = 0
    if files:
        wc_output = run_command(f"git ls-files -- {patterns} | xargs wc -lc 2>/dev/null")
        if wc_output:
            parts = wc_output.splitlines()[-1].split()
            if len(parts) >= 2:
                try:
                    num_lines = int(parts[0])
                    num_chars = int(parts[1])
                except ValueError:
                    num_lines = 0
                    num_chars = 0
    return {
        "files": len(files),
        "lines": num_lines,
        "characters": num_chars,
        "tokens_approx": num_chars // 4,
    }


def generate_header() -> str:
    """Generate the report header markdown."""
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_info = _get_git_info()
    sys_info = _get_system_info()
    jax_info = _get_jax_info()
    bloat = _get_bloat_info()

    dirty = git_info["dirty"]
    dirty_label = "dirty" if dirty else "clean" if dirty is False else "unknown"
    memory = sys_info["memory_gb"]
    memory_line = f"- Memory: {memory:.1f} GB\n" if isinstance(memory, float) else "- Memory: unknown\n"

    lines = [
        "# nanochat-jax training report",
        "",
        f"Generated: {timestamp}",
        "",
        "## Environment",
        "",
        "### Git Information",
        f"- Branch: {git_info['branch']}",
        f"- Commit: {git_info['commit']} ({dirty_label})",
        f"- Message: {git_info['message']}",
        "",
        "### Hardware",
        f"- Hostname: {sys_info['hostname']}",
        f"- Platform: {sys_info['platform']}",
        f"- CPUs: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)",
        memory_line.rstrip(),
    ]
    if jax_info.get("available"):
        lines.extend([
            f"- JAX devices: {jax_info['device_count']} total / {jax_info['local_device_count']} local",
            f"- JAX platforms: {jax_info['platforms']}",
            f"- JAX device kinds: {jax_info['device_kinds']}",
            f"- JAX process: {jax_info['process_index']} / {jax_info['process_count']}",
        ])
    else:
        lines.append(f"- JAX devices: unavailable ({jax_info.get('error', 'unknown')})")
    lines.extend([
        "",
        "### Software",
        f"- Python: {sys_info['python_version']}",
        f"- JAX: {jax_info.get('jax_version', 'unknown')}",
        f"- jaxlib: {jax_info.get('jaxlib_version', 'unknown')}",
        f"- Working directory: {sys_info['working_dir']}",
        f"- NANOCHAT_JAX_BASE_DIR: {sys_info['nanochat_jax_base_dir']}",
        "",
        "### Bloat",
        f"- Characters: {bloat['characters']:,}",
        f"- Lines: {bloat['lines']:,}",
        f"- Files: {bloat['files']:,}",
        f"- Tokens (approx): {bloat['tokens_approx']:,}",
        "",
    ])
    return "\n".join(lines)


class Report:
    """Maintains per-stage markdown sections and assembles the final report."""

    def __init__(self, report_dir: str):
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

    def log(self, section: str, data: list[Any]) -> str:
        """Write one report section."""
        path = os.path.join(self.report_dir, f"{slugify(section)}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    continue
                if isinstance(item, str):
                    f.write(item)
                    if not item.endswith("\n"):
                        f.write("\n")
                    continue
                if isinstance(item, Mapping):
                    for key, value in item.items():
                        if isinstance(value, Mapping):
                            f.write(f"- {key}:\n")
                            for nested_key, nested_value in value.items():
                                f.write(f"  - {nested_key}: {_format_value(nested_value)}\n")
                        else:
                            f.write(f"- {key}: {_format_value(value)}\n")
                    continue
                f.write(f"- {item}\n")
            f.write("\n")
        return path

    def reset(self) -> str:
        """Clear prior markdown report sections and write a fresh header."""
        for name in os.listdir(self.report_dir):
            if name.endswith(".md"):
                os.remove(os.path.join(self.report_dir, name))
        header = generate_header()
        start_time = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_path = os.path.join(self.report_dir, "header.md")
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(header.rstrip())
            f.write("\n\n")
            f.write(f"Run started: {start_time}\n\n---\n\n")
        print(f"Reset report and wrote header to {header_path}")
        return header_path

    def generate(self) -> str:
        """Generate the final combined report and copy it to ./report.md."""
        report_file = os.path.join(self.report_dir, "report.md")
        final_metrics: dict[str, dict[str, str]] = {}
        start_time = None
        end_time = None
        bloat_data = "[bloat data missing]"

        with open(report_file, "w", encoding="utf-8") as out:
            header_file = os.path.join(self.report_dir, "header.md")
            if os.path.exists(header_file):
                header = open(header_file, encoding="utf-8").read()
                out.write(header)
                start_time = _extract_timestamp(header, "Run started:")
                match = re.search(r"### Bloat\n(.*?)\n\n", header, re.DOTALL)
                if match:
                    bloat_data = match.group(1)
            else:
                print(f"Warning: {header_file} does not exist; skipping header")

            for filename in EXPECTED_FILES:
                section_file = os.path.join(self.report_dir, filename)
                if not os.path.exists(section_file):
                    print(f"Warning: {section_file} does not exist, skipping")
                    continue
                section = open(section_file, encoding="utf-8").read()
                if "rl" not in filename:
                    end_time = _extract_timestamp(section, "timestamp:") or end_time
                if filename == "base-model-evaluation.md":
                    final_metrics["base"] = _extract_metrics(section, SUMMARY_KEYS)
                elif filename == "chat-evaluation-sft.md":
                    final_metrics["sft"] = _extract_metrics(section, SUMMARY_KEYS)
                elif filename == "chat-evaluation-rl.md":
                    final_metrics["rl"] = _extract_metrics(section, SUMMARY_KEYS)
                out.write(section)
                out.write("\n")

            out.write("## Summary\n\n")
            out.write(bloat_data)
            out.write("\n\n")
            self._write_summary_table(out, final_metrics)
            if start_time and end_time:
                total_seconds = int((end_time - start_time).total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out.write(f"\nTotal wall clock time: {hours}h{minutes}m\n")
            else:
                out.write("\nTotal wall clock time: unknown\n")

        print("Copying report.md to current directory for convenience")
        shutil.copy(report_file, "report.md")
        return report_file

    @staticmethod
    def _write_summary_table(out, final_metrics: dict[str, dict[str, str]]) -> None:
        metrics = sorted(
            {key for values in final_metrics.values() for key in values},
            key=lambda key: (
                key != "CORE metric",
                key != "ChatCORE metric",
                key,
            ),
        )
        stages = ["base", "sft", "rl"]
        out.write("| Metric | BASE | SFT | RL |\n")
        out.write("|---|---:|---:|---:|\n")
        if not metrics:
            out.write("| - | - | - | - |\n")
            return
        for metric in metrics:
            values = [final_metrics.get(stage, {}).get(metric, "-") for stage in stages]
            out.write(f"| {metric} | {values[0]} | {values[1]} | {values[2]} |\n")


class DummyReport:
    """No-op report used on nonzero JAX processes."""

    def log(self, *args, **kwargs) -> None:
        return None

    def reset(self, *args, **kwargs) -> None:
        return None

    def generate(self, *args, **kwargs) -> None:
        return None


def get_report() -> Report | DummyReport:
    """Return a report writer on JAX process 0, otherwise a no-op writer."""
    try:
        import jax

        if jax.process_index() != 0:
            return DummyReport()
    except Exception:
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            return DummyReport()
    return Report(os.path.join(get_base_dir(), "report"))


def log_report_safe(section: str, data: list[Any]) -> None:
    """Best-effort report logging that never fails the caller stage."""
    try:
        get_report().log(section=section, data=data)
    except Exception as exc:
        print(f"[warn] Report logging failed for {section!r}: {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate or reset nanochat-jax training reports.")
    parser.add_argument(
        "command",
        nargs="?",
        default="generate",
        choices=["generate", "reset"],
        help="Operation to perform (default: generate)",
    )
    args = parser.parse_args(argv)
    report = get_report()
    if args.command == "reset":
        report.reset()
    else:
        report.generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
