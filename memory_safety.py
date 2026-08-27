"""外部依存なしで取得できるシステムメモリ情報と安全弁。"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MemorySnapshot:
    platform: str
    total_bytes: Optional[int]
    available_percent: Optional[float]


class MemoryBudgetExceeded(RuntimeError):
    """次の生成を開始するには空きメモリが少なすぎる。"""


def snapshot() -> MemorySnapshot:
    """macOS/Linuxのシステム全体メモリを概算する。"""
    current_platform = platform.system()
    if current_platform == "Darwin":
        try:
            result = subprocess.run(
                ["memory_pressure", "-Q"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            total_match = re.search(r"has (\d+) \(", result.stdout)
            free_match = re.search(
                r"memory free percentage:\s*(\d+(?:\.\d+)?)%",
                result.stdout,
                flags=re.IGNORECASE,
            )
            return MemorySnapshot(
                platform=current_platform,
                total_bytes=int(total_match.group(1)) if total_match else None,
                available_percent=float(free_match.group(1)) if free_match else None,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    if current_platform == "Linux":
        try:
            values = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, separator, raw_value = line.partition(":")
                if not separator:
                    continue
                values[key] = int(raw_value.strip().split()[0]) * 1024
            total = values.get("MemTotal")
            available = values.get("MemAvailable")
            return MemorySnapshot(
                platform=current_platform,
                total_bytes=total,
                available_percent=(available / total * 100 if total and available else None),
            )
        except (OSError, ValueError):
            pass

    return MemorySnapshot(
        platform=current_platform,
        total_bytes=None,
        available_percent=None,
    )


def ensure_available(minimum_percent: float) -> MemorySnapshot:
    """次の画像生成を開始してよいか確認する。"""
    current = snapshot()
    available = current.available_percent
    if available is not None and available < minimum_percent:
        raise MemoryBudgetExceeded(
            f"Available system memory is {available:.1f}%, below the configured "
            f"minimum of {minimum_percent:.1f}%. Stop generation and release memory."
        )
    return current
