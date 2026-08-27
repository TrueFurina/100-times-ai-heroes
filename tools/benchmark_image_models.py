#!/usr/bin/env python3
"""候補画像モデルを同じキャラクター設定とseedで比較する。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from comfyui_image_gen import ComfyUIImageGenerator  # noqa: E402
from image_model_profiles import (  # noqa: E402
    ImageModelProfile,
    build_image_prompt,
    get_image_model_profile,
)
from memory_safety import MemorySnapshot, snapshot as read_memory_snapshot  # noqa: E402


DEFAULT_PROFILE_IDS = [
    "animagine-xl-4.0-opt",
    "illustrious-xl-v2",
    "pony-v6-xl",
    "noobai-xl-1.1",
]

BENCHMARK_CASES = [
    {
        "id": "human-warrior",
        "age": "young adult",
        "gender": "male",
        "species": "human",
        "role": "Swordsman, disciplined guardian",
        "ability": "Can cut through digital noise with a single stroke",
        "concept": "A young human swordsman who protects people from information chaos with a calm sense of duty.",
    },
    {
        "id": "aquatic-designer",
        "age": "adult",
        "gender": "female",
        "species": "half-human half-aquatic woman",
        "role": "Nostalgic Experience Designer",
        "ability": "Can reverse causality through rhythmic movement",
        "concept": "An elegant aquatic dancer who designs shared memories and dreams of a cooperative utopia.",
    },
    {
        "id": "lycanthrope-artist",
        "age": "middle-aged",
        "gender": "male",
        "species": "lycanthrope",
        "role": "Biotechnology Tattoo Artist",
        "ability": "Can absorb emotions as colors and animate tattoos",
        "concept": "A middle-aged lycanthrope artist whose living tattoos connect people through visible emotions.",
    },
]


def _memory_snapshot() -> MemorySnapshot:
    return read_memory_snapshot()


class MemorySampler:
    """画像生成中の利用可能メモリの最低値を記録する。"""

    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = interval_seconds
        self.samples: List[MemorySnapshot] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "MemorySampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(_memory_snapshot())
            self._stop.wait(self.interval_seconds)

    @property
    def minimum_available_percent(self) -> Optional[float]:
        values = [sample.available_percent for sample in self.samples]
        values = [value for value in values if value is not None]
        return min(values) if values else None


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _profile_generator(
    profile: ImageModelProfile,
    *,
    url: str,
    workflow_path: Path,
    output_dir: Path,
    seed: int,
) -> ComfyUIImageGenerator:
    return ComfyUIImageGenerator(
        base_url=url,
        workflow_path=workflow_path,
        checkpoint_name=profile.checkpoint_name,
        timeout_seconds=300,
        poll_interval_seconds=1,
        width=profile.width,
        height=profile.height,
        steps=profile.steps,
        cfg=profile.cfg,
        sampler=profile.sampler,
        scheduler=profile.scheduler,
        negative_prompt=profile.negative_prompt,
        clip_skip=profile.clip_skip,
        seed_factory=lambda seed=seed: seed,
    )


def _iter_cases(case_count: int) -> Iterable[dict]:
    if case_count < 1 or case_count > len(BENCHMARK_CASES):
        raise ValueError(f"--cases must be between 1 and {len(BENCHMARK_CASES)}")
    return BENCHMARK_CASES[:case_count]


def _case_prompt(profile: ImageModelProfile, case: dict) -> str:
    values = {key: value for key, value in case.items() if key != "id"}
    return build_image_prompt(profile, **values)


def run_benchmark(args: argparse.Namespace) -> dict:
    profile_ids = DEFAULT_PROFILE_IDS if args.profiles == ["all"] else args.profiles
    profiles = [
        get_image_model_profile(profile_id, args.profiles_path)
        for profile_id in profile_ids
    ]
    cases = list(_iter_cases(args.cases))
    seeds = args.seeds
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "host": platform.platform(),
        "profiles": [asdict(profile) for profile in profiles],
        "cases": [case["id"] for case in cases],
        "seeds": seeds,
        "results": [],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        profile_dir = args.output / profile.profile_id
        if args.dry_run:
            report["results"].append(
                {
                    "profile": profile.profile_id,
                    "status": "dry-run",
                    "prompts": [
                        {
                            "case": case["id"],
                            "prompt": _case_prompt(profile, case),
                        }
                        for case in cases
                    ],
                }
            )
            continue

        generator = _profile_generator(
            profile,
            url=args.url,
            workflow_path=args.workflow_path,
            output_dir=profile_dir,
            seed=seeds[0],
        )
        try:
            generator.check_connection()
        except Exception as exc:
            report["results"].append(
                {
                    "profile": profile.profile_id,
                    "status": "connection-error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        for case in cases:
            for seed in seeds:
                prompt = _case_prompt(profile, case)
                before = _memory_snapshot()
                started = time.monotonic()
                sampler = MemorySampler()
                result = {
                    "profile": profile.profile_id,
                    "case": case["id"],
                    "seed": seed,
                    "prompt": prompt,
                    "memory_before": asdict(before),
                }
                try:
                    with sampler:
                        image = _profile_generator(
                            profile,
                            url=args.url,
                            workflow_path=args.workflow_path,
                            output_dir=profile_dir / case["id"],
                            seed=seed,
                        ).generate(
                            prompt,
                            profile_dir / case["id"],
                            f"{case['id']}_{seed}",
                        )
                    result.update(
                        {
                            "status": "success",
                            "image_path": str(image.path),
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                        }
                    )
                except Exception as exc:
                    result.update(
                        {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                        }
                    )
                result["minimum_available_percent"] = sampler.minimum_available_percent
                result["memory_after"] = asdict(_memory_snapshot())
                report["results"].append(result)

        if not args.keep_models_loaded:
            try:
                generator.free_memory()
            except Exception as exc:
                report["results"].append(
                    {
                        "profile": profile.profile_id,
                        "status": "free-memory-warning",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    report["summary"] = {
        "success": sum(result.get("status") == "success" for result in report["results"]),
        "errors": sum(result.get("status") == "error" for result in report["results"]),
        "connection_errors": sum(
            result.get("status") == "connection-error" for result in report["results"]
        ),
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=_parse_csv,
        default=["all"],
        help="比較するprofile ID（カンマ区切り、既定: all）",
    )
    parser.add_argument("--cases", type=int, default=3, help="固定ケース数（1-3）")
    parser.add_argument(
        "--seeds",
        type=_parse_csv,
        default=["101", "202"],
        help="固定seed（カンマ区切り）",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument(
        "--workflow-path",
        type=Path,
        default=PROJECT_ROOT / "config/comfyui/text2image_api_workflow.json",
    )
    parser.add_argument(
        "--profiles-path",
        type=Path,
        default=PROJECT_ROOT / "config/comfyui/model_profiles.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / f"model_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ComfyUIへ接続せず、profileとpromptだけ検証する",
    )
    parser.add_argument(
        "--keep-models-loaded",
        action="store_true",
        help="profile切り替え時にComfyUIの/freeを呼ばない",
    )
    namespace = parser.parse_args()
    namespace.seeds = [int(seed) for seed in namespace.seeds]
    if not namespace.seeds:
        parser.error("--seeds must contain at least one integer")
    return namespace


if __name__ == "__main__":
    arguments = parse_args()
    result = run_benchmark(arguments)
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(f"Report: {arguments.output / 'report.json'}")
