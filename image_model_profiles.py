"""ローカル画像生成モデルの設定とプロンプト形式。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, bad anatomy, bad hands, extra fingers, cropped, "
    "duplicate, text, watermark, signature"
)


@dataclass(frozen=True)
class ImageModelProfile:
    """ComfyUIへ渡すモデル固有の設定。"""

    profile_id: str
    checkpoint_name: str
    prompt_style: str
    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    negative_prompt: str
    clip_skip: Optional[int] = None
    experimental: bool = False
    notes: str = ""
    source_url: str = ""
    license_name: str = ""


BUILTIN_PROFILES: Dict[str, ImageModelProfile] = {
    "generic-sdxl": ImageModelProfile(
        profile_id="generic-sdxl",
        checkpoint_name="CHANGE_ME.safetensors",
        prompt_style="prose",
        width=768,
        height=1152,
        steps=24,
        cfg=7.0,
        sampler="euler",
        scheduler="normal",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        notes="既存の汎用workflow互換用。最終候補の評価対象ではない。",
    ),
    "animagine-xl-4.0-opt": ImageModelProfile(
        profile_id="animagine-xl-4.0-opt",
        checkpoint_name="animagine-xl-4.0-opt.safetensors",
        prompt_style="animagine",
        width=832,
        height=1216,
        steps=28,
        cfg=5.0,
        sampler="euler_ancestral",
        scheduler="normal",
        negative_prompt=(
            "lowres, bad anatomy, bad hands, text, error, missing finger, "
            "extra digits, fewer digits, cropped, worst quality, low quality, "
            "low score, bad score, average score, signature, watermark, username, blurry"
        ),
        notes="アニメ・JRPG系の最初の比較候補。",
        source_url="https://huggingface.co/cagliostrolab/animagine-xl-4.0",
        license_name="CreativeML Open RAIL++-M",
    ),
    "illustrious-xl-v2": ImageModelProfile(
        profile_id="illustrious-xl-v2",
        checkpoint_name="Illustrious-XL-v2.0.safetensors",
        prompt_style="illustrious",
        width=832,
        height=1216,
        steps=30,
        cfg=7.0,
        sampler="euler_ancestral",
        scheduler="normal",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        notes="線と色を重視するイラスト寄りの比較候補。",
        source_url="https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0",
        license_name="CreativeML Open RAIL-M",
    ),
    "pony-v6-xl": ImageModelProfile(
        profile_id="pony-v6-xl",
        checkpoint_name="ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
        prompt_style="pony",
        width=832,
        height=1216,
        steps=25,
        cfg=7.0,
        sampler="euler_ancestral",
        scheduler="normal",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        clip_skip=-2,
        experimental=True,
        notes="獣人・異種族には有力。ただしclip skip 2とライセンス条件を確認する。",
        source_url="https://huggingface.co/LyliaEngine/Pony_Diffusion_V6_XL",
        license_name="Modified Fair AI Public License 1.0-SD",
    ),
    "noobai-xl-1.1": ImageModelProfile(
        profile_id="noobai-xl-1.1",
        checkpoint_name="NoobAI-XL-v1.1.safetensors",
        prompt_style="noobai",
        width=832,
        height=1216,
        steps=28,
        cfg=5.0,
        sampler="euler_ancestral",
        scheduler="normal",
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        experimental=True,
        notes="品質候補。ただし利用条件とComfyUI固有設定を検証してから採用する。",
        source_url="https://huggingface.co/Laxhar/noobai-XL-1.1",
        license_name="Fair AI Public License 1.0-SD",
    ),
}


def _profile_from_dict(profile_id: str, value: dict) -> ImageModelProfile:
    required = {
        "checkpoint_name",
        "prompt_style",
        "width",
        "height",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
        "negative_prompt",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"Image model profile {profile_id} is missing: {', '.join(missing)}")
    return ImageModelProfile(
        profile_id=profile_id,
        checkpoint_name=str(value["checkpoint_name"]),
        prompt_style=str(value["prompt_style"]),
        width=int(value["width"]),
        height=int(value["height"]),
        steps=int(value["steps"]),
        cfg=float(value["cfg"]),
        sampler=str(value["sampler"]),
        scheduler=str(value["scheduler"]),
        negative_prompt=str(value["negative_prompt"]),
        clip_skip=(int(value["clip_skip"]) if value.get("clip_skip") is not None else None),
        experimental=bool(value.get("experimental", False)),
        notes=str(value.get("notes", "")),
        source_url=str(value.get("source_url", "")),
        license_name=str(value.get("license_name", "")),
    )


def load_image_model_profiles(path: Optional[Path] = None) -> Dict[str, ImageModelProfile]:
    """組み込み設定に、存在する場合はJSON設定を重ねる。"""
    profiles = dict(BUILTIN_PROFILES)
    if path is None or not Path(path).exists():
        return profiles

    try:
        with Path(path).open(encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read image model profiles: {path}") from exc

    raw_profiles = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(raw_profiles, dict):
        raise ValueError("Image model profiles JSON must contain an object named profiles")
    for profile_id, value in raw_profiles.items():
        if not isinstance(value, dict):
            raise ValueError(f"Image model profile {profile_id} must be an object")
        profiles[str(profile_id)] = _profile_from_dict(str(profile_id), value)
    return profiles


def get_image_model_profile(profile_id: str, path: Optional[Path] = None) -> ImageModelProfile:
    profiles = load_image_model_profiles(path)
    try:
        return profiles[profile_id]
    except KeyError as exc:
        available = ", ".join(sorted(profiles))
        raise ValueError(
            f"Unknown image model profile: {profile_id}. Available profiles: {available}"
        ) from exc


def _compact(value: str) -> str:
    value = re.sub(r"[\r\n]+", " ", str(value or ""))
    return " ".join(value.split()).strip(" ,")


def _gender_tag(gender: str) -> str:
    normalized = _compact(gender).lower()
    if any(token in normalized for token in ("female", "woman", "girl", "女性", "女")):
        return "1girl"
    if any(token in normalized for token in ("male", "man", "boy", "男性", "男")):
        return "1boy"
    if any(token in normalized for token in ("nonbinary", "non-binary", "ノンバイナリー")):
        return "1other"
    return "1person"


def _tag_phrase(value: str) -> str:
    """自由記述を1つの短い視覚属性として扱う。"""
    return _compact(value).replace(";", ",")


def build_image_prompt(
    profile: ImageModelProfile,
    *,
    concept: str,
    age: str = "",
    gender: str = "",
    species: str = "",
    ability: str = "",
    role: str = "",
) -> str:
    """プロファイルごとのprompt形式で、全身キャラクター向けpromptを作る。"""
    if profile.prompt_style == "prose":
        return (
            "The full-length character illustration from video games, likely from "
            "role-playing games (JRPG) or fighting games. A camera angle that captures "
            "the entire body evenly from waist height. Standing upright and looking "
            "straight ahead. White background. "
            f"{_compact(concept)}, delicate hand-drawn lines, Japanese manga and anime "
            "influence, realistic proportions, detailed textures, sophisticated haute "
            "couture fashion, edgy character design, strong individuality."
        )

    tags = [
        _gender_tag(gender),
        "solo",
        "full body",
        "standing",
        "looking at viewer",
        "feet visible",
        "centered composition",
        "white background",
    ]
    for value in (age, species, role, ability):
        phrase = _tag_phrase(value)
        if phrase:
            tags.append(phrase)

    if profile.prompt_style == "pony":
        tags.extend(
            [
                "source_anime",
                "score_9",
                "score_8_up",
                "score_7_up",
            ]
        )
    elif profile.prompt_style == "noobai":
        tags.extend(["masterpiece", "best quality", "highly detailed"])
    else:
        tags.extend(["masterpiece", "high score", "great score", "absurdres"])

    concept_text = _compact(concept)
    prompt = ", ".join(tags)
    return f"{prompt}, {concept_text}" if concept_text else prompt
