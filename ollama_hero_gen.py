"""
100 Times AI Heroes - ローカル完結版キャラクター生成システム

既定では、ローカルCSVとlocalhost上のOllamaだけを使用します。
画像生成は --generate-images を指定した場合だけ、localhost上のComfyUIを使用します。
クラウドLLMは --provider openai を明示した場合だけ有効になります。
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import ollama
from dotenv import load_dotenv
from image_model_profiles import (
    build_image_prompt,
    get_image_model_profile,
)
from memory_safety import MemoryBudgetExceeded, ensure_available


# =============================================================================
# Exceptions and protocols
# =============================================================================


class OllamaConnectionError(ConnectionError):
    """Ollamaサーバーへ接続できない。"""


class OllamaModelNotFoundError(RuntimeError):
    """指定されたOllamaモデルが導入されていない。"""


class ProviderConfigError(ValueError):
    """LLMプロバイダーの設定が不正。"""


class SeedDataError(ValueError):
    """seed CSVの形式または内容が不正。"""


class ComfyUIError(RuntimeError):
    """ComfyUIとの通信または画像生成に失敗した。"""


class StageError(RuntimeError):
    """生成パイプラインの特定段階で失敗した。"""

    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        self.cause = cause
        super().__init__(str(cause))


class TextGenerator(Protocol):
    """テキスト生成プロバイダーの共通インターフェース。"""

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        ...


# =============================================================================
# Configuration
# =============================================================================


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (true/false): {value}")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer: {value}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number: {value}") from exc


@dataclass(frozen=True)
class Config:
    """アプリケーション設定。Google Sheetsの設定は持たない。"""

    model: str
    host: str
    data_dir: str
    num_iterations: int = 100
    provider: str = "ollama"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    generate_images: bool = False
    comfyui_url: str = "http://127.0.0.1:8188"
    comfyui_model_profile: str = "generic-sdxl"
    comfyui_model_profiles_path: str = "./config/comfyui/model_profiles.json"
    comfyui_workflow_path: str = "./config/comfyui/text2image_api_workflow.json"
    comfyui_checkpoint_name: str = "CHANGE_ME.safetensors"
    comfyui_timeout_seconds: float = 300.0
    comfyui_poll_interval_seconds: float = 1.0
    comfyui_width: int = 768
    comfyui_height: int = 1152
    comfyui_steps: int = 24
    comfyui_cfg: float = 7.0
    comfyui_sampler: str = "euler"
    comfyui_scheduler: str = "normal"
    comfyui_clip_skip: Optional[int] = None
    memory_guard_enabled: bool = True
    minimum_available_memory_percent: float = 15.0
    comfyui_negative_prompt: str = (
        "low quality, blurry, distorted hands, extra fingers, cropped, duplicate"
    )

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()

        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        if provider not in {"ollama", "openai"}:
            raise ProviderConfigError(
                f"Unsupported LLM_PROVIDER: {provider}. Use ollama or openai."
            )

        ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        selected_model = openai_model if provider == "openai" else ollama_model

        model_profile_id = os.getenv(
            "COMFYUI_MODEL_PROFILE", "animagine-xl-4.0-opt"
        ).strip()
        model_profiles_path = os.getenv(
            "COMFYUI_MODEL_PROFILES_PATH",
            "./config/comfyui/model_profiles.json",
        )
        image_profile = get_image_model_profile(
            model_profile_id, Path(model_profiles_path)
        )
        configured_checkpoint = os.getenv("COMFYUI_CHECKPOINT_NAME", "").strip()
        checkpoint_name = (
            configured_checkpoint
            if configured_checkpoint and configured_checkpoint != "CHANGE_ME.safetensors"
            else image_profile.checkpoint_name
        )

        config = cls(
            model=selected_model,
            host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            data_dir=os.getenv("DATA_DIR", "./data"),
            provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=openai_model,
            generate_images=_env_bool("GENERATE_IMAGES", False),
            comfyui_url=os.getenv("COMFYUI_URL", "http://127.0.0.1:8188"),
            comfyui_model_profile=model_profile_id,
            comfyui_model_profiles_path=model_profiles_path,
            comfyui_workflow_path=os.getenv(
                "COMFYUI_WORKFLOW_PATH",
                "./config/comfyui/text2image_api_workflow.json",
            ),
            comfyui_checkpoint_name=checkpoint_name,
            comfyui_timeout_seconds=_env_float("COMFYUI_TIMEOUT_SECONDS", 300.0),
            comfyui_poll_interval_seconds=_env_float(
                "COMFYUI_POLL_INTERVAL_SECONDS", 1.0
            ),
            comfyui_width=_env_int("COMFYUI_WIDTH", image_profile.width),
            comfyui_height=_env_int("COMFYUI_HEIGHT", image_profile.height),
            comfyui_steps=_env_int("COMFYUI_STEPS", image_profile.steps),
            comfyui_cfg=_env_float("COMFYUI_CFG", image_profile.cfg),
            comfyui_sampler=os.getenv("COMFYUI_SAMPLER", image_profile.sampler),
            comfyui_scheduler=os.getenv(
                "COMFYUI_SCHEDULER", image_profile.scheduler
            ),
            comfyui_clip_skip=image_profile.clip_skip,
            memory_guard_enabled=_env_bool("MEMORY_GUARD_ENABLED", True),
            minimum_available_memory_percent=_env_float(
                "MINIMUM_AVAILABLE_MEMORY_PERCENT", 15.0
            ),
            comfyui_negative_prompt=os.getenv(
                "COMFYUI_NEGATIVE_PROMPT",
                image_profile.negative_prompt,
            ),
        )

        if config.comfyui_timeout_seconds <= 0:
            raise ValueError("COMFYUI_TIMEOUT_SECONDS must be greater than 0")
        if config.comfyui_poll_interval_seconds <= 0:
            raise ValueError("COMFYUI_POLL_INTERVAL_SECONDS must be greater than 0")
        if config.comfyui_width <= 0 or config.comfyui_height <= 0:
            raise ValueError("COMFYUI_WIDTH and COMFYUI_HEIGHT must be greater than 0")
        if config.num_iterations < 1:
            raise ValueError("num_iterations must be greater than 0")
        if config.minimum_available_memory_percent < 0 or config.minimum_available_memory_percent > 100:
            raise ValueError(
                "MINIMUM_AVAILABLE_MEMORY_PERCENT must be between 0 and 100"
            )
        return config


# =============================================================================
# LLM providers
# =============================================================================


class OllamaInference:
    """localhost上のOllamaを使うテキスト生成クライアント。"""

    SYSTEM_PROMPT = (
        "人間の仕事を助ける優秀なAIアシスタントとして、"
        "指示に従い、必要な情報のみを端的に出力します。"
    )

    def __init__(self, config: Config, client: Any = None):
        self.config = config
        self.client = client or ollama.Client(host=config.host, timeout=120)
        self._ensure_model_available()

    @staticmethod
    def _model_name(model_info: Any) -> str:
        if hasattr(model_info, "model"):
            return str(model_info.model)
        if isinstance(model_info, dict):
            return str(model_info.get("name") or model_info.get("model") or "")
        return str(getattr(model_info, "name", ""))

    def _ensure_model_available(self) -> None:
        """モデルを確認する。モデルの自動pullは行わない。"""
        try:
            models = self.client.list()
        except Exception as exc:
            raise OllamaConnectionError(
                f"Ollama server is not available at {self.config.host}. "
                "Start it with: ollama serve"
            ) from exc

        if isinstance(models, dict):
            model_list = models.get("models", [])
        else:
            model_list = getattr(models, "models", [])
        available = {self._model_name(item) for item in model_list}
        requested = self.config.model
        matches = requested in available or f"{requested}:latest" in available
        if not matches:
            raise OllamaModelNotFoundError(
                f"Model '{requested}' is not installed. "
                f"Run: ollama pull {requested}"
            )

    @staticmethod
    def _response_content(response: Any) -> str:
        if isinstance(response, dict):
            message = response.get("message", {})
            if isinstance(message, dict):
                return str(message.get("content", "")).strip()
            return str(getattr(message, "content", "")).strip()

        message = getattr(response, "message", None)
        return str(getattr(message, "content", "")).strip()

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """指数バックオフ付きでテキストを生成する。"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options={
                        "num_predict": 2048,
                        "temperature": 0.8,
                    },
                )
                content = self._response_content(response)
                if not content:
                    raise RuntimeError("Ollama returned an empty response")
                return content
            except Exception:
                if attempt == max_retries - 1:
                    raise
                delay = 2**attempt
                print(
                    f"Retry {attempt + 1}/{max_retries} after Ollama error; "
                    f"waiting {delay}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)

        raise RuntimeError("Unreachable")


class OpenAIInference:
    """明示指定時だけ利用するOpenAI Responses APIクライアント。"""

    SYSTEM_PROMPT = OllamaInference.SYSTEM_PROMPT

    def __init__(self, config: Config, client: Any = None):
        if not config.openai_api_key:
            raise ProviderConfigError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai or "
                "--provider openai is used."
            )

        if client is not None:
            self.client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderConfigError(
                    "OpenAI provider requires the optional dependency. "
                    "Install it with: pip install -r requirements-cloud.txt"
                ) from exc
            self.client = OpenAI(api_key=config.openai_api_key)

        self.config = config

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.responses.create(
                    model=self.config.model,
                    instructions=self.SYSTEM_PROMPT,
                    input=prompt,
                    max_output_tokens=2048,
                    temperature=0.8,
                    store=False,
                )
                content = str(getattr(response, "output_text", "")).strip()
                if not content:
                    raise RuntimeError("OpenAI returned an empty response")
                return content
            except Exception:
                if attempt == max_retries - 1:
                    raise
                delay = 2**attempt
                print(
                    f"Retry {attempt + 1}/{max_retries} after OpenAI error; "
                    f"waiting {delay}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)

        raise RuntimeError("Unreachable")


def create_text_generator(config: Config) -> TextGenerator:
    if config.provider == "ollama":
        return OllamaInference(config)
    if config.provider == "openai":
        return OpenAIInference(config)
    raise ProviderConfigError(
        f"Unsupported provider: {config.provider}. Use ollama or openai."
    )


# =============================================================================
# Local CSV storage
# =============================================================================


class LocalStorage:
    """ローカルCSVストレージ。Google Sheetsは使用しない。"""

    SEED_HEADERS = {
        "age": "age",
        "gender": "gender",
        "species": "species",
        "ability": "ability",
        "wants": "wants",
        "role": "role",
    }
    OUTPUT_HEADERS = [
        "name",
        "profile",
        "catchphrase",
        "image_prompt",
        "concept",
        "age",
        "gender",
        "species",
        "ability",
        "wants",
        "role",
        "image_path",
        "image_seed",
    ]

    DEFAULT_SEEDS = {
        "age": [
            "Child",
            "Teenager",
            "Young Adult",
            "Middle-aged",
            "Elder",
            "Ageless",
        ],
        "gender": [
            "Male",
            "Female",
            "Non-binary",
            "Genderfluid",
            "Androgynous",
        ],
        "species": [
            "Human",
            "Elf",
            "Dwarf",
            "Demon",
            "Angel",
            "Dragon-kin",
            "Beastfolk",
            "Undead",
            "Cyborg",
            "Alien",
        ],
        "ability": [
            "Can manipulate fire at will",
            "Has the power to read minds",
            "Possesses superhuman strength",
            "Can turn invisible",
            "Has the ability to heal others",
            "Can control time for brief moments",
            "Possesses perfect memory",
            "Can communicate with animals",
        ],
        "wants": [
            "I want to find my lost family",
            "I want to become the strongest warrior",
            "I want to discover the truth about my past",
            "I want to protect the innocent",
            "I want to achieve immortality",
            "I want to find true love",
            "I want to conquer the world",
            "I want to bring peace to all nations",
        ],
        "role": [
            "Warrior. A skilled fighter dedicated to protecting others",
            "Mage. A wielder of arcane arts seeking forbidden knowledge",
            "Healer. A compassionate soul devoted to saving lives",
            "Assassin. A shadow operative with deadly precision",
            "Scholar. A seeker of ancient wisdom and lost lore",
            "Merchant. A cunning trader with connections everywhere",
            "Noble. A person of high birth with political influence",
            "Wanderer. A mysterious traveler with no fixed home",
        ],
    }

    def __init__(self, config: Config):
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.seed_files = {
            key: self.data_dir / f"seed_{key}.csv" for key in self.SEED_HEADERS
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = self.data_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.run_dir / "output.csv"
        self.errors_file = self.run_dir / "errors.jsonl"

        self._ensure_seed_data()
        self._seed_values: Dict[str, List[str]] = {
            key: self._read_seed_values(path)
            for key, path in self.seed_files.items()
        }
        self._init_output_file()

    def _ensure_seed_data(self) -> None:
        for key, seed_file in self.seed_files.items():
            if seed_file.exists():
                continue
            with open(seed_file, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([self.SEED_HEADERS[key]])
                writer.writerows([[value] for value in self.DEFAULT_SEEDS[key]])

    def _read_seed_values(self, seed_file: Path) -> List[str]:
        key = seed_file.stem.removeprefix("seed_")
        expected_header = self.SEED_HEADERS.get(key)
        if expected_header is None:
            raise SeedDataError(f"Unknown seed file: {seed_file}")

        try:
            with open(seed_file, "r", newline="", encoding="utf-8-sig") as file:
                rows = list(csv.reader(file))
        except OSError as exc:
            raise SeedDataError(f"Cannot read seed file: {seed_file}: {exc}") from exc

        if not rows:
            raise SeedDataError(
                f"Seed file is empty: {seed_file}. "
                f"The first row must be '{expected_header}'."
            )

        if rows[0] != [expected_header]:
            raise SeedDataError(
                f"Invalid header in {seed_file}: expected one column named "
                f"'{expected_header}'."
            )

        values: List[str] = []
        for row_number, row in enumerate(rows[1:], start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 1:
                raise SeedDataError(
                    f"Invalid row {row_number} in {seed_file}: "
                    "each data row must contain exactly one column."
                )
            value = row[0].strip()
            if value:
                values.append(value)

        if not values:
            raise SeedDataError(
                f"Seed file has no usable values: {seed_file}. "
                "Add at least one non-empty value below the header."
            )
        return values

    def get_random_attribute(self, attr_type: str) -> str:
        if attr_type not in self._seed_values:
            raise SeedDataError(f"Unknown seed attribute: {attr_type}")
        values = self._seed_values[attr_type]
        if not values:
            raise SeedDataError(f"Seed attribute has no usable values: {attr_type}")
        return random.choice(values)

    def _init_output_file(self) -> None:
        with open(self.output_file, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(self.OUTPUT_HEADERS)

    def append_output(self, row: List[Any]) -> None:
        if len(row) != len(self.OUTPUT_HEADERS):
            raise ValueError(
                f"Output row has {len(row)} columns; "
                f"expected {len(self.OUTPUT_HEADERS)}."
            )
        with open(self.output_file, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(row)

    def append_seed(self, attr_type: str, value: str) -> None:
        if attr_type not in self.seed_files:
            raise SeedDataError(f"Unknown seed attribute: {attr_type}")
        normalized = value.strip()
        if not normalized:
            raise SeedDataError(
                f"Cannot append an empty value to seed_{attr_type}.csv"
            )
        with open(self.seed_files[attr_type], "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([normalized])
        self._seed_values[attr_type].append(normalized)

    def commit_character(
        self, row: List[Any], seed_updates: Dict[str, str]
    ) -> None:
        """生成済みの1キャラクターを保存する。生成途中では呼び出さない。"""
        self.append_output(row)
        for attr_type in ("ability", "wants", "role"):
            self.append_seed(attr_type, seed_updates[attr_type])

    def record_error(
        self,
        iteration: int,
        stage: str,
        error: Exception,
    ) -> None:
        payload = {
            "iteration": iteration,
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        with open(self.errors_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def relative_path(self, path: Path) -> str:
        return path.relative_to(self.data_dir).as_posix()


# =============================================================================
# Prompts
# =============================================================================


class Prompts:
    """キャラクター生成用プロンプト。"""

    @staticmethod
    def character_concept(physical: str, role: str, ability: str, wants: str) -> str:
        return f"""以下のキャラクター属性を、重要な要素を損なわないように要約し、英文1段落で出力してください。

## 属性
- 身体的特徴: {physical}
- 役割: {role}
- 能力: {ability}
- 願望: {wants}

## ルール
- 英語で出力
- 1段落のみ
- 説明や補足は不要

## 出力"""

    @staticmethod
    def name(concept: str) -> str:
        return f"""以下のキャラクター設定にふさわしい人名を1つ生成してください。

## キャラクター設定
{concept}

## ルール
- 名前のみを出力（説明不要）
- 英語表記
- 国籍・文化・架空言語の名前も可
- 1行のみ

## 出力例
Kain Astralion
Yuichi Aihara

## 出力"""

    @staticmethod
    def profile(concept: str) -> str:
        return f"""以下のキャラクター設定を日本語で説明してください。

## キャラクター設定
{concept}

## ルール
- 日本語で出力
- 性別不明・Theyの場合は「彼は」を使用
- 1段落のみ

## 出力例
彼はプリティーンのノンバイナリー半人半神で、デジタル栄養コンサルタントとして活動しています。

## 出力"""

    @staticmethod
    def catchphrase(concept: str) -> str:
        return f"""以下のキャラクターの意思を表す印象的な決め台詞を生成してください。

## キャラクター設定
{concept}

## ルール
- 日本語で出力
- キャラクターにふさわしい口調
- 一人称から始める
- 1文のみ

## 出力例
私は、歴史の断片を手に取り、宇宙の隅々に宿る感情を感じ取るよ。

## 出力"""

    @staticmethod
    def new_ability(concept: str) -> str:
        return f"""以下のキャラクターと対になるキャラクターが持つ特殊能力を1つ生成してください。

## 元キャラクター
{concept}

## ルール
- 英語で出力
- 能力名と説明を1文で
- 1つのみ

## 出力例
Has the ability to materialize memories: Can share past events with others.

## 出力"""

    @staticmethod
    def new_wants(concept: str) -> str:
        return f"""以下のキャラクターと対になるキャラクターの切実な願望を1つ生成してください。

## 元キャラクター
{concept}

## ルール
- 英語で出力
- "I want to..." の形式
- 1文のみ

## 出力例
I want to establish a new human settlement in space.

## 出力"""

    @staticmethod
    def new_role(concept: str) -> str:
        return f"""以下のキャラクターと対になるキャラクターのユニークな役割を1つ生成してください。

## 元キャラクター
{concept}

## ルール
- 英語で出力
- 役割名と説明
- 1つのみ

## 出力例
Swordsman. Skilled in the art of swordsmanship with a strong sense of duty.

## 出力"""


# =============================================================================
# Image prompt and generation orchestration
# =============================================================================


def generate_image_prompt(
    concept: str,
    profile_id: str = "generic-sdxl",
    age: str = "",
    gender: str = "",
    species: str = "",
    ability: str = "",
    role: str = "",
    profiles_path: str = "./config/comfyui/model_profiles.json",
) -> str:
    """画像モデルのプロファイルに合わせてpromptを作る。"""
    profile = get_image_model_profile(profile_id, Path(profiles_path))
    return build_image_prompt(
        profile,
        concept=concept,
        age=age,
        gender=gender,
        species=species,
        ability=ability,
        role=role,
    )


def _stage(stage: str, function: Callable[[], Any]) -> Any:
    try:
        return function()
    except StageError:
        raise
    except Exception as exc:
        raise StageError(stage, exc) from exc


def _safe_filename(value: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value.strip()
    )
    return normalized.strip("._") or "character"


def _create_image_generator(config: Config) -> Any:
    from comfyui_image_gen import ComfyUIImageGenerator

    return ComfyUIImageGenerator(
        base_url=config.comfyui_url,
        workflow_path=Path(config.comfyui_workflow_path),
        checkpoint_name=config.comfyui_checkpoint_name,
        timeout_seconds=config.comfyui_timeout_seconds,
        poll_interval_seconds=config.comfyui_poll_interval_seconds,
        width=config.comfyui_width,
        height=config.comfyui_height,
        steps=config.comfyui_steps,
        cfg=config.comfyui_cfg,
        sampler=config.comfyui_sampler,
        scheduler=config.comfyui_scheduler,
        negative_prompt=config.comfyui_negative_prompt,
        clip_skip=config.comfyui_clip_skip,
    )


def generate_characters(
    config: Config,
    text_generator: Optional[TextGenerator] = None,
    storage: Optional[LocalStorage] = None,
    image_generator: Any = None,
) -> Path:
    """指定回数のキャラクターを生成し、output.csvのパスを返す。"""
    llm = text_generator or create_text_generator(config)
    local_storage = storage or LocalStorage(config)

    if config.generate_images:
        image_generator = image_generator or _create_image_generator(config)
        _stage("image", image_generator.check_connection)

    print(f"Starting generation with provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Data directory: {config.data_dir}")
    print(f"Run directory: {local_storage.run_dir}")
    print(f"Output file: {local_storage.output_file}")

    for index in range(config.num_iterations):
        iteration = index + 1
        print(f"\n[{iteration}/{config.num_iterations}] Generating character...")
        try:
            age = local_storage.get_random_attribute("age")
            gender = local_storage.get_random_attribute("gender")
            species = local_storage.get_random_attribute("species")
            physical = f"{age} {gender} {species}"
            ability = local_storage.get_random_attribute("ability")
            wants = local_storage.get_random_attribute("wants")
            role = local_storage.get_random_attribute("role")

            concept = _stage(
                "character_concept",
                lambda: llm.generate(
                    Prompts.character_concept(physical, role, ability, wants)
                ),
            )
            name = _stage("name", lambda: llm.generate(Prompts.name(concept)))
            profile = _stage(
                "profile", lambda: llm.generate(Prompts.profile(concept))
            )
            catchphrase = _stage(
                "catchphrase", lambda: llm.generate(Prompts.catchphrase(concept))
            )
            image_prompt = generate_image_prompt(
                concept,
                profile_id=config.comfyui_model_profile,
                age=age,
                gender=gender,
                species=species,
                ability=ability,
                role=role,
                profiles_path=config.comfyui_model_profiles_path,
            )

            new_ability = _stage(
                "new_ability", lambda: llm.generate(Prompts.new_ability(concept))
            )
            new_wants = _stage(
                "new_wants", lambda: llm.generate(Prompts.new_wants(concept))
            )
            new_role = _stage(
                "new_role", lambda: llm.generate(Prompts.new_role(concept))
            )

            image_path = ""
            image_seed = ""
            if config.generate_images:
                def generate_image() -> Any:
                    if config.memory_guard_enabled:
                        try:
                            memory = ensure_available(
                                config.minimum_available_memory_percent
                            )
                        except MemoryBudgetExceeded as exc:
                            raise ComfyUIError(str(exc)) from exc
                        if memory.available_percent is not None:
                            print(
                                "  Available memory before image: "
                                f"{memory.available_percent:.1f}%"
                            )
                    return image_generator.generate(
                        image_prompt,
                        local_storage.run_dir / "images",
                        f"{iteration:03d}_{_safe_filename(name)}",
                    )

                image_result = _stage(
                    "image",
                    generate_image,
                )
                image_path = local_storage.relative_path(image_result.path)
                image_seed = str(image_result.seed)

            row = [
                name,
                profile,
                catchphrase,
                image_prompt,
                concept,
                age,
                gender,
                species,
                ability,
                wants,
                role,
                image_path,
                image_seed,
            ]
            _stage(
                "persistence",
                lambda: local_storage.commit_character(
                    row,
                    {
                        "ability": new_ability,
                        "wants": new_wants,
                        "role": new_role,
                    },
                ),
            )
            print(f"  Name: {name}")
        except StageError as exc:
            local_storage.record_error(iteration, exc.stage, exc.cause)
            print(
                f"ERROR iteration={iteration} stage={exc.stage}: {exc.cause}\n"
                f"Results kept at: {local_storage.run_dir}",
                file=sys.stderr,
            )
            raise
        except Exception as exc:
            local_storage.record_error(iteration, "input", exc)
            print(
                f"ERROR iteration={iteration} stage=input: {exc}\n"
                f"Results kept at: {local_storage.run_dir}",
                file=sys.stderr,
            )
            raise StageError("input", exc) from exc

    print("\n処理が完了しました。")
    print(f"実行ディレクトリ: {local_storage.run_dir}")
    print(f"出力ファイル: {local_storage.output_file}")
    return local_storage.output_file


def main(
    iterations: Optional[int] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    generate_images: Optional[bool] = None,
) -> int:
    try:
        config = Config.from_env()
        overrides: Dict[str, Any] = {}
        if iterations is not None:
            overrides["num_iterations"] = iterations
        if model is not None:
            overrides["model"] = model
        if provider is not None:
            overrides["provider"] = provider
            if provider == "openai" and model is None:
                overrides["model"] = config.openai_model
            if provider == "ollama" and model is None:
                overrides["model"] = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        if generate_images is not None:
            overrides["generate_images"] = generate_images
        config = replace(config, **overrides)
        if config.num_iterations < 1:
            raise ValueError("iterations must be greater than 0")

        generate_characters(config)
        return 0
    except (
        OllamaConnectionError,
        OllamaModelNotFoundError,
        ProviderConfigError,
        SeedDataError,
        ComfyUIError,
        StageError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="100 Times AI Heroes - local character generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
既定の実行:
  python ollama_hero_gen.py --iterations 1

ローカル画像生成:
  python ollama_hero_gen.py --iterations 1 --generate-images

任意のクラウドLLM:
  python ollama_hero_gen.py --provider openai --iterations 1
""",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=None, help="生成するキャラクター数"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        metavar="MODEL",
        help="選択したLLMプロバイダーで使うモデル名",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default=None,
        help="LLMプロバイダー（既定: .envのLLM_PROVIDER、未設定時ollama）",
    )
    parser.add_argument(
        "--generate-images",
        action="store_true",
        help="localhost上のComfyUIで画像も生成する",
    )
    args = parser.parse_args()

    raise SystemExit(
        main(
            iterations=args.iterations,
            model=args.model,
            provider=args.provider,
            generate_images=True if args.generate_images else None,
        )
    )
