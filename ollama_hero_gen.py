"""
100 Times AI Heroes - Ollama版キャラクター生成システム（ローカル完結版）

Usage:
    python ollama_hero_gen.py
    python ollama_hero_gen.py --iterations 10
    python ollama_hero_gen.py --model gpt-oss:20b-q4_K_M  # 量子化版
    python ollama_hero_gen.py --model gpt-oss:120b          # 高性能版

Available models:
    gpt-oss:20b          標準（デフォルト）: 12GB VRAM、バランス重視
    gpt-oss:20b-q4_K_M   量子化版: 低メモリ環境向け（約6GB）
    gpt-oss:120b         高性能版: 高スペックマシン向け
"""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama
from dotenv import load_dotenv


class SeedDataError(Exception):
    """seed CSV の不備を、原因・ファイル・修正方法とともに報告する例外。"""


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class Config:
    """不変の設定オブジェクト"""

    model: str
    host: str
    data_dir: str
    num_iterations: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            data_dir=os.getenv("DATA_DIR", "./data"),
        )


# =============================================================================
# Ollama Inference
# =============================================================================


class OllamaInference:
    """Ollama推論クライアント"""

    SYSTEM_PROMPT = (
        "人間の仕事を助ける優秀なAIアシスタントとして、"
        "指示に従い、必要な情報のみを端的に出力します。"
    )

    def __init__(self, config: Config):
        self.config = config
        self.client = ollama.Client(host=config.host, timeout=120)
        self._ensure_model_available()

    def _ensure_model_available(self) -> None:
        """モデルの存在確認、なければpull"""
        try:
            models = self.client.list()
            model_list = models.get("models", [])

            # モデルリストの形式に対応（新旧両方）
            available = []
            for m in model_list:
                if hasattr(m, "model"):
                    available.append(m.model)
                elif isinstance(m, dict) and "name" in m:
                    available.append(m["name"])

            model_name = self.config.model

            if not any(model_name in name for name in available):
                print(f"Pulling model: {model_name}")
                self.client.pull(model_name)
        except Exception as e:
            raise ConnectionError(
                f"Ollama server not running at {self.config.host}. "
                f"Start with: ollama serve"
            ) from e

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """リトライ付き推論"""
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
                return response["message"]["content"].strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = 2**attempt
                print(f"Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(delay)

        raise RuntimeError("Unreachable")


# =============================================================================
# Local Storage (CSV)
# =============================================================================


class LocalStorage:
    """ローカルCSVストレージ"""

    def __init__(self, config: Config):
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # シードデータファイル（全実行で共有）
        self.seed_files = {
            "age": self.data_dir / "seed_age.csv",
            "gender": self.data_dir / "seed_gender.csv",
            "species": self.data_dir / "seed_species.csv",
            "ability": self.data_dir / "seed_ability.csv",
            "wants": self.data_dir / "seed_wants.csv",
            "role": self.data_dir / "seed_role.csv",
        }

        # 実行ごとに固有のディレクトリを作成して出力を保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = self.data_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.run_dir / "output.csv"

        # シードデータ初期化
        self._ensure_seed_data()

        # 出力ファイルヘッダー作成
        self._init_output_file()

    def _ensure_seed_data(self) -> None:
        """シードデータがなければ初期データを作成"""
        default_seeds = {
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

        for key, seed_file in self.seed_files.items():
            if not seed_file.exists():
                print(f"Creating seed file: {seed_file}")
                with open(seed_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([key])  # ヘッダー
                    for value in default_seeds[key]:
                        writer.writerow([value])

        # 全 seed ファイルを検証（存在しない場合は上で作成済み）
        for key, seed_file in self.seed_files.items():
            self._validate_seed_file(key, seed_file)

    def _validate_seed_file(self, key: str, seed_file: Path) -> None:
        """seed CSV の形式を検証し、不備があれば SeedDataError を投げる。

        仕様: UTF-8・1列のみ・1行目が必須ヘッダー・データ行は空でない。
        """
        if not seed_file.exists():
            raise SeedDataError(
                f"SeedDataError: seed file not found: {seed_file} — "
                f"expected header '{key}'. Delete the file or restore it, "
                f"then rerun to regenerate from defaults."
            )
        with open(seed_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            raise SeedDataError(
                f"SeedDataError: seed file is empty: {seed_file} — "
                f"expected header '{key}' followed by at least one value."
            )
        header = rows[0]
        if not header or header[0].strip() != key:
            raise SeedDataError(
                f"SeedDataError: invalid header in {seed_file} — "
                f"expected '{key}', got {header[:1]!r}. Fix the first line."
            )
        valid_rows = 0
        for row in rows[1:]:
            if len(row) > 1:
                raise SeedDataError(
                    f"SeedDataError: multi-column row in {seed_file}: {row!r} — "
                    f"each data row must be a single column. Quote values "
                    f"containing commas."
                )
            if row and row[0].strip():
                valid_rows += 1
        if valid_rows == 0:
            raise SeedDataError(
                f"SeedDataError: no valid values in {seed_file} — "
                f"add at least one non-empty value under the '{key}' header."
            )

    def _init_output_file(self) -> None:
        """出力ファイルのヘッダーを書き込む"""
        headers = [
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
        ]
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _read_seed_values(self, seed_file: Path) -> list:
        """シードファイルから値を読み込む（検証済みの形式を前提とする）"""
        with open(seed_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # ヘッダースキップ
            return [row[0].strip() for row in reader if row and row[0].strip()]

    def get_random_attribute(self, attr_type: str) -> str:
        """ランダムに属性を取得"""
        seed_file = self.seed_files[attr_type]
        self._validate_seed_file(attr_type, seed_file)
        values = self._read_seed_values(seed_file)
        if not values:
            raise SeedDataError(
                f"SeedDataError: no values available in {seed_file} — "
                f"add at least one non-empty value under the '{attr_type}' header."
            )
        return random.choice(values)

    def append_output(self, row: list) -> None:
        """出力ファイルに行を追加"""
        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def append_seed(self, attr_type: str, value: str) -> None:
        """シードデータに新しい値を追加（空値は SeedDataError）"""
        if value is None or not str(value).strip():
            raise SeedDataError(
                f"SeedDataError: cannot append an empty value to seed '{attr_type}' — "
                f"provide a non-empty string."
            )
        seed_file = self.seed_files[attr_type]
        with open(seed_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([str(value).strip()])


# =============================================================================
# Prompts
# =============================================================================


class Prompts:
    """Ollama最適化プロンプト"""

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
- 性別不明・They の場合は「彼は」を使用
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
# Image Prompt Generation
# =============================================================================


def generate_image_prompt(concept: str) -> str:
    """画像プロンプト生成（固定テンプレート）"""
    subject = (
        "The full-length character illustration from video games, "
        "likely from role-playing games(JRPG) or fighting games."
    )
    angle = "A camera angle that captures the entire body evenly from waist height."
    pose = (
        "Standing upright and looking straight ahead, his pose visually conveys "
        "role, personality, attitude, ability, cultural background and physical attractiveness."
    )
    background = "white background."
    artstyle = (
        "The art style combines delicate hand-drawn lines with exaggerated "
        "expressions influenced by Japanese manga and anime."
    )

    return f"{subject} {angle} {pose} {background} {concept}, {artstyle}"


# =============================================================================
# Main
# =============================================================================


def main(iterations: Optional[int] = None, model: Optional[str] = None) -> None:
    config = Config.from_env()

    if iterations is not None or model is not None:
        config = Config(
            model=model if model is not None else config.model,
            host=config.host,
            data_dir=config.data_dir,
            num_iterations=iterations if iterations is not None else config.num_iterations,
        )

    print(f"Starting generation with model: {config.model}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Data directory: {config.data_dir}")

    llm = OllamaInference(config)
    storage = LocalStorage(config)

    print(f"Run directory: {storage.run_dir}")
    print(f"Output file: {storage.output_file}")

    for i in range(config.num_iterations):
        print(f"\n[{i + 1}/{config.num_iterations}] Generating character...")

        # 属性取得
        age = storage.get_random_attribute("age")
        gender = storage.get_random_attribute("gender")
        species = storage.get_random_attribute("species")
        physical = f"{age} {gender} {species}"
        ability = storage.get_random_attribute("ability")
        wants = storage.get_random_attribute("wants")
        role = storage.get_random_attribute("role")

        # 推論
        concept = llm.generate(Prompts.character_concept(physical, role, ability, wants))
        name = llm.generate(Prompts.name(concept))
        profile = llm.generate(Prompts.profile(concept))
        catchphrase = llm.generate(Prompts.catchphrase(concept))
        image_prompt = generate_image_prompt(concept)

        # 出力
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
        ]
        storage.append_output(row)

        # 対キャラの属性生成
        new_ability = llm.generate(Prompts.new_ability(concept))
        new_wants = llm.generate(Prompts.new_wants(concept))
        new_role = llm.generate(Prompts.new_role(concept))

        storage.append_seed("ability", new_ability)
        storage.append_seed("wants", new_wants)
        storage.append_seed("role", new_role)

        print(f"  Name: {name}")

    print(f"\n処理が完了しました。")
    print(f"実行ディレクトリ: {storage.run_dir}")
    print(f"出力ファイル: {storage.output_file}")


if __name__ == "__main__":
    import argparse

    # 利用可能なモデル一覧
    AVAILABLE_MODELS = [
        "gpt-oss:20b",           # 標準（デフォルト）
        "gpt-oss:20b-q4_K_M",    # 量子化版（低メモリ環境向け）
        "gpt-oss:120b",          # 高性能版（高スペックマシン向け）
    ]

    parser = argparse.ArgumentParser(
        description="100 Times AI Heroes - Ollama版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
モデル選択ガイド:
  gpt-oss:20b          標準モデル（デフォルト）。M4 MacBook Pro等の推奨環境向け
  gpt-oss:20b-q4_K_M   量子化版。メモリが限られている環境向け（約6GB）
  gpt-oss:120b         高性能モデル。128GB以上のメモリを持つ高スペックマシン向け

出力先:
  各実行の結果は data/run_YYYYMMDD_HHMMSS/output.csv に保存されます。
  複数回の実行でもディレクトリが分かれるため結果が上書きされません。
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
        choices=AVAILABLE_MODELS,
        metavar="MODEL",
        help=(
            f"使用するOllamaモデル。選択肢: {', '.join(AVAILABLE_MODELS)} "
            f"（デフォルト: gpt-oss:20b）"
        ),
    )
    args = parser.parse_args()

    main(iterations=args.iterations, model=args.model)
