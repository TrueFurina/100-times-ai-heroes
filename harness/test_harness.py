"""ローカル版の自動テスト。

通常のテストでは、Ollama、ComfyUI、OpenAIの実サービスへ接続しない。
実サービスを使う確認は、別途手動のE2Eとして行う。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from comfyui_image_gen import (  # noqa: E402
    ComfyUIConfigurationError,
    ComfyUIImageGenerator,
    GeneratedImage,
)
from image_model_profiles import (  # noqa: E402
    build_image_prompt,
    get_image_model_profile,
)
from memory_safety import MemoryBudgetExceeded  # noqa: E402
import ollama_hero_gen as app  # noqa: E402
from ollama_hero_gen import (  # noqa: E402
    Config,
    LocalStorage,
    OllamaInference,
    OllamaModelNotFoundError,
    OpenAIInference,
    Prompts,
    SeedDataError,
    StageError,
    generate_characters,
    generate_image_prompt,
)


def make_config(tmp_path: Path, **overrides) -> Config:
    values = {
        "model": "test-model",
        "host": "http://127.0.0.1:11434",
        "data_dir": str(tmp_path),
        "num_iterations": 1,
    }
    values.update(overrides)
    return Config(**values)


class FakeTextGenerator:
    def __init__(self, responses=None, fail_at=None):
        self.responses = list(
            responses
            or [
                "A concept",
                "Test Name",
                "プロフィール",
                "私は進む。",
                "New ability",
                "I want to win.",
                "New role",
            ]
        )
        self.fail_at = fail_at
        self.calls = 0

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise TimeoutError("fake timeout")
        return self.responses.pop(0)


class FakeImageGenerator:
    def __init__(self):
        self.connection_checks = 0
        self.calls = []

    def check_connection(self):
        self.connection_checks += 1

    def generate(self, prompt, output_dir: Path, filename_stem: str):
        self.calls.append(prompt)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{filename_stem}.png"
        path.write_bytes(b"fake png")
        return GeneratedImage(path=path, seed=12345)


def test_config_defaults_and_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_MODEL", "env-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    config = Config.from_env()

    assert config.model == "env-model"
    assert config.host == "http://127.0.0.1:9999"
    assert config.data_dir == str(tmp_path / "data")
    assert config.provider == "ollama"
    assert config.generate_images is False


def test_image_model_profiles_cover_comparison_candidates(tmp_path):
    profile_ids = [
        "animagine-xl-4.0-opt",
        "illustrious-xl-v2",
        "pony-v6-xl",
        "noobai-xl-1.1",
    ]

    profiles = [
        get_image_model_profile(profile_id, tmp_path / "missing.json")
        for profile_id in profile_ids
    ]

    assert len({profile.checkpoint_name for profile in profiles}) == 4
    assert all(profile.width == 832 for profile in profiles)
    assert all(profile.height == 1216 for profile in profiles)


def test_model_profile_prompt_uses_project_attributes(tmp_path):
    profile = get_image_model_profile("animagine-xl-4.0-opt", tmp_path / "missing.json")

    prompt = build_image_prompt(
        profile,
        concept="A guardian who turns noise into music.",
        age="young adult",
        gender="female",
        species="human",
        ability="Can transform digital noise into music",
        role="Sound Cartographer",
    )

    assert prompt.startswith("1girl, solo, full body")
    assert "Sound Cartographer" in prompt
    assert "A guardian who turns noise into music." in prompt


def test_ollama_model_missing_does_not_pull(tmp_path):
    client = MagicMock()
    client.list.return_value = {"models": []}
    config = make_config(tmp_path)

    with pytest.raises(OllamaModelNotFoundError, match="ollama pull test-model"):
        OllamaInference(config, client=client)

    client.pull.assert_not_called()


def test_ollama_model_list_object_is_supported(tmp_path):
    client = MagicMock()
    client.list.return_value = SimpleNamespace(
        models=[SimpleNamespace(model="test-model")]
    )

    inference = OllamaInference(make_config(tmp_path), client=client)

    assert inference.config.model == "test-model"


def test_ollama_inference_reads_dict_response(tmp_path):
    client = MagicMock()
    client.list.return_value = {"models": [{"name": "test-model"}]}
    client.chat.return_value = {"message": {"content": " hello "}}

    result = OllamaInference(make_config(tmp_path), client=client).generate("prompt")

    assert result == "hello"
    client.chat.assert_called_once()


def test_local_storage_creates_and_validates_seed_files(tmp_path):
    storage = LocalStorage(make_config(tmp_path))

    assert set(storage.seed_files) == {
        "age",
        "gender",
        "species",
        "ability",
        "wants",
        "role",
    }
    assert storage.get_random_attribute("age")
    assert storage.output_file.exists()

    with storage.output_file.open(newline="", encoding="utf-8") as file:
        assert next(csv.reader(file)) == LocalStorage.OUTPUT_HEADERS


def test_seed_header_error(tmp_path):
    (tmp_path / "seed_age.csv").write_text("wrong\nvalue\n", encoding="utf-8")

    with pytest.raises(SeedDataError, match="seed_age.csv"):
        LocalStorage(make_config(tmp_path))


def test_seed_empty_error(tmp_path):
    (tmp_path / "seed_age.csv").write_text("age\n\n", encoding="utf-8")

    with pytest.raises(SeedDataError, match="no usable values"):
        LocalStorage(make_config(tmp_path))


def test_seed_multiple_columns_error(tmp_path):
    (tmp_path / "seed_age.csv").write_text("age\nchild,extra\n", encoding="utf-8")

    with pytest.raises(SeedDataError, match="exactly one column"):
        LocalStorage(make_config(tmp_path))


def test_seed_append_normalizes_value(tmp_path):
    storage = LocalStorage(make_config(tmp_path))

    storage.append_seed("ability", "  A new ability  ")

    assert storage._seed_values["ability"][-1] == "A new ability"
    with pytest.raises(SeedDataError):
        storage.append_seed("ability", "   ")


def test_generation_writes_csv_and_expands_seeds(tmp_path):
    config = make_config(tmp_path)
    storage = LocalStorage(config)
    text_generator = FakeTextGenerator()

    output_file = generate_characters(
        config,
        text_generator=text_generator,
        storage=storage,
    )

    with output_file.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    assert len(rows) == 2
    assert rows[0] == LocalStorage.OUTPUT_HEADERS
    assert len(rows[1]) == len(LocalStorage.OUTPUT_HEADERS)
    assert rows[1][0] == "Test Name"
    assert rows[1][-2:] == ["", ""]
    assert "New ability" in storage._seed_values["ability"]
    assert "I want to win." in storage._seed_values["wants"]
    assert "New role" in storage._seed_values["role"]


def test_generation_loop_writes_requested_number(tmp_path):
    responses = []
    for index in range(2):
        responses.extend(
            [
                f"Concept {index}",
                f"Name {index}",
                f"Profile {index}",
                f"Catchphrase {index}",
                f"Ability {index}",
                f"Want {index}",
                f"Role {index}",
            ]
        )
    config = make_config(tmp_path, num_iterations=2)
    storage = LocalStorage(config)
    output_file = generate_characters(
        config,
        text_generator=FakeTextGenerator(responses=responses),
        storage=storage,
    )

    with output_file.open(newline="", encoding="utf-8") as file:
        assert len(list(csv.reader(file))) == 3


def test_generation_failure_keeps_previous_rows_and_records_error(tmp_path):
    config = make_config(tmp_path, num_iterations=2)
    storage = LocalStorage(config)
    text_generator = FakeTextGenerator(
        responses=[
            "Concept 0",
            "Name 0",
            "Profile 0",
            "Catchphrase 0",
            "Ability 0",
            "Want 0",
            "Role 0",
        ],
        fail_at=8,
    )

    with pytest.raises(StageError) as error:
        generate_characters(config, text_generator=text_generator, storage=storage)

    assert error.value.stage == "character_concept"
    with storage.output_file.open(newline="", encoding="utf-8") as file:
        assert len(list(csv.reader(file))) == 2
    assert not storage.errors_file.exists() or storage.errors_file.read_text(
        encoding="utf-8"
    )
    event = json.loads(storage.errors_file.read_text(encoding="utf-8").splitlines()[0])
    assert event["iteration"] == 2
    assert event["stage"] == "character_concept"
    assert event["error_type"] == "TimeoutError"


def test_image_generation_is_opt_in(tmp_path):
    config = make_config(tmp_path)
    storage = LocalStorage(config)
    image_generator = MagicMock()

    generate_characters(
        config,
        text_generator=FakeTextGenerator(),
        storage=storage,
        image_generator=image_generator,
    )

    image_generator.check_connection.assert_not_called()
    image_generator.generate.assert_not_called()


def test_image_generation_writes_path_and_seed(tmp_path):
    config = make_config(tmp_path, generate_images=True)
    storage = LocalStorage(config)
    image_generator = FakeImageGenerator()

    output_file = generate_characters(
        config,
        text_generator=FakeTextGenerator(),
        storage=storage,
        image_generator=image_generator,
    )

    assert image_generator.connection_checks == 1
    assert len(image_generator.calls) == 1
    with output_file.open(newline="", encoding="utf-8") as file:
        row = list(csv.reader(file))[1]
    assert row[-2].endswith("/images/001_Test_Name.png")
    assert row[-1] == "12345"
    assert (tmp_path / row[-2]).exists()


def test_memory_guard_records_image_stage_and_stops(tmp_path, monkeypatch):
    def reject_generation(_minimum_percent):
        raise MemoryBudgetExceeded("memory budget exceeded")

    monkeypatch.setattr(app, "ensure_available", reject_generation)
    config = make_config(tmp_path, generate_images=True)
    storage = LocalStorage(config)

    with pytest.raises(StageError) as error:
        generate_characters(
            config,
            text_generator=FakeTextGenerator(),
            storage=storage,
            image_generator=FakeImageGenerator(),
        )

    assert error.value.stage == "image"
    event = json.loads(storage.errors_file.read_text(encoding="utf-8").splitlines()[0])
    assert event["stage"] == "image"


class FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self):
        return self.payload

    def close(self):
        pass


def test_comfyui_api_queue_history_and_download(tmp_path):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "4": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
                "5": {"class_type": "KSampler", "inputs": {}},
                "6": {"class_type": "CLIPTextEncode", "inputs": {}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {}},
                "8": {"class_type": "EmptyLatentImage", "inputs": {}},
                "9": {"class_type": "VAEDecode", "inputs": {}},
                "10": {"class_type": "SaveImage", "inputs": {}},
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def opener(request, timeout=None):
        calls.append((request.get_method(), request.full_url))
        if request.full_url.endswith("/system_stats"):
            return FakeHTTPResponse(b"{}")
        if request.full_url.endswith("/prompt"):
            return FakeHTTPResponse(b'{"prompt_id":"abc"}')
        if "/history/abc" in request.full_url:
            return FakeHTTPResponse(
                b'{"abc":{"status":{"status_str":"success"},"outputs":{"10":{"images":[{"filename":"ComfyUI_00001.png","subfolder":"","type":"output"}]}}}}'
            )
        if "/view?" in request.full_url:
            return FakeHTTPResponse(b"png bytes")
        raise AssertionError(request.full_url)

    generator = ComfyUIImageGenerator(
        base_url="http://127.0.0.1:8188",
        workflow_path=workflow_path,
        checkpoint_name="model.safetensors",
        opener=opener,
        sleeper=lambda _: None,
        seed_factory=lambda: 42,
    )
    generator.check_connection()
    result = generator.generate("A hero", tmp_path / "images", "001_Hero")

    assert result.seed == 42
    assert result.path.read_bytes() == b"png bytes"
    assert [method for method, _ in calls] == ["GET", "POST", "GET", "GET"]


def test_comfyui_profile_can_apply_clip_skip(tmp_path):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "4": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
                "5": {"class_type": "KSampler", "inputs": {}},
                "6": {"class_type": "CLIPTextEncode", "inputs": {}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {}},
                "8": {"class_type": "EmptyLatentImage", "inputs": {}},
                "9": {"class_type": "VAEDecode", "inputs": {}},
                "10": {"class_type": "SaveImage", "inputs": {}},
                "11": {"class_type": "CLIPSetLastLayer", "inputs": {}},
            }
        ),
        encoding="utf-8",
    )
    generator = ComfyUIImageGenerator(
        base_url="http://127.0.0.1:8188",
        workflow_path=workflow_path,
        checkpoint_name="pony.safetensors",
        clip_skip=-2,
    )

    workflow = generator._build_workflow("A hero", 42)

    assert workflow["11"]["inputs"]["stop_at_clip_layer"] == -2
    assert workflow["6"]["inputs"]["clip"] == ["11", 0]
    assert workflow["7"]["inputs"]["clip"] == ["11", 0]


def test_comfyui_free_memory_calls_local_endpoint(tmp_path):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "4": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
                "5": {"class_type": "KSampler", "inputs": {}},
                "6": {"class_type": "CLIPTextEncode", "inputs": {}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {}},
                "8": {"class_type": "EmptyLatentImage", "inputs": {}},
                "9": {"class_type": "VAEDecode", "inputs": {}},
                "10": {"class_type": "SaveImage", "inputs": {}},
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def opener(request, timeout=None):
        calls.append((request.get_method(), request.full_url, request.data))
        return FakeHTTPResponse(b"{}")

    generator = ComfyUIImageGenerator(
        base_url="http://127.0.0.1:8188",
        workflow_path=workflow_path,
        checkpoint_name="model.safetensors",
        opener=opener,
    )

    generator.free_memory()

    assert calls[0][0:2] == ("POST", "http://127.0.0.1:8188/free")
    assert json.loads(calls[0][2]) == {
        "unload_models": True,
        "free_memory": True,
    }


def test_comfyui_rejects_non_local_url(tmp_path):
    with pytest.raises(ComfyUIConfigurationError, match="local HTTP server"):
        ComfyUIImageGenerator(
            base_url="https://example.com",
            workflow_path=tmp_path / "workflow.json",
            checkpoint_name="model.safetensors",
        )


def test_openai_provider_uses_responses_api_without_real_network(tmp_path):
    client = MagicMock()
    client.responses.create.return_value = SimpleNamespace(output_text=" answer ")
    config = make_config(
        tmp_path,
        provider="openai",
        model="gpt-4o-mini",
        openai_api_key="test-key",
    )

    result = OpenAIInference(config, client=client).generate("prompt")

    assert result == "answer"
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["store"] is False
    assert kwargs["max_output_tokens"] == 2048


def test_openai_provider_requires_key(tmp_path):
    config = make_config(tmp_path, provider="openai", model="gpt-4o-mini")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIInference(config, client=MagicMock())


def test_prompts_and_image_prompt():
    concept_prompt = Prompts.character_concept("Young Male Human", "Warrior", "Power", "Peace")
    assert "Young Male Human" in concept_prompt
    assert "Warrior" in concept_prompt
    assert "英語" in concept_prompt
    assert "A young warrior" in generate_image_prompt("A young warrior")
