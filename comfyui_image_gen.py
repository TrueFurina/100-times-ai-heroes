"""ComfyUIのローカルAPIを使った画像生成クライアント。"""

from __future__ import annotations

import copy
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class ComfyUIConfigurationError(ValueError):
    """ComfyUIまたはworkflow設定が不正。"""


class ComfyUIConnectionError(ConnectionError):
    """ComfyUIへ接続できない。"""


class ComfyUITimeoutError(TimeoutError):
    """ComfyUIの処理がタイムアウトした。"""


class ComfyUIError(RuntimeError):
    """ComfyUIの処理に失敗した。"""


@dataclass(frozen=True)
class GeneratedImage:
    path: Path
    seed: int


class ComfyUIImageGenerator:
    """ComfyUI API workflowを使って画像を生成する。"""

    REQUIRED_NODES = {
        "4": "CheckpointLoaderSimple",
        "5": "KSampler",
        "6": "CLIPTextEncode",
        "7": "CLIPTextEncode",
        "8": "EmptyLatentImage",
        "9": "VAEDecode",
        "10": "SaveImage",
    }

    def __init__(
        self,
        base_url: str,
        workflow_path: Path,
        checkpoint_name: str,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 1.0,
        width: int = 768,
        height: int = 1152,
        steps: int = 24,
        cfg: float = 7.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        clip_skip: Optional[int] = None,
        negative_prompt: str = (
            "low quality, blurry, distorted hands, extra fingers, cropped, duplicate"
        ),
        opener: Optional[Callable[..., Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        seed_factory: Callable[[], int] = lambda: secrets.randbelow(2**63),
    ):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ComfyUIConfigurationError(
                "COMFYUI_URL must point to a local HTTP server "
                "(127.0.0.1, localhost, or ::1)."
            )
        if checkpoint_name == "CHANGE_ME.safetensors":
            raise ComfyUIConfigurationError(
                "Set COMFYUI_CHECKPOINT_NAME to an installed checkpoint filename."
            )

        self.base_url = base_url.rstrip("/")
        self.workflow_path = Path(workflow_path)
        self.checkpoint_name = checkpoint_name
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.width = width
        self.height = height
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.scheduler = scheduler
        self.clip_skip = clip_skip
        self.negative_prompt = negative_prompt
        self.opener = opener or urllib.request.urlopen
        self.clock = clock
        self.sleeper = sleeper
        self.seed_factory = seed_factory

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request_bytes(
        self,
        request: urllib.request.Request,
        timeout: Optional[float] = None,
    ) -> bytes:
        try:
            response = self.opener(request, timeout=timeout or self.timeout_seconds)
            try:
                return response.read()
            finally:
                close = getattr(response, "close", None)
                if close:
                    close()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            raise ComfyUIConnectionError(
                f"Could not connect to ComfyUI at {self.base_url}: {exc}"
            ) from exc

    def _request_json(
        self,
        request: urllib.request.Request,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        raw = self._request_bytes(request, timeout=timeout)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ComfyUIError("ComfyUI returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ComfyUIError("ComfyUI returned a JSON value instead of an object")
        return payload

    def check_connection(self) -> None:
        """画像生成開始前にlocalhostのComfyUIへ接続できることを確認する。"""
        request = urllib.request.Request(self._url("system_stats"), method="GET")
        self._request_json(request, timeout=min(self.timeout_seconds, 10.0))

    def free_memory(self) -> None:
        """ComfyUIにロード済みモデルの解放を依頼する。"""
        body = json.dumps(
            {"unload_models": True, "free_memory": True}
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url("free"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self._request_bytes(request, timeout=min(self.timeout_seconds, 10.0))

    def _load_workflow(self) -> Dict[str, Any]:
        try:
            with open(self.workflow_path, "r", encoding="utf-8") as file:
                workflow = json.load(file)
        except OSError as exc:
            raise ComfyUIConfigurationError(
                f"Cannot read ComfyUI workflow: {self.workflow_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ComfyUIConfigurationError(
                f"ComfyUI workflow is not valid JSON: {self.workflow_path}"
            ) from exc

        if not isinstance(workflow, dict):
            raise ComfyUIConfigurationError("ComfyUI workflow must be a JSON object")
        for node_id, class_type in self.REQUIRED_NODES.items():
            node = workflow.get(node_id)
            if not isinstance(node, dict) or node.get("class_type") != class_type:
                raise ComfyUIConfigurationError(
                    f"Workflow node {node_id} must have class_type {class_type}"
                )
        return workflow

    def _build_workflow(self, prompt: str, seed: int) -> Dict[str, Any]:
        workflow = copy.deepcopy(self._load_workflow())
        workflow["4"]["inputs"]["ckpt_name"] = self.checkpoint_name
        workflow["6"]["inputs"]["text"] = prompt
        workflow["7"]["inputs"]["text"] = self.negative_prompt
        if self.clip_skip is not None:
            clip_node = workflow.get("11")
            if not isinstance(clip_node, dict) or clip_node.get("class_type") != "CLIPSetLastLayer":
                raise ComfyUIConfigurationError(
                    "The selected image model profile requires workflow node 11 "
                    "with class_type CLIPSetLastLayer."
                )
            clip_node.setdefault("inputs", {})["stop_at_clip_layer"] = self.clip_skip
            workflow["6"]["inputs"]["clip"] = ["11", 0]
            workflow["7"]["inputs"]["clip"] = ["11", 0]
        workflow["5"]["inputs"].update(
            {
                "seed": seed,
                "steps": self.steps,
                "cfg": self.cfg,
                "sampler_name": self.sampler,
                "scheduler": self.scheduler,
            }
        )
        workflow["8"]["inputs"].update(
            {"width": self.width, "height": self.height, "batch_size": 1}
        )
        return workflow

    def _queue(self, workflow: Dict[str, Any]) -> str:
        body = json.dumps({"prompt": workflow}).encode("utf-8")
        request = urllib.request.Request(
            self._url("prompt"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = self._request_json(request)
        prompt_id = payload.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            details = payload.get("node_errors") or payload
            raise ComfyUIError(f"ComfyUI did not return prompt_id: {details}")
        return prompt_id

    def _wait_for_history(self, prompt_id: str) -> Dict[str, Any]:
        deadline = self.clock() + self.timeout_seconds
        while self.clock() < deadline:
            request = urllib.request.Request(
                self._url(f"history/{urllib.parse.quote(prompt_id, safe='')}")
            )
            history = self._request_json(request)
            result = history.get(prompt_id)
            if isinstance(result, dict):
                status = result.get("status") or {}
                status_string = status.get("status_str")
                if status_string == "success":
                    return result
                if status_string == "error" or status.get("completed") is False:
                    details = status.get("messages") or result.get("node_errors")
                    raise ComfyUIError(
                        f"ComfyUI failed for prompt {prompt_id}: {details}"
                    )
            self.sleeper(self.poll_interval_seconds)
        raise ComfyUITimeoutError(
            f"ComfyUI timed out after {self.timeout_seconds}s for prompt {prompt_id}"
        )

    @staticmethod
    def _first_image(history: Dict[str, Any]) -> Dict[str, str]:
        outputs = history.get("outputs") or {}
        if not isinstance(outputs, dict):
            raise ComfyUIError("ComfyUI history has no outputs object")
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            images = node_output.get("images") or []
            for image in images:
                if not isinstance(image, dict):
                    continue
                filename = image.get("filename")
                if filename:
                    return {
                        "filename": str(filename),
                        "subfolder": str(image.get("subfolder", "")),
                        "type": str(image.get("type", "output")),
                    }
        raise ComfyUIError("ComfyUI completed without returning an image")

    def _download_image(self, image: Dict[str, str]) -> bytes:
        query = urllib.parse.urlencode(image)
        request = urllib.request.Request(self._url(f"view?{query}"))
        data = self._request_bytes(request)
        if not data:
            raise ComfyUIError("ComfyUI returned an empty image")
        return data

    def generate(
        self,
        prompt: str,
        output_dir: Path,
        filename_stem: str,
    ) -> GeneratedImage:
        output_dir.mkdir(parents=True, exist_ok=True)
        seed = self.seed_factory()
        workflow = self._build_workflow(prompt, seed)
        prompt_id = self._queue(workflow)
        history = self._wait_for_history(prompt_id)
        image_info = self._first_image(history)
        image_bytes = self._download_image(image_info)

        destination = output_dir / f"{filename_stem}.png"
        temporary = destination.with_suffix(".png.tmp")
        try:
            with open(temporary, "wb") as file:
                file.write(image_bytes)
            os.replace(temporary, destination)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ComfyUIError(f"Cannot save generated image: {destination}") from exc

        return GeneratedImage(path=destination, seed=seed)
