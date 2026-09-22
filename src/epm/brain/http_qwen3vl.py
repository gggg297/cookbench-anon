from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import requests
from requests import Response
from PIL import Image, ImageDraw

from epm.core.http_403_pause import classify_network_exception


@dataclass(frozen=True)
class Qwen3VlHttpClient:
    """
    Client for the custom Qwen3-VL HTTP service:
      - GET  /health
      - POST /v1/chat  {"prompt": "...", "image_b64": "...", "max_new_tokens": 800}
        -> {"text": "..."}
    """

    base_url: str
    connect_timeout_s: Optional[float] = 5.0
    read_timeout_s: Optional[float] = 60.0

    # Image payload controls (to avoid huge base64 requests / connection resets)
    image_max_side: int = 1024
    image_format: str = "jpeg"  # "jpeg" | "png"
    jpeg_quality: int = 85

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def health(self) -> Dict[str, Any]:
        resp = requests.get(self._url("/health"), timeout=(self.connect_timeout_s, self.read_timeout_s))
        self._raise_for_status_with_body(resp)
        data = resp.json()
        return data if isinstance(data, dict) else {"raw": data}

    def chat(
        self,
        *,
        prompt: str,
        image_b64: Optional[str] = None,
        image_path: Optional[str | Path] = None,
        image_paths: Optional[Sequence[str | Path]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if image_b64 and (image_path or image_paths):
            raise ValueError("Provide image_b64 or image_path(s), not both.")
        if image_path is not None and image_paths:
            raise ValueError("Provide only one of image_path or image_paths.")
        if image_paths:
            image_b64 = self._encode_images_for_request([Path(p) for p in image_paths if str(p or "").strip()])
        elif image_path is not None:
            image_b64 = self._encode_image_for_request(Path(image_path))

        payload: Dict[str, Any] = {"prompt": str(prompt)}
        if image_b64:
            payload["image_b64"] = str(image_b64)
        if max_new_tokens is not None:
            payload["max_new_tokens"] = int(max_new_tokens)

        resp = requests.post(self._url("/v1/chat"), json=payload, timeout=(self.connect_timeout_s, self.read_timeout_s))
        self._raise_for_status_with_body(resp)
        data = resp.json()
        if not isinstance(data, dict) or "text" not in data:
            raise ValueError(f"Invalid response: {data!r}")
        text = data.get("text")
        if not isinstance(text, str):
            raise ValueError("Response field 'text' must be a string")
        return text

    def _encode_image_for_request(self, path: Path) -> str:
        """
        Encode an image for the /v1/chat endpoint.

        Default behavior downsizes + recompresses to reduce payload size. This helps
        avoid server-side request limits and client-side ConnectionResetError on large PNGs.
        """
        fmt = (self.image_format or "jpeg").strip().lower()
        if fmt not in ("jpeg", "jpg", "png"):
            fmt = "jpeg"

        img = Image.open(path)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        if fmt in ("jpeg", "jpg") and img.mode == "RGBA":
            img = img.convert("RGB")

        max_side = int(self.image_max_side) if self.image_max_side else 0
        if max_side > 0 and max(img.size) > max_side:
            img.thumbnail((max_side, max_side), resample=Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        if fmt in ("jpeg", "jpg"):
            img.save(buf, format="JPEG", quality=int(self.jpeg_quality), optimize=True)
        else:
            img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _encode_images_for_request(self, paths: Sequence[Path]) -> str:
        valid = [p for p in paths if p is not None]
        if not valid:
            return ""
        if len(valid) == 1:
            return self._encode_image_for_request(valid[0])

        fmt = (self.image_format or "jpeg").strip().lower()
        if fmt not in ("jpeg", "jpg", "png"):
            fmt = "jpeg"

        panels: list[Image.Image] = []
        max_side = int(self.image_max_side) if self.image_max_side else 0
        for path in valid:
            img = Image.open(path)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            if fmt in ("jpeg", "jpg") and img.mode == "RGBA":
                img = img.convert("RGB")
            if max_side > 0 and max(img.size) > max_side:
                img.thumbnail((max_side, max_side), resample=Image.Resampling.LANCZOS)
            panels.append(img)

        gap = 12
        label_h = 36
        total_w = sum(img.width for img in panels) + gap * (len(panels) - 1)
        total_h = max(img.height for img in panels) + label_h
        mode = "RGB" if fmt in ("jpeg", "jpg") else "RGBA"
        bg = (255, 255, 255) if mode == "RGB" else (255, 255, 255, 255)
        canvas = Image.new(mode, (total_w, total_h), bg)
        draw = ImageDraw.Draw(canvas)

        x = 0
        for idx, img in enumerate(panels):
            draw.text((x, 8), f"image_{idx + 1}", fill=(0, 0, 0))
            y = label_h
            if img.mode != mode:
                img = img.convert(mode)
            canvas.paste(img, (x, y))
            x += img.width + gap

        buf = io.BytesIO()
        if fmt in ("jpeg", "jpg"):
            if canvas.mode == "RGBA":
                canvas = canvas.convert("RGB")
            canvas.save(buf, format="JPEG", quality=int(self.jpeg_quality), optimize=True)
        else:
            canvas.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _raise_for_status_with_body(resp: Response) -> None:
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            network_signal = classify_network_exception(e, source="http_qwen3vl")
            if network_signal is not None:
                raise network_signal from None
            body = ""
            try:
                body = resp.text
            except Exception:
                body = "<unreadable>"
            raise requests.HTTPError(
                f"{e} | response_body={body[:2000]!r}",
                response=resp,
                request=e.request,
            ) from None
