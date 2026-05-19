"""Prompt 版本管理器。

从 prompts/ 目录加载指定版本的 prompt 模板，渲染变量，记录使用版本号。

目录结构:
    prompts/
      config.yaml            # 版本配置
      chat_system/
        v1.zh.md             # 中文版 v1
        v1.en.md             # 英文版 v1
        v2.zh.md             # 中文版 v2（新版本）
      ...

用法:
    from prompts import prompt_manager

    text = prompt_manager.load("chat_system", lang="zh", currency="CAD")
    print(prompt_manager.usage_log)  # {"chat_system": "v1"}
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent


class PromptManager:
    """加载、渲染、追踪 prompt 模板版本。"""

    def __init__(self, prompts_dir: str | Path | None = None):
        self._dir = Path(prompts_dir) if prompts_dir else _PROMPTS_DIR
        self._config: dict = {}
        self._usage: dict[str, str] = {}
        self._cache: dict[str, str] = {}
        self._load_config()

    # ── config ────────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        config_path = self._dir / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                self._config = yaml.safe_load(f).get("prompts", {})
        else:
            self._config = {}

    def reload_config(self) -> None:
        """Hot-reload config.yaml（切换版本后调用，无需重启）。"""
        self._load_config()
        self._cache.clear()

    # ── core API ──────────────────────────────────────────────────────────

    def get_version(self, name: str) -> str:
        """返回 config.yaml 中指定 prompt 的当前版本号。"""
        return self._config.get(name, {}).get("version", "v1")

    def load(
        self,
        name: str,
        lang: str = "zh",
        version: str | None = None,
        **variables: str,
    ) -> str:
        """加载 prompt 模板，渲染变量，记录版本。

        Args:
            name:      prompt 名称，如 "chat_system"
            lang:      语言代码 "zh" / "en"
            version:   指定版本（默认读 config.yaml）
            **variables: 模板变量，如 currency="CAD", today="2025-07-01"

        Returns:
            渲染后的 prompt 字符串

        Raises:
            FileNotFoundError: 找不到对应的 prompt 文件
        """
        ver = version or self.get_version(name)
        cache_key = f"{name}/{ver}.{lang}"

        # 读模板（带缓存）
        if cache_key not in self._cache:
            template = self._read_template(name, ver, lang)
            self._cache[cache_key] = template
        else:
            template = self._cache[cache_key]

        rendered = self._render(template, **variables)
        self._usage[name] = ver
        return rendered

    def list_versions(self, name: str) -> list[str]:
        """列出某个 prompt 的所有可用版本。"""
        prompt_dir = self._dir / name
        if not prompt_dir.is_dir():
            return []
        versions = set()
        for f in prompt_dir.iterdir():
            if f.suffix == ".md":
                # "v1.zh.md" → "v1", "v2.md" → "v2"
                versions.add(f.stem.split(".")[0])
        return sorted(versions)

    @property
    def usage_log(self) -> dict[str, str]:
        """本次会话中实际使用的 {prompt_name: version}。"""
        return dict(self._usage)

    # ── internal ──────────────────────────────────────────────────────────

    def _read_template(self, name: str, version: str, lang: str) -> str:
        candidates = [
            self._dir / name / f"{version}.{lang}.md",
            self._dir / name / f"{version}.md",
        ]
        for path in candidates:
            if path.exists():
                logger.debug("Loaded prompt %s/%s.%s", name, version, lang)
                return path.read_text(encoding="utf-8").strip()

        raise FileNotFoundError(
            f"Prompt '{name}' version={version} lang={lang} not found. "
            f"Searched: {[str(c) for c in candidates]}"
        )

    @staticmethod
    def _render(template: str, **variables: str) -> str:
        """替换 ${variable} 占位符。不影响 JSON 中的普通 { }。"""
        for key, val in variables.items():
            template = template.replace(f"${{{key}}}", str(val))
        return template


# 模块级单例，整个项目共用
prompt_manager = PromptManager()
