from __future__ import annotations

import getpass
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROVIDERS = {"openai", "anthropic"}


def default_config_path() -> Path:
    return Path(os.getenv("MECHFERRET_CONFIG", ".mechferret/config.json"))


@dataclass(slots=True)
class ProviderSettings:
    api_key: str = ""
    model: str = ""


@dataclass(slots=True)
class MechFerretConfig:
    default_provider: str = "local"
    providers: dict[str, ProviderSettings] = field(default_factory=dict)

    def provider(self, name: str) -> ProviderSettings:
        return self.providers.setdefault(name, ProviderSettings())


def _safe_string(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _provider_settings(value: Any) -> ProviderSettings:
    if not isinstance(value, dict):
        return ProviderSettings()
    return ProviderSettings(
        api_key=_safe_string(value.get("api_key")).strip(),
        model=_safe_string(value.get("model")).strip(),
    )


def _config_payload(config: MechFerretConfig) -> dict[str, Any]:
    default_provider = config.default_provider if config.default_provider in PROVIDERS else "local"
    providers = {}
    for name, settings in config.providers.items():
        if name not in PROVIDERS:
            continue
        if not isinstance(settings, ProviderSettings):
            settings = _provider_settings(settings)
        providers[name] = {"api_key": settings.api_key, "model": settings.model}
    return {"default_provider": default_provider, "providers": providers}


def load_config(path: str | Path | None = None) -> MechFerretConfig:
    config_path = Path(path) if path else default_config_path()
    if not config_path.exists():
        return MechFerretConfig()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return MechFerretConfig()
    if not isinstance(payload, dict):
        return MechFerretConfig()
    default_provider = payload.get("default_provider", "local")
    if default_provider not in PROVIDERS:
        default_provider = "local"
    raw_providers = payload.get("providers", {})
    raw_providers = raw_providers if isinstance(raw_providers, dict) else {}
    providers = {
        name: _provider_settings(settings)
        for name, settings in raw_providers.items()
        if name in PROVIDERS
    }
    return MechFerretConfig(default_provider=default_provider, providers=providers)


def save_config(config: MechFerretConfig, path: str | Path | None = None) -> Path:
    config_path = Path(path) if path else default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _config_payload(config if isinstance(config, MechFerretConfig) else MechFerretConfig())
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return config_path


def configured_api_key(provider: str, config: MechFerretConfig | None = None) -> str:
    if provider not in PROVIDERS:
        return ""
    env_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    env_value = os.getenv(env_name)
    if env_value:
        return env_value
    cfg = config or load_config()
    return cfg.providers.get(provider, ProviderSettings()).api_key


def configured_model(provider: str, config: MechFerretConfig | None = None, override: str | None = None) -> str:
    if isinstance(override, str) and override.strip():
        return override.strip()
    if provider not in PROVIDERS:
        return "local"
    cfg = config or load_config()
    settings = cfg.providers.get(provider, ProviderSettings())
    if isinstance(settings.model, str) and settings.model:
        return settings.model
    if provider == "openai":
        return os.getenv("MECHFERRET_OPENAI_MODEL", "").strip()
    if provider == "anthropic":
        return os.getenv("MECHFERRET_ANTHROPIC_MODEL", "").strip()
    return ""


def prompt_api_key(provider: str) -> str:
    env_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    return getpass.getpass(f"{provider} API key ({env_name}): ").strip()


def configure_provider(
    provider: str,
    api_key: str,
    model: str | None = None,
    make_default: bool = True,
    path: str | Path | None = None,
) -> Path:
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    config = load_config(path)
    settings = config.provider(provider)
    settings.api_key = _safe_string(api_key).strip()
    if isinstance(model, str) and model.strip():
        settings.model = model.strip()
    elif not settings.model:
        settings.model = configured_model(provider, config)
    if make_default and configured_api_key(provider, config) and configured_model(provider, config):
        config.default_provider = provider
    return save_config(config, path)
