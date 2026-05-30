from __future__ import annotations

import getpass
import json
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


def load_config(path: str | Path | None = None) -> MechFerretConfig:
    config_path = Path(path) if path else default_config_path()
    if not config_path.exists():
        return MechFerretConfig()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    providers = {
        name: ProviderSettings(**settings)
        for name, settings in payload.get("providers", {}).items()
        if name in PROVIDERS and isinstance(settings, dict)
    }
    return MechFerretConfig(
        default_provider=payload.get("default_provider", "local"),
        providers=providers,
    )


def save_config(config: MechFerretConfig, path: str | Path | None = None) -> Path:
    config_path = Path(path) if path else default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "default_provider": config.default_provider,
        "providers": {name: asdict(settings) for name, settings in config.providers.items()},
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return config_path


def configured_api_key(provider: str, config: MechFerretConfig | None = None) -> str:
    env_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    env_value = os.getenv(env_name)
    if env_value:
        return env_value
    cfg = config or load_config()
    return cfg.providers.get(provider, ProviderSettings()).api_key


def configured_model(provider: str, config: MechFerretConfig | None = None, override: str | None = None) -> str:
    if override:
        return override
    cfg = config or load_config()
    settings = cfg.providers.get(provider, ProviderSettings())
    if settings.model:
        return settings.model
    if provider == "openai":
        return os.getenv("MECHFERRET_OPENAI_MODEL", "gpt-5")
    if provider == "anthropic":
        return os.getenv("MECHFERRET_ANTHROPIC_MODEL", "claude-sonnet-4-5")
    return "local"


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
    settings.api_key = api_key
    if model:
        settings.model = model
    elif not settings.model:
        settings.model = configured_model(provider, config)
    if make_default:
        config.default_provider = provider
    return save_config(config, path)

