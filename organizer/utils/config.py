from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
import hashlib
import re

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def compute_reference_hash() -> str:
    """Compute hash of all YAML config files.

    Returns a SHA-256 hash of all YAML config files in the config directory.
    This is used to track config version per snapshot.

    Returns:
        Hexadecimal hash string
    """
    config_files = sorted(CONFIG_DIR.glob("*.yaml"))
    hasher = hashlib.sha256()
    for f in config_files:
        hasher.update(f.read_bytes())
    return hasher.hexdigest()


class VariantGroup(Enum):
    SEASON = "season"
    TIME = "time"
    STYLE = "style"
    LAYOUT = "layout"
    MEDIA = "media"
    OTHER = "other"
    UNCATEGORIZED = "uncategorized"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return data


def _normalize_token(token: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", token.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _tokenize_variant_phrase(token: str) -> list[str]:
    cleaned = _normalize_token(token)
    return [part for part in cleaned.split() if part]


def _build_variant_cache(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    token_to_variants: dict[str, set[str]] = {}
    type_index: dict[str, set[str]] = {}
    grouping_index: dict[str, set[str]] = {}
    type_by_token: dict[str, str] = {}
    grouping_by_token: dict[str, str] = {}
    normalized_tokens: set[str] = set()
    phrase_tokens: set[str] = set()

    for name, data in variants.items():
        synonyms = data.get("synonyms") or []
        variant_type = data.get("type") or "variant"
        grouping = data.get("grouping") or VariantGroup.UNCATEGORIZED.value
        if grouping not in {group.value for group in VariantGroup}:
            raise ValueError(f"Unknown variant grouping '{grouping}' for '{name}'")

        type_index.setdefault(variant_type, set()).add(name)
        grouping_index.setdefault(grouping, set()).add(name)

        tokens = [name, *synonyms]
        for token in tokens:
            normalized = _normalize_token(token)
            normalized_tokens.add(normalized)
            if " " in normalized:
                phrase_tokens.add(normalized)
            token_to_variants.setdefault(normalized, set()).add(name)
            type_by_token[normalized] = variant_type
            grouping_by_token[normalized] = grouping

    expanded_tokens = set(normalized_tokens)
    for token in phrase_tokens:
        expanded_tokens.update(_tokenize_variant_phrase(token))

    return {
        "variants": variants,
        "token_to_variants": token_to_variants,
        "types": type_index,
        "groupings": grouping_index,
        "type_by_token": type_by_token,
        "grouping_by_token": grouping_by_token,
        "known_tokens": expanded_tokens,
        "phrases": phrase_tokens,
    }


def _build_creator_remove_cache(creators: dict[str, dict[str, Any]]) -> dict[str, Any]:
    creator_to_removes: dict[str, list[str]] = {}
    creator_strings: set[str] = set()
    remove_to_creators: dict[str, set[str]] = {}
    for creator, data in creators.items():
        creator_strings.add(_normalize_token(creator))
        for synonym in data.get("synonyms") or []:
            creator_strings.add(_normalize_token(synonym))
        removes = data.get("tokens_to_remove") or []
        creator_to_removes[creator] = removes
        remove_items = removes
        for remove_item in remove_items:
            if remove_item == "":
                continue
            normalized = _normalize_token(remove_item)
            remove_to_creators.setdefault(normalized, set()).add(creator)

    return {
        "creators": creators,
        "creator_to_removes": creator_to_removes,
        "creator_strings": creator_strings,
        "remove_to_creators": remove_to_creators,
    }


@dataclass(frozen=True)
class Config:
    creators: dict[str, dict[str, Any]]
    creator_removes: dict[str, list[str]]
    creator_strings: set[str]
    file_name_exceptions: dict[str, str]
    replace_exceptions: dict[str, str]
    clean_exceptions: set[str]
    should_ignore: set[str]
    grouping_exceptions: tuple[str, ...]
    variants: dict[str, dict[str, Any]]
    known_variant_tokens: set[str]
    variant_type_by_string: dict[str, str]
    variant_grouping_by_string: dict[str, str]
    variant_types: set[str]
    media_types: set[str]
    format_types: set[str]
    relational_cache: dict[str, Any]

    def is_variant(self, value: str) -> bool:
        normalized = _normalize_token(value)
        return self.variant_type_by_string.get(normalized) == "variant"

    def is_media_type(self, value: str) -> bool:
        normalized = _normalize_token(value)
        return self.variant_type_by_string.get(normalized) == "media_type"

    def is_group(self, value: str, group: VariantGroup) -> bool:
        normalized = _normalize_token(value)
        return self.variant_grouping_by_string.get(normalized) == group.value


@lru_cache(maxsize=1)
def get_config() -> Config:
    filename_config = _load_yaml(CONFIG_DIR / "filename_cleaning.yaml")
    creators_config = _load_yaml(CONFIG_DIR / "creators.yaml")
    grouping_config = _load_yaml(CONFIG_DIR / "grouping.yaml")
    variant_config = _load_yaml(CONFIG_DIR / "variants.yaml")

    creators = creators_config.get("creators", {})
    file_name_exceptions = filename_config.get("file_name_exceptions", {})
    replace_exceptions = filename_config.get("replace_exceptions", {})
    clean_exceptions = set(filename_config.get("clean_exceptions", []))
    should_ignore = set(filename_config.get("should_ignore", []))

    grouping_exceptions = tuple(grouping_config.get("grouping_exceptions", []))

    variants = variant_config.get("variants") or {}
    variant_cache = _build_variant_cache(variants)
    creator_cache = _build_creator_remove_cache(creators)

    variant_types = {
        name.lower() for name in variant_cache["types"].get("variant", set())
    }
    media_types = {
        name.lower() for name in variant_cache["types"].get("media_type", set())
    }
    format_types = {
        name.lower() for name in variant_cache["types"].get("media_format", set())
    }

    relational_cache = {
        "variant_tokens": variant_cache,
        "creator_removes": creator_cache,
    }

    return Config(
        creators=creators,
        creator_removes=creator_cache["creator_to_removes"],
        creator_strings=creator_cache["creator_strings"],
        file_name_exceptions=file_name_exceptions,
        replace_exceptions=replace_exceptions,
        clean_exceptions=clean_exceptions,
        should_ignore=should_ignore,
        grouping_exceptions=grouping_exceptions,
        variants=variants,
        known_variant_tokens=variant_cache["known_tokens"],
        variant_type_by_string=variant_cache["type_by_token"],
        variant_grouping_by_string=variant_cache["grouping_by_token"],
        variant_types=variant_types,
        media_types=media_types,
        format_types=format_types,
        relational_cache=relational_cache,
    )


def get_minimal_config() -> Config:
    creators: dict[str, dict[str, Any]] = {}
    variants: dict[str, dict[str, Any]] = {}

    variant_cache = _build_variant_cache(variants)
    creator_cache = _build_creator_remove_cache(creators)

    relational_cache = {
        "variant_tokens": variant_cache,
        "creator_removes": creator_cache,
    }

    return Config(
        creators=creators,
        creator_removes=creator_cache["creator_to_removes"],
        creator_strings=creator_cache["creator_strings"],
        file_name_exceptions={},
        replace_exceptions={},
        clean_exceptions=set(),
        should_ignore=set(),
        grouping_exceptions=tuple(),
        variants=variants,
        known_variant_tokens=variant_cache["known_tokens"],
        variant_type_by_string=variant_cache["type_by_token"],
        variant_grouping_by_string=variant_cache["grouping_by_token"],
        variant_types=set(),
        media_types=set(),
        format_types=set(),
        relational_cache=relational_cache,
    )


_CONFIG = get_config()

CREATOR_REMOVES = _CONFIG.creator_removes
FILE_NAME_EXCEPTIONS = _CONFIG.file_name_exceptions
REPLACE_EXCEPTIONS = _CONFIG.replace_exceptions
CLEAN_EXCEPTIONS = _CONFIG.clean_exceptions
GROUPING_EXCEPTIONS = _CONFIG.grouping_exceptions

KNOWN_VARIANT_TOKENS = _CONFIG.known_variant_tokens
RELATIONAL_CACHE = _CONFIG.relational_cache
CREATOR_STRINGS = _CONFIG.creator_strings
