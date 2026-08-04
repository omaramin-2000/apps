#!/usr/bin/env python3
"""Validate the config.yaml and translations of every app in this repository.

The Supervisor silently skips an app whose config.yaml cannot be parsed or
validated, which makes the app disappear from the store instead of reporting an
error. This script catches those problems before they are merged.

It also checks each app's translations against its options, since an option with
no translation shows up in the UI as a raw key name rather than failing loudly.

Usage (from the repository root):

    python3 .github/scripts/validate_configs.py [config.yaml ...]

Exits non-zero if any error is found. Warnings are reported but do not fail.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_KEYS = ("name", "version", "slug", "description", "arch")

VALID_ARCH = {"aarch64", "amd64", "armhf", "armv7", "i386"}

# Top-level keys the Supervisor understands. Only used to catch typos, so keys
# missing from this set are reported as warnings rather than errors.
KNOWN_KEYS = {
    "advanced", "apparmor", "arch", "audio", "auth_api", "backup",
    "backup_exclude", "backup_post", "backup_pre", "boot", "breaking_versions",
    "codenotary", "description", "devices", "devicetree", "discovery", "dns",
    "docker_api", "environment", "full_access", "gpio", "hassio_api",
    "hassio_role", "homeassistant", "homeassistant_api", "host_dbus",
    "host_ipc", "host_network", "host_pid", "host_uts", "image", "ingress",
    "ingress_entry", "ingress_port", "ingress_stream", "init", "journald",
    "kernel_modules", "labels", "map", "name", "options", "panel_admin",
    "panel_icon", "panel_title", "ports", "ports_description", "privileged",
    "realtime", "schema", "services", "slug", "startup", "stage", "stdin",
    "timeout", "tmpfs", "translations", "typing", "udev", "uart", "url",
    "usb", "version", "video", "watchdog", "webui",
}

# Mirrors the Supervisor's RE_SCHEMA_ELEMENT.
RE_SCHEMA_ELEMENT = re.compile(
    r"^(?:"
    r"list\((?P<list>.+)\)"
    r"|match\((?P<match>.*)\)"
    r"|device(?:\((?P<device_filter>.*)\))?"
    r"|(?P<type>bool|email|url|port|str|password|int|float)"
    r"(?:\((?P<range_start>[\d.]*),(?P<range_end>[\d.]*)\))?"
    r")(?P<optional>\?)?$"
)

# The Supervisor coerces these to booleans, so they are valid `bool` defaults.
BOOL_STRINGS = {"1", "0", "y", "n", "yes", "no", "true", "false", "on", "off"}


class Report:
    """Collects errors and warnings for a single config file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rel_path = os.path.relpath(path, REPO_ROOT)
        self.errors: list[tuple[str, int | None]] = []
        self.warnings: list[tuple[str, int | None]] = []
        # Populated once the file parses as a mapping.
        self.config: dict[str, Any] | None = None
        try:
            self._lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            self._lines = []

    def error(self, message: str, key: str | None = None, line: int | None = None) -> None:
        self.errors.append((message, line if line is not None else self.find_line(key)))

    def warning(self, message: str, key: str | None = None, line: int | None = None) -> None:
        self.warnings.append((message, line if line is not None else self.find_line(key)))

    def find_line(self, key: str | None) -> int | None:
        """Return the 1-based line of `key:`, for editor and CI annotations."""
        if key is None:
            return None
        # Nested keys arrive dotted (`nested.inner[0]`) but appear in the file
        # as the bare leaf name.
        leaf = re.sub(r"\[\d+\]$", "", key.rsplit(".", 1)[-1])
        pattern = re.compile(rf"^\s*[\"']?{re.escape(leaf)}[\"']?\s*:")
        for index, line in enumerate(self._lines, start=1):
            if pattern.match(line):
                return index
        return None


def parse_schema_element(element: str) -> dict[str, Any] | None:
    """Parse a scalar schema element such as `int(0,4)` or `str?`."""
    match = RE_SCHEMA_ELEMENT.match(element)
    if match is None:
        return None
    groups = match.groupdict()
    if groups["list"] is not None:
        kind, detail = "list", groups["list"].split("|")
    elif groups["match"] is not None:
        kind, detail = "match", groups["match"]
    elif groups["type"] is not None:
        kind, detail = groups["type"], None
    else:
        kind, detail = "device", None
    return {
        "kind": kind,
        "detail": detail,
        "optional": groups["optional"] == "?",
        "range_start": groups["range_start"] or None,
        "range_end": groups["range_end"] or None,
    }


def describe(value: Any) -> str:
    """Render a default value the way it appears in the file."""
    if isinstance(value, str):
        return f'"{value}"'
    if value is None:
        return "null"
    return str(value)


def check_default(report: Report, key: str, value: Any, element: str) -> None:
    """Check that a default in `options` satisfies its `schema` entry."""
    parsed = parse_schema_element(element)
    if parsed is None:
        report.error(
            f"schema.{key}: {describe(element)} is not a valid schema type",
            key=key,
        )
        return

    if value is None:
        if not parsed["optional"]:
            report.error(
                f"options.{key} is null but schema says {describe(element)}; "
                f"use '{element}?' to allow an empty value",
                key=key,
            )
        return

    kind = parsed["kind"]

    if isinstance(value, (dict, list)):
        report.error(
            f"options.{key} is a {type(value).__name__} but schema says "
            f"{describe(element)}",
            key=key,
        )
        return

    if kind in ("str", "password", "email", "url", "device"):
        # The Supervisor coerces any scalar to a string, so only the length
        # constraint can actually fail here.
        check_range(report, key, len(str(value)), parsed, "length of")
        return

    if kind == "bool":
        if isinstance(value, bool):
            return
        if isinstance(value, int) and value in (0, 1):
            return
        if isinstance(value, str) and value.lower() in BOOL_STRINGS:
            return
        report.error(
            f"options.{key}: {describe(value)} is not a boolean", key=key
        )
        return

    if kind in ("int", "port", "float"):
        number = coerce_number(value, integer=kind != "float")
        if number is None:
            report.error(
                f"options.{key}: {describe(value)} is not a valid {kind} "
                f"(schema says {describe(element)})",
                key=key,
            )
            return
        if kind == "port" and not 1 <= number <= 65535:
            report.error(
                f"options.{key}: {number} is not a valid port (1-65535)", key=key
            )
            return
        check_range(report, key, number, parsed, "")
        return

    if kind == "list":
        if str(value) not in parsed["detail"]:
            report.error(
                f"options.{key}: {describe(value)} is not one of "
                f"{', '.join(parsed['detail'])}",
                key=key,
            )
        return

    if kind == "match" and not re.match(parsed["detail"], str(value)):
        report.error(
            f"options.{key}: {describe(value)} does not match "
            f"{describe(parsed['detail'])}",
            key=key,
        )


def coerce_number(value: Any, integer: bool) -> int | float | None:
    """Coerce a default to a number the way the Supervisor would, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if integer else float(value)
    try:
        return int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None


def check_range(
    report: Report, key: str, number: float, parsed: dict[str, Any], prefix: str
) -> None:
    start, end = parsed["range_start"], parsed["range_end"]
    label = f"{prefix} " if prefix else ""
    if start is not None and number < float(start):
        report.error(
            f"options.{key}: {label}{number} is below the schema minimum {start}",
            key=key,
        )
    if end is not None and number > float(end):
        report.error(
            f"options.{key}: {label}{number} is above the schema maximum {end}",
            key=key,
        )


def check_options(report: Report, options: Any, schema: Any, path: str = "") -> None:
    """Compare the `options` defaults against the `schema` definition."""
    if not isinstance(options, dict):
        report.error(f"{path or 'options'} must be a mapping", key="options")
        return
    if not isinstance(schema, dict):
        report.error(f"{path or 'schema'} must be a mapping", key="schema")
        return

    for key, value in options.items():
        full_key = f"{path}{key}"
        if key not in schema:
            report.error(
                f"options.{full_key} has no entry in schema; the Supervisor "
                f"rejects options that the schema does not define",
                key=key,
            )
            continue
        element = schema[key]
        if isinstance(element, dict):
            check_options(report, value, element, path=f"{full_key}.")
        elif isinstance(element, list):
            check_list(report, full_key, value, element)
        elif isinstance(element, str):
            check_default(report, full_key, value, element)
        else:
            report.error(
                f"schema.{full_key}: {describe(element)} is not a valid "
                f"schema type",
                key=key,
            )

    for key, element in schema.items():
        if key in options:
            continue
        # A required entry with no default forces the user to fill it in before
        # the app can start, which is occasionally intentional.
        if isinstance(element, str) and not element.endswith("?"):
            report.warning(
                f"schema.{path}{key} is required but has no default in options",
                key=key,
            )


def check_list(report: Report, key: str, value: Any, element: list[Any]) -> None:
    """Check a list default against a single-entry list schema."""
    if not isinstance(value, list):
        report.error(f"options.{key} must be a list", key=key)
        return
    if len(element) != 1:
        report.error(
            f"schema.{key} must contain exactly one entry describing the "
            f"list items",
            key=key,
        )
        return
    item_schema = element[0]
    for index, item in enumerate(value):
        if isinstance(item_schema, dict):
            check_options(report, item, item_schema, path=f"{key}[{index}].")
        elif isinstance(item_schema, str):
            check_default(report, f"{key}[{index}]", item, item_schema)


def load_yaml(report: Report) -> Any:
    """Parse the report's file, recording a precise error if it fails."""
    try:
        raw = report.path.read_text(encoding="utf-8")
    except OSError as err:
        report.error(f"cannot read file: {err}")
        return None

    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as err:
        mark = getattr(err, "problem_mark", None)
        detail = getattr(err, "problem", None) or str(err).splitlines()[0]
        report.error(
            f"invalid YAML: {detail}",
            line=mark.line + 1 if mark is not None else None,
        )
        return None


def find_english(translations_dir: Path) -> Path | None:
    for name in ("en.yaml", "en.yml"):
        candidate = translations_dir / name
        if candidate.is_file():
            return candidate
    return None


def check_translations(report: Report) -> list[Report]:
    """Check that every option has a translation entry, and vice versa.

    Returns a report per translations file. A missing translations file is
    recorded on the config report instead, since there is no file to annotate.
    """
    config = report.config or {}
    options = config.get("options")
    option_keys = set(options) if isinstance(options, dict) else set()
    translations_dir = report.path.parent / "translations"
    english = find_english(translations_dir)

    if english is None:
        if option_keys:
            report.warning(
                "no translations/en.yaml; options are shown in the UI as raw "
                "key names",
                key="options",
            )
        return []

    reports = [
        check_translation_file(path, option_keys, complete=path == english)
        for path in sorted(translations_dir.glob("*.y*ml"))
    ]
    return [report for report in reports if report.errors or report.warnings]


def check_translation_file(
    path: Path, option_keys: set[str], complete: bool
) -> Report:
    """Check one translations file against the options in config.yaml.

    Only English is required to be complete; other languages lag behind by
    nature, so they are checked for stale keys only.
    """
    report = Report(path)
    data = load_yaml(report)
    if report.errors:
        return report
    if data is None:
        data = {}
    if not isinstance(data, dict):
        report.error("translations must be a mapping of keys to values")
        return report

    configuration = data.get("configuration")
    if configuration is None:
        if option_keys and complete:
            report.warning(
                "no 'configuration' section; options are shown in the UI as "
                "raw key names"
            )
        return report
    if not isinstance(configuration, dict):
        report.error("configuration must be a mapping", key="configuration")
        return report

    if complete:
        for key in sorted(option_keys - set(configuration)):
            report.warning(
                f"option '{key}' has no translation; it is shown in the UI as "
                f"a raw key name"
            )

    for key, entry in configuration.items():
        if key not in option_keys:
            report.warning(
                f"configuration.{key} does not match any option in config.yaml",
                key=key,
            )
            continue
        if not isinstance(entry, dict):
            report.error(
                f"configuration.{key} must be a mapping with 'name' and "
                f"'description'",
                key=key,
            )
            continue
        for field in ("name", "description"):
            if not entry.get(field):
                report.warning(f"configuration.{key} has no '{field}'", key=key)

    return report


def validate(path: Path) -> list[Report]:
    """Validate one app config plus its translations."""
    report = validate_config(path)
    if report.config is None:
        return [report]
    return [report, *check_translations(report)]


def validate_config(path: Path) -> Report:
    report = Report(path)

    config = load_yaml(report)
    if report.errors:
        return report

    if not isinstance(config, dict):
        report.error("config must be a mapping of keys to values")
        return report

    report.config = config

    for key in REQUIRED_KEYS:
        if key not in config:
            report.error(f"missing required key '{key}'")

    # `version: 1.10` parses as the float 1.1, silently changing the version.
    version = config.get("version")
    if version is not None and not isinstance(version, str):
        report.error(
            f"version must be a string; {describe(version)} was parsed as "
            f"{type(version).__name__} — quote it",
            key="version",
        )

    slug = config.get("slug")
    if isinstance(slug, str) and slug != path.parent.name:
        report.warning(
            f"slug '{slug}' does not match the directory name "
            f"'{path.parent.name}'",
            key="slug",
        )

    arch = config.get("arch")
    if isinstance(arch, list):
        for entry in sorted(set(arch) - VALID_ARCH):
            report.error(f"arch: '{entry}' is not a known architecture", key="arch")
    elif arch is not None:
        report.error("arch must be a list", key="arch")

    for key in config:
        if key not in KNOWN_KEYS:
            report.warning(f"unknown top-level key '{key}'", key=key)

    schema = config.get("schema")
    options = config.get("options")

    # `schema: false` disables option validation entirely.
    if schema is False:
        return report
    if options is None and schema is None:
        return report
    if schema is None:
        report.error("options is set but schema is missing", key="options")
        return report
    if options is None:
        report.error("schema is set but options is missing", key="schema")
        return report

    check_options(report, options, schema)
    return report


def annotate(level: str, report: Report, message: str, line: int | None) -> None:
    """Emit a GitHub Actions annotation so the problem shows up in the diff."""
    location = f"file={report.rel_path}"
    if line is not None:
        location += f",line={line}"
    print(f"::{level} {location}::{message}")


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(arg).resolve() for arg in argv]
    else:
        paths = sorted(REPO_ROOT.glob("*/config.yaml"))
        paths += sorted(REPO_ROOT.glob("*/config.yml"))

    if not paths:
        print("No app config files found", file=sys.stderr)
        return 1

    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    total_errors = 0
    total_warnings = 0

    for path in paths:
        for report in validate(path):
            total_errors += len(report.errors)
            total_warnings += len(report.warnings)

            if not report.errors and not report.warnings:
                print(f"OK       {report.rel_path}")
                continue

            print(f"{'FAIL' if report.errors else 'WARN'}     {report.rel_path}")
            for message, line in report.errors:
                where = f":{line}" if line is not None else ""
                print(f"  error{where}: {message}")
                if in_actions:
                    annotate("error", report, message, line)
            for message, line in report.warnings:
                where = f":{line}" if line is not None else ""
                print(f"  warning{where}: {message}")
                if in_actions:
                    annotate("warning", report, message, line)

    print(
        f"\nChecked {len(paths)} config file(s): "
        f"{total_errors} error(s), {total_warnings} warning(s)"
    )
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
