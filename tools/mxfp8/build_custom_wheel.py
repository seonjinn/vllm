# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build a deterministic, distributable adaptive-MXFP8 vLLM wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

EXPECTED_BASE_WHEEL_SHA256 = (
    "76ccf4c0554556c06f6b0fb1643742d4cf97dcc69f6ef3f04556d0764126035a"
)
UPSTREAM_VLLM_BASE_COMMIT = "5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1"
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_BUILD_TAG_PREFIX = "1mxfp8g"
_PROVENANCE_SCHEMA_VERSION = 1
_PACKAGE_PROVENANCE_PATH = "vllm/mxfp8_wheel_provenance.json"

_PACKAGE_DATA_PATTERNS = (
    "vllm/libs/*.so*",
    "vllm/model_executor/layers/fused_moe/configs/*.json",
    "vllm/model_executor/layers/quantization/utils/configs/*.json",
    "vllm/model_executor/kernels/linear/mxfp8/tactic_configs/*.json",
    "vllm/entrypoints/serve/instrumentator/static/*.js",
    "vllm/entrypoints/serve/instrumentator/static/*.css",
    "vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/*.cpp",
    "vllm/third_party/deep_gemm/include/**/*.cuh",
    "vllm/third_party/deep_gemm/include/**/*.h",
    "vllm/third_party/deep_gemm/include/**/*.hpp",
)


class BuildError(RuntimeError):
    """Raised when a wheel input or output violates the build contract."""


@dataclass(frozen=True)
class HostPlatform:
    """Host identity used by the fail-closed architecture gate."""

    system: str
    machine: str


@dataclass(frozen=True)
class BuildPolicy:
    """Immutable compatibility identity for the custom wheel."""

    expected_base_sha256: str = EXPECTED_BASE_WHEEL_SHA256
    upstream_base_commit: str = UPSTREAM_VLLM_BASE_COMMIT
    distribution: str = "vllm"
    version: str = "0.20.2"
    python_tag: str = "cp38"
    abi_tag: str = "abi3"
    platform_tag: str = "manylinux_2_35_aarch64"
    wheel_metadata_platform_tag: str = "linux_aarch64"

    @property
    def tag(self) -> str:
        """Return the exact Python-ABI-platform tag tuple."""
        return f"{self.python_tag}-{self.abi_tag}-{self.platform_tag}"

    @property
    def base_filename(self) -> str:
        """Return the only accepted official base-wheel filename."""
        return f"{self.distribution}-{self.version}-{self.tag}.whl"

    @property
    def wheel_metadata_tag(self) -> str:
        """Return the official wheel's unchanged internal WHEEL tag."""
        return f"{self.python_tag}-{self.abi_tag}-{self.wheel_metadata_platform_tag}"


@dataclass(frozen=True)
class WheelBuildRequest:
    """Inputs required for one clean-source custom-wheel build."""

    repo_root: Path
    source_commit: str
    base_wheel: Path
    output_dir: Path
    policy: BuildPolicy
    host: HostPlatform


@dataclass(frozen=True)
class BuildArtifacts:
    """Published wheel plus adjacent deterministic verification artifacts."""

    wheel: Path
    metadata: Path
    sha256: Path


@dataclass(frozen=True)
class _WheelMember:
    data: bytes
    mode: int = 0o100644
    compression: int = zipfile.ZIP_DEFLATED


@dataclass(frozen=True)
class _SourceSnapshot:
    files: Mapping[str, _WheelMember]
    deleted_runtime_files: tuple[str, ...]
    changed_runtime_files: tuple[str, ...]
    tactic_json_files: tuple[str, ...]
    source_tree_sha256: str


@dataclass(frozen=True)
class _GitTreeEntry:
    path: str
    mode: str
    object_type: str
    object_id: str


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(
    repo_root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=check,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        diagnostic = ""
        if isinstance(error, subprocess.CalledProcessError):
            diagnostic = (error.stderr or error.stdout or "").strip()
        suffix = f": {diagnostic}" if diagnostic else ""
        raise BuildError(f"git {' '.join(arguments)} failed{suffix}") from error


def _validate_host(host: HostPlatform) -> None:
    if host.system != "Linux":
        raise BuildError(f"custom wheel builds require Linux, found {host.system!r}")
    if host.machine != "aarch64":
        raise BuildError(f"custom wheel builds require aarch64, found {host.machine!r}")


def _validate_source_checkout(
    repo_root: Path,
    source_commit: str,
    upstream_base_commit: str,
) -> None:
    if not repo_root.is_dir():
        raise BuildError(f"repository root does not exist: {repo_root}")
    if _COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise BuildError("source commit must be a full lowercase 40-character Git SHA")
    if _COMMIT_PATTERN.fullmatch(upstream_base_commit) is None:
        raise BuildError("upstream base commit must be a full lowercase Git SHA")

    head = _run_git(repo_root, "rev-parse", "HEAD").stdout.strip()
    if head != source_commit:
        raise BuildError(
            f"source commit {source_commit} is not the checked-out HEAD {head}"
        )
    dirty = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    if dirty:
        details = ", ".join(line.rstrip() for line in dirty.splitlines()[:8])
        raise BuildError(f"source tree is dirty: {details}")

    ancestry = _run_git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        upstream_base_commit,
        source_commit,
        check=False,
    )
    if ancestry.returncode != 0:
        raise BuildError(
            f"expected upstream base {upstream_base_commit} is not an ancestor "
            f"of source commit {source_commit}"
        )


def _validate_external_build_paths(
    repo_root: Path,
    base_wheel: Path,
    output_dir: Path,
) -> None:
    for label, path in (
        ("base wheel", base_wheel),
        ("output directory", output_dir),
    ):
        try:
            path.relative_to(repo_root)
        except ValueError:
            continue
        raise BuildError(f"{label} must be outside the source checkout: {path}")


def _is_runtime_source(path: str) -> bool:
    if not path.startswith("vllm/"):
        return False
    if path.endswith((".py", ".pyi")) or path == "vllm/py.typed":
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in _PACKAGE_DATA_PATTERNS)


def _run_git_bytes(
    repo_root: Path,
    *arguments: str,
) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        diagnostic = ""
        if isinstance(error, subprocess.CalledProcessError):
            diagnostic = (
                (error.stderr or error.stdout or b"").decode(errors="replace").strip()
            )
        suffix = f": {diagnostic}" if diagnostic else ""
        raise BuildError(f"git {' '.join(arguments)} failed{suffix}") from error


def _read_git_runtime_tree(
    repo_root: Path,
    commit: str,
) -> dict[str, _GitTreeEntry]:
    raw_tree = _run_git_bytes(
        repo_root,
        "ls-tree",
        "-rz",
        "--full-tree",
        commit,
        "--",
        "vllm",
    )
    entries: dict[str, _GitTreeEntry] = {}
    for raw_record in raw_tree.split(b"\0"):
        if not raw_record:
            continue
        raw_metadata, separator, raw_path = raw_record.partition(b"\t")
        metadata_fields = raw_metadata.split(b" ")
        if not separator or len(metadata_fields) != 3:
            raise BuildError("git ls-tree returned an invalid NUL record")
        try:
            mode, object_type, object_id = (
                field.decode("ascii") for field in metadata_fields
            )
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BuildError("Git tree contains a non-UTF-8 path or field") from error
        _validate_member_name(path)
        if not _is_runtime_source(path):
            continue
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise BuildError(
                f"unsupported Git mode/type for runtime member {path}: "
                f"{mode} {object_type}"
            )
        if path in entries:
            raise BuildError(f"Git tree contains duplicate runtime path: {path}")
        entries[path] = _GitTreeEntry(
            path=path,
            mode=mode,
            object_type=object_type,
            object_id=object_id,
        )
    return entries


def _read_git_blob(
    repo_root: Path,
    entry: _GitTreeEntry,
) -> bytes:
    return _run_git_bytes(repo_root, "cat-file", "blob", entry.object_id)


def _load_source_snapshot(
    repo_root: Path,
    source_commit: str,
    upstream_base_commit: str,
) -> _SourceSnapshot:
    source_tree = _read_git_runtime_tree(
        repo_root,
        source_commit,
    )
    base_tree = _read_git_runtime_tree(
        repo_root,
        upstream_base_commit,
    )
    files = {
        path: _WheelMember(
            data=_read_git_blob(repo_root, entry),
            mode=int(entry.mode, 8),
        )
        for path, entry in sorted(source_tree.items())
    }
    changed_runtime_files = tuple(
        path
        for path, entry in sorted(source_tree.items())
        if base_tree.get(path) != entry
    )
    changed_mxfp8_python = tuple(
        path
        for path in changed_runtime_files
        if path.endswith(".py") and "mxfp8" in path.lower()
    )
    if not changed_mxfp8_python:
        raise BuildError(
            "source commit has no changed tracked MXFP8 Python files relative "
            f"to upstream base {upstream_base_commit}"
        )

    deleted_runtime_files = tuple(sorted(set(base_tree) - set(source_tree)))
    tactic_json_files = tuple(
        path
        for path in files
        if fnmatch.fnmatch(
            path,
            "vllm/model_executor/kernels/linear/mxfp8/tactic_configs/*.json",
        )
    )
    if not tactic_json_files:
        raise BuildError("source commit has no tracked MXFP8 tactic JSON package data")

    digest = hashlib.sha256()
    for path, member in sorted(files.items()):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(f"{member.mode:o}".encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(member.data).digest())
        digest.update(b"\0")
    return _SourceSnapshot(
        files=files,
        deleted_runtime_files=deleted_runtime_files,
        changed_runtime_files=changed_runtime_files,
        tactic_json_files=tactic_json_files,
        source_tree_sha256=digest.hexdigest(),
    )


def _validate_member_name(name: str) -> None:
    pure_path = PurePosixPath(name)
    if not name or name.startswith("/") or "\\" in name or ".." in pure_path.parts:
        raise BuildError(f"wheel contains unsafe member path: {name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise BuildError(f"wheel member path contains a control character: {name!r}")


def _read_wheel_members(path: Path) -> dict[str, _WheelMember]:
    if not path.is_file():
        raise BuildError(f"wheel does not exist: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise BuildError(f"wheel contains duplicate members: {path}")
            members: dict[str, _WheelMember] = {}
            for info in archive.infolist():
                _validate_member_name(info.filename)
                mode = info.external_attr >> 16
                if mode == 0:
                    mode = 0o100644
                members[info.filename] = _WheelMember(
                    data=archive.read(info.filename),
                    mode=mode,
                    compression=info.compress_type,
                )
    except (OSError, zipfile.BadZipFile) as error:
        raise BuildError(f"invalid wheel archive {path}: {error}") from error
    return members


def _dist_info_dir(members: Mapping[str, _WheelMember]) -> str:
    candidates = {
        name.split("/", maxsplit=1)[0] for name in members if ".dist-info/" in name
    }
    if len(candidates) != 1:
        raise BuildError(
            f"wheel must contain exactly one dist-info directory, found {candidates}"
        )
    return next(iter(candidates))


def _metadata_headers(content: bytes) -> dict[str, str]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BuildError("wheel metadata is not UTF-8") from error
    headers: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            break
        name, separator, value = line.partition(":")
        if separator:
            headers[name.strip().lower()] = value.strip()
    return headers


def _native_members(members: Mapping[str, _WheelMember]) -> dict[str, bytes]:
    return {
        name: member.data
        for name, member in members.items()
        if name.startswith("vllm/") and name.endswith(".so")
    }


def _validate_record(members: Mapping[str, _WheelMember], dist_info: str) -> None:
    record_path = f"{dist_info}/RECORD"
    record = members.get(record_path)
    if record is None:
        raise BuildError("wheel has no RECORD")
    try:
        rows = tuple(csv.reader(io.StringIO(record.data.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as error:
        raise BuildError("wheel RECORD is invalid") from error
    if any(len(row) != 3 for row in rows):
        raise BuildError("wheel RECORD rows must have exactly three fields")
    row_names = tuple(row[0] for row in rows)
    if len(row_names) != len(set(row_names)):
        raise BuildError("wheel RECORD contains duplicate paths")
    if set(row_names) != set(members):
        raise BuildError("wheel RECORD member set does not match archive members")
    for name, digest_field, size_field in rows:
        member = members[name]
        if name == record_path:
            if digest_field or size_field:
                raise BuildError("wheel RECORD self-entry must omit digest and size")
            continue
        expected_digest = (
            base64.urlsafe_b64encode(hashlib.sha256(member.data).digest())
            .decode()
            .rstrip("=")
        )
        if digest_field != f"sha256={expected_digest}":
            raise BuildError(f"wheel RECORD digest mismatch for {name}")
        if size_field != str(len(member.data)):
            raise BuildError(f"wheel RECORD size mismatch for {name}")


def _validate_base_wheel(
    path: Path,
    policy: BuildPolicy,
) -> tuple[dict[str, _WheelMember], str, dict[str, bytes]]:
    if path.name != policy.base_filename:
        raise BuildError(
            "base wheel filename/tag mismatch: "
            f"expected {policy.base_filename}, found {path.name}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != policy.expected_base_sha256:
        raise BuildError(
            "base wheel SHA256 mismatch: "
            f"expected {policy.expected_base_sha256}, found {actual_sha256}"
        )
    members = _read_wheel_members(path)
    dist_info = _dist_info_dir(members)
    metadata_path = f"{dist_info}/METADATA"
    wheel_path = f"{dist_info}/WHEEL"
    if metadata_path not in members or wheel_path not in members:
        raise BuildError("base wheel is missing METADATA or WHEEL")
    metadata = _metadata_headers(members[metadata_path].data)
    if metadata.get("name", "").lower() != policy.distribution:
        raise BuildError("base wheel distribution metadata mismatch")
    if metadata.get("version") != policy.version:
        raise BuildError("base wheel version metadata mismatch")
    wheel_headers = _metadata_headers(members[wheel_path].data)
    if wheel_headers.get("tag") != policy.wheel_metadata_tag:
        raise BuildError(
            "base wheel WHEEL tag mismatch: "
            f"expected {policy.wheel_metadata_tag}, "
            f"found {wheel_headers.get('tag')!r}"
        )
    native_members = _native_members(members)
    if not native_members:
        raise BuildError("base wheel contains no native extension")
    _validate_record(members, dist_info)
    return members, dist_info, native_members


def _with_build_header(content: bytes, build_tag: str) -> bytes:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise BuildError("WHEEL metadata is not UTF-8") from error
    updated = [line for line in lines if line and not line.lower().startswith("build:")]
    updated.append(f"Build: {build_tag}")
    return ("\n".join(updated) + "\n\n").encode()


def _changed_file_hashes(
    snapshot: _SourceSnapshot,
) -> dict[str, str]:
    return {
        path: _sha256_bytes(snapshot.files[path].data)
        for path in snapshot.changed_runtime_files
    }


def _embedded_provenance(
    *,
    source_commit: str,
    base_wheel: Path,
    policy: BuildPolicy,
    build_tag: str,
    snapshot: _SourceSnapshot,
) -> dict[str, object]:
    return {
        "schema_version": _PROVENANCE_SCHEMA_VERSION,
        "distribution": policy.distribution,
        "version": policy.version,
        "source_commit": source_commit,
        "upstream_base_commit": policy.upstream_base_commit,
        "source_tree_sha256": snapshot.source_tree_sha256,
        "changed_runtime_files": _changed_file_hashes(snapshot),
        "tactic_json_files": list(snapshot.tactic_json_files),
        "base_wheel_filename": base_wheel.name,
        "base_wheel_sha256": policy.expected_base_sha256,
        "python_tag": policy.python_tag,
        "abi_tag": policy.abi_tag,
        "platform_tag": policy.platform_tag,
        "wheel_metadata_tag": policy.wheel_metadata_tag,
        "build_tag": build_tag,
    }


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    ).encode()


def _record_member(
    members: Mapping[str, _WheelMember],
    record_path: str,
) -> _WheelMember:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    names = sorted((*members.keys(), record_path))
    for name in names:
        if name == record_path:
            writer.writerow((name, "", ""))
            continue
        member = members[name]
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(member.data).digest())
            .decode()
            .rstrip("=")
        )
        writer.writerow((name, f"sha256={digest}", len(member.data)))
    return _WheelMember(stream.getvalue().encode())


def _write_wheel(
    path: Path,
    members: Mapping[str, _WheelMember],
) -> None:
    try:
        with zipfile.ZipFile(
            path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, member in sorted(members.items()):
                info = zipfile.ZipInfo(name, _FIXED_ZIP_TIME)
                info.compress_type = member.compression
                info.external_attr = member.mode << 16
                info.create_system = 3
                archive.writestr(info, member.data, compresslevel=9)
    except OSError as error:
        raise BuildError(f"cannot write wheel {path}: {error}") from error


def _fsync_file(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise BuildError(f"cannot fsync file {path}: {error}") from error


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise BuildError(f"cannot fsync directory {path}: {error}") from error


def _expected_wheel_filename(
    source_commit: str,
    policy: BuildPolicy,
) -> tuple[str, str]:
    build_tag = f"{_BUILD_TAG_PREFIX}{source_commit[:12]}"
    filename = f"{policy.distribution}-{policy.version}-{build_tag}-{policy.tag}.whl"
    return filename, build_tag


def _expected_embedded_provenance(
    *,
    repo_root: Path,
    source_commit: str,
    base_wheel: Path,
    policy: BuildPolicy,
    build_tag: str,
) -> tuple[_SourceSnapshot, dict[str, object]]:
    snapshot = _load_source_snapshot(
        repo_root,
        source_commit,
        policy.upstream_base_commit,
    )
    provenance = _embedded_provenance(
        source_commit=source_commit,
        base_wheel=base_wheel,
        policy=policy,
        build_tag=build_tag,
        snapshot=snapshot,
    )
    return snapshot, provenance


def validate_custom_wheel(
    wheel: Path,
    *,
    repo_root: Path,
    source_commit: str,
    base_wheel: Path,
    policy: BuildPolicy,
) -> None:
    """Validate source, native ABI, metadata, provenance, and RECORD."""
    expected_filename, build_tag = _expected_wheel_filename(source_commit, policy)
    if wheel.name != expected_filename:
        raise BuildError(
            f"custom wheel filename/tag mismatch: expected {expected_filename}, "
            f"found {wheel.name}"
        )
    base_members, base_dist_info, base_native = _validate_base_wheel(
        base_wheel,
        policy,
    )
    members = _read_wheel_members(wheel)
    dist_info = _dist_info_dir(members)
    if dist_info != base_dist_info:
        raise BuildError("custom wheel changed the base dist-info directory")

    metadata_path = f"{dist_info}/METADATA"
    wheel_path = f"{dist_info}/WHEEL"
    metadata = _metadata_headers(members[metadata_path].data)
    if metadata.get("name", "").lower() != policy.distribution:
        raise BuildError("custom wheel distribution metadata mismatch")
    if metadata.get("version") != policy.version:
        raise BuildError("custom wheel version metadata mismatch")
    wheel_headers = _metadata_headers(members[wheel_path].data)
    if wheel_headers.get("tag") != policy.wheel_metadata_tag:
        raise BuildError("custom wheel did not preserve the native wheel tag")
    if wheel_headers.get("build") != build_tag:
        raise BuildError("custom wheel WHEEL build tag mismatch")

    snapshot, expected_provenance = _expected_embedded_provenance(
        repo_root=repo_root,
        source_commit=source_commit,
        base_wheel=base_wheel,
        policy=policy,
        build_tag=build_tag,
    )
    expected_members = _assemble_custom_members(
        base_members=base_members,
        dist_info=base_dist_info,
        snapshot=snapshot,
        provenance=expected_provenance,
        build_tag=build_tag,
    )
    if set(members) != set(expected_members):
        missing = sorted(set(expected_members) - set(members))
        extra = sorted(set(members) - set(expected_members))
        raise BuildError(
            f"custom wheel member set mismatch: missing={missing}, extra={extra}"
        )

    for path, expected_member in snapshot.files.items():
        member = members.get(path)
        if member != expected_member:
            raise BuildError(
                f"custom wheel member {path} does not match source commit "
                f"{source_commit}"
            )
    for path in snapshot.deleted_runtime_files:
        if path in members:
            raise BuildError(f"custom wheel retained deleted source member {path}")
    for path in snapshot.tactic_json_files:
        if path not in members:
            raise BuildError(f"custom wheel omitted tracked tactic JSON {path}")

    custom_native = _native_members(members)
    if set(custom_native) != set(base_native):
        raise BuildError("custom wheel native extension set differs from base wheel")
    for path, base_content in base_native.items():
        if custom_native[path] != base_content:
            raise BuildError(
                f"custom wheel native extension differs from base wheel: {path}"
            )

    expected_provenance_bytes = _json_bytes(expected_provenance)
    provenance_paths = (
        _PACKAGE_PROVENANCE_PATH,
        f"{dist_info}/mxfp8-provenance.json",
    )
    for path in provenance_paths:
        member = members.get(path)
        if member is None or member.data != expected_provenance_bytes:
            raise BuildError(f"custom wheel provenance mismatch at {path}")

    record_path = f"{dist_info}/RECORD"
    comparison_order = sorted(name for name in members if name != record_path)
    comparison_order.append(record_path)
    for path in comparison_order:
        if members[path] != expected_members[path]:
            raise BuildError(
                f"custom wheel member {path} differs from expected transformation"
            )
    _validate_record(members, dist_info)


def _assemble_custom_members(
    *,
    base_members: Mapping[str, _WheelMember],
    dist_info: str,
    snapshot: _SourceSnapshot,
    provenance: Mapping[str, object],
    build_tag: str,
) -> dict[str, _WheelMember]:
    record_path = f"{dist_info}/RECORD"
    members = {
        name: member
        for name, member in base_members.items()
        if name != record_path and name not in snapshot.deleted_runtime_files
    }
    for path, member in snapshot.files.items():
        members[path] = member
    provenance_member = _WheelMember(_json_bytes(provenance))
    members[_PACKAGE_PROVENANCE_PATH] = provenance_member
    members[f"{dist_info}/mxfp8-provenance.json"] = provenance_member
    wheel_path = f"{dist_info}/WHEEL"
    members[wheel_path] = _WheelMember(
        _with_build_header(members[wheel_path].data, build_tag)
    )
    members[record_path] = _record_member(members, record_path)
    return members


def _publish_create_only(
    temporary_paths: Sequence[Path],
    target_paths: Sequence[Path],
) -> None:
    if len(temporary_paths) != len(target_paths):
        raise BuildError("internal publication path mismatch")
    existing = [str(path) for path in target_paths if path.exists()]
    if existing:
        raise BuildError(f"refusing to overwrite existing artifact: {existing}")
    target_parents = {path.parent.resolve() for path in target_paths}
    if len(target_parents) != 1:
        raise BuildError("all custom wheel artifacts must share one directory")
    target_parent = next(iter(target_parents))
    published: list[Path] = []
    try:
        for temporary, target in zip(temporary_paths, target_paths, strict=True):
            os.link(temporary, target)
            published.append(target)
        _fsync_directory(target_parent)
    except (OSError, BuildError) as error:
        for path in published:
            path.unlink(missing_ok=True)
        with suppress(BuildError):
            _fsync_directory(target_parent)
        if isinstance(error, BuildError):
            raise
        raise BuildError(f"cannot publish custom wheel artifacts: {error}") from error


def build_custom_wheel(request: WheelBuildRequest) -> BuildArtifacts:
    """Build and validate a deterministic custom wheel from an exact Git SHA."""
    repo_root = request.repo_root.resolve()
    base_wheel = request.base_wheel.resolve()
    output_dir = request.output_dir.resolve()
    _validate_host(request.host)
    _validate_external_build_paths(repo_root, base_wheel, output_dir)
    _validate_source_checkout(
        repo_root,
        request.source_commit,
        request.policy.upstream_base_commit,
    )
    base_members, dist_info, _ = _validate_base_wheel(
        base_wheel,
        request.policy,
    )
    snapshot, provenance = _expected_embedded_provenance(
        repo_root=repo_root,
        source_commit=request.source_commit,
        base_wheel=base_wheel,
        policy=request.policy,
        build_tag=f"{_BUILD_TAG_PREFIX}{request.source_commit[:12]}",
    )
    wheel_filename, build_tag = _expected_wheel_filename(
        request.source_commit,
        request.policy,
    )
    members = _assemble_custom_members(
        base_members=base_members,
        dist_info=dist_info,
        snapshot=snapshot,
        provenance=provenance,
        build_tag=build_tag,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    wheel_target = output_dir / wheel_filename
    metadata_target = output_dir / f"{wheel_filename}.metadata.json"
    sha256_target = output_dir / f"{wheel_filename}.sha256"
    targets = (wheel_target, metadata_target, sha256_target)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise BuildError(f"refusing to overwrite existing artifact: {existing}")

    with tempfile.TemporaryDirectory(
        prefix=".mxfp8-wheel-",
        dir=output_dir,
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        temporary_wheel = temporary_root / wheel_filename
        _write_wheel(temporary_wheel, members)
        validate_custom_wheel(
            temporary_wheel,
            repo_root=repo_root,
            source_commit=request.source_commit,
            base_wheel=base_wheel,
            policy=request.policy,
        )
        wheel_sha256 = _sha256_file(temporary_wheel)
        sidecar = dict(provenance)
        sidecar["wheel_filename"] = wheel_filename
        sidecar["wheel_sha256"] = wheel_sha256
        temporary_metadata = temporary_root / metadata_target.name
        temporary_metadata.write_bytes(_json_bytes(sidecar))
        temporary_sha256 = temporary_root / sha256_target.name
        temporary_sha256.write_text(
            f"{wheel_sha256}  {wheel_filename}\n",
            encoding="utf-8",
        )
        for path in (
            temporary_wheel,
            temporary_metadata,
            temporary_sha256,
        ):
            _fsync_file(path)
        _fsync_directory(temporary_root)
        _validate_source_checkout(
            repo_root,
            request.source_commit,
            request.policy.upstream_base_commit,
        )
        _publish_create_only(
            (temporary_wheel, temporary_metadata, temporary_sha256),
            targets,
        )

    return BuildArtifacts(
        wheel=wheel_target,
        metadata=metadata_target,
        sha256=sha256_target,
    )


def resolve_precompiled_wheel(environment: Mapping[str, str]) -> Path:
    """Resolve the explicit official wheel from vLLM's precompiled-build env."""
    if environment.get("VLLM_USE_PRECOMPILED") != "1":
        raise BuildError("set VLLM_USE_PRECOMPILED=1")
    raw_location = environment.get("VLLM_PRECOMPILED_WHEEL_LOCATION", "")
    location = Path(raw_location)
    if not location.is_absolute() or "://" in raw_location:
        raise BuildError(
            "VLLM_PRECOMPILED_WHEEL_LOCATION must be a local absolute path"
        )
    resolved = location.resolve()
    if not resolved.is_file():
        raise BuildError(f"precompiled wheel does not exist: {resolved}")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="clean vLLM Git checkout (default: current directory)",
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="exact full Git SHA to package; must equal clean HEAD",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory or directory without colliding wheel artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed-policy vLLM 0.20.2 aarch64 wheel build."""
    arguments = _parser().parse_args(argv)
    policy = BuildPolicy()
    try:
        base_wheel = resolve_precompiled_wheel(os.environ)
        artifacts = build_custom_wheel(
            WheelBuildRequest(
                repo_root=arguments.repo_root,
                source_commit=arguments.source_commit,
                base_wheel=base_wheel,
                output_dir=arguments.output_dir,
                policy=policy,
                host=HostPlatform(
                    system=platform.system(),
                    machine=platform.machine(),
                ),
            )
        )
    except BuildError as error:
        _parser().error(str(error))
    print(artifacts.wheel)
    print(artifacts.metadata)
    print(artifacts.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
