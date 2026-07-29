"""CPU-only contract tests for the distributable MXFP8 vLLM wheel."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_MODULE_PATH = _REPO_ROOT / "tools/mxfp8/build_custom_wheel.py"
_SPEC = importlib.util.spec_from_file_location("build_mxfp8_custom_wheel", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load MXFP8 custom wheel builder")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

BuildError = _MODULE.BuildError
BuildPolicy = _MODULE.BuildPolicy
HostPlatform = _MODULE.HostPlatform
WheelBuildRequest = _MODULE.WheelBuildRequest
build_custom_wheel = _MODULE.build_custom_wheel
resolve_precompiled_wheel = _MODULE.resolve_precompiled_wheel
validate_custom_wheel = _MODULE.validate_custom_wheel

_TAG = "cp38-abi3-manylinux_2_35_aarch64"
_OFFICIAL_WHEEL_METADATA_TAG = "cp38-abi3-linux_aarch64"
_DIST_INFO = "vllm-0.20.2.dist-info"
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _initialize_source_repo(repo: Path) -> tuple[str, str]:
    repo.mkdir()
    _run_git(repo, "init", "-b", "main")
    _run_git(repo, "config", "user.name", "Wheel Test")
    _run_git(repo, "config", "user.email", "wheel-test@example.com")
    _write(repo / "vllm/__init__.py", '__version__ = "0.20.2"\n')
    _write(
        repo / "vllm/model_executor/kernels/linear/mxfp8/flashinfer.py",
        'IMPLEMENTATION = "upstream"\n',
    )
    _write(repo / "setup.py", "package_data = {}\n")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "upstream")
    upstream_commit = _run_git(repo, "rev-parse", "HEAD")

    _write(
        repo / "vllm/model_executor/kernels/linear/mxfp8/flashinfer.py",
        'IMPLEMENTATION = "adaptive"\n',
    )
    _write(
        repo / "vllm/model_executor/kernels/linear/mxfp8/tactic_configs/qualified.json",
        '{"schema_version":1}\n',
    )
    _write(repo / "tools/mxfp8/developer_only.py", "DO_NOT_PACKAGE = True\n")
    _write(
        repo / "setup.py",
        'package_data = {"vllm": ['
        '"model_executor/kernels/linear/mxfp8/tactic_configs/*.json"]}\n',
    )
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "custom")
    source_commit = _run_git(repo, "rev-parse", "HEAD")
    return upstream_commit, source_commit


def _record_bytes(members: dict[str, bytes], record_path: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name in sorted(members):
        digest = base64.urlsafe_b64encode(hashlib.sha256(members[name]).digest())
        writer.writerow(
            (
                name,
                f"sha256={digest.decode().rstrip('=')}",
                len(members[name]),
            )
        )
    writer.writerow((record_path, "", ""))
    return stream.getvalue().encode()


def _write_wheel(path: Path, members: dict[str, bytes]) -> None:
    record_path = f"{_DIST_INFO}/RECORD"
    complete_members = dict(members)
    complete_members[record_path] = _record_bytes(members, record_path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(complete_members.items()):
            info = zipfile.ZipInfo(name, _FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)


def _base_members(
    *,
    tag: str = _OFFICIAL_WHEEL_METADATA_TAG,
) -> dict[str, bytes]:
    return {
        "vllm/__init__.py": b'__version__ = "0.20.2"\n',
        "vllm/model_executor/kernels/linear/mxfp8/flashinfer.py": (
            b'IMPLEMENTATION = "upstream"\n'
        ),
        "vllm/_C.abi3.so": b"synthetic native _C",
        "vllm/_moe_C.abi3.so": b"synthetic native _moe_C",
        f"{_DIST_INFO}/METADATA": (
            b"Metadata-Version: 2.4\nName: vllm\nVersion: 0.20.2\n"
        ),
        f"{_DIST_INFO}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: synthetic-test\n"
            "Root-Is-Purelib: false\n"
            f"Tag: {tag}\n"
        ).encode(),
    }


def _make_policy(
    base_wheel: Path,
    upstream_commit: str,
    **changes: Any,
) -> Any:
    policy = BuildPolicy(
        expected_base_sha256=hashlib.sha256(base_wheel.read_bytes()).hexdigest(),
        upstream_base_commit=upstream_commit,
    )
    return replace(policy, **changes)


def _make_request(
    repo: Path,
    source_commit: str,
    base_wheel: Path,
    output_dir: Path,
    policy: Any,
    *,
    host: Any | None = None,
) -> Any:
    return WheelBuildRequest(
        repo_root=repo,
        source_commit=source_commit,
        base_wheel=base_wheel,
        output_dir=output_dir,
        policy=policy,
        host=host or HostPlatform(system="Linux", machine="aarch64"),
    )


def _read_wheel(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _assert_record_valid(members: dict[str, bytes]) -> None:
    record_path = f"{_DIST_INFO}/RECORD"
    rows = list(csv.reader(io.StringIO(members[record_path].decode())))
    assert [row[0] for row in rows] == sorted(members)
    for name, digest_field, size_field in rows:
        if name == record_path:
            assert digest_field == ""
            assert size_field == ""
            continue
        expected_digest = (
            base64.urlsafe_b64encode(hashlib.sha256(members[name]).digest())
            .decode()
            .rstrip("=")
        )
        assert digest_field == f"sha256={expected_digest}"
        assert size_field == str(len(members[name]))


def _rewrite_member_with_valid_record(
    source: Path,
    destination: Path,
    member_name: str,
    content: bytes,
) -> None:
    members = _read_wheel(source)
    members.pop(f"{_DIST_INFO}/RECORD")
    members[member_name] = content
    _write_wheel(destination, members)


def test_build_overlays_tracked_runtime_source_and_preserves_native_tag(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    upstream_commit, source_commit = _initialize_source_repo(repo)
    base_wheel = tmp_path / f"vllm-0.20.2-{_TAG}.whl"
    _write_wheel(base_wheel, _base_members())
    policy = _make_policy(base_wheel, upstream_commit)

    artifacts = build_custom_wheel(
        _make_request(
            repo,
            source_commit,
            base_wheel,
            tmp_path / "dist",
            policy,
        )
    )

    assert artifacts.wheel.name == (
        f"vllm-0.20.2-1mxfp8g{source_commit[:12]}-{_TAG}.whl"
    )
    members = _read_wheel(artifacts.wheel)
    assert (
        members["vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"]
        == b'IMPLEMENTATION = "adaptive"\n'
    )
    assert (
        members[
            "vllm/model_executor/kernels/linear/mxfp8/tactic_configs/qualified.json"
        ]
        == b'{"schema_version":1}\n'
    )
    assert members["vllm/_C.abi3.so"] == b"synthetic native _C"
    assert members["vllm/_moe_C.abi3.so"] == b"synthetic native _moe_C"
    assert "tools/mxfp8/developer_only.py" not in members
    assert b"Version: 0.20.2\n" in members[f"{_DIST_INFO}/METADATA"]
    assert (
        f"Tag: {_OFFICIAL_WHEEL_METADATA_TAG}\n".encode()
        in members[f"{_DIST_INFO}/WHEEL"]
    )
    assert (
        f"Build: 1mxfp8g{source_commit[:12]}\n".encode()
        in members[f"{_DIST_INFO}/WHEEL"]
    )

    package_provenance = json.loads(members["vllm/mxfp8_wheel_provenance.json"])
    dist_info_provenance = json.loads(members[f"{_DIST_INFO}/mxfp8-provenance.json"])
    assert package_provenance == dist_info_provenance
    assert package_provenance["source_commit"] == source_commit
    assert package_provenance["base_wheel_sha256"] == policy.expected_base_sha256
    _assert_record_valid(members)

    sidecar = json.loads(artifacts.metadata.read_text(encoding="utf-8"))
    wheel_sha256 = hashlib.sha256(artifacts.wheel.read_bytes()).hexdigest()
    assert sidecar["wheel_sha256"] == wheel_sha256
    assert sidecar["source_commit"] == source_commit
    assert artifacts.sha256.read_text(encoding="utf-8") == (
        f"{wheel_sha256}  {artifacts.wheel.name}\n"
    )


def test_build_is_byte_deterministic(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    upstream_commit, source_commit = _initialize_source_repo(repo)
    base_wheel = tmp_path / f"vllm-0.20.2-{_TAG}.whl"
    _write_wheel(base_wheel, _base_members())
    policy = _make_policy(base_wheel, upstream_commit)

    first = build_custom_wheel(
        _make_request(repo, source_commit, base_wheel, tmp_path / "dist-a", policy)
    )
    second = build_custom_wheel(
        _make_request(repo, source_commit, base_wheel, tmp_path / "dist-b", policy)
    )

    assert first.wheel.read_bytes() == second.wheel.read_bytes()
    assert first.metadata.read_bytes() == second.metadata.read_bytes()
    assert first.sha256.read_bytes() == second.sha256.read_bytes()


def test_build_preserves_official_filename_and_internal_wheel_tags(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    upstream_commit, source_commit = _initialize_source_repo(repo)
    base_wheel = tmp_path / f"vllm-0.20.2-{_TAG}.whl"
    _write_wheel(
        base_wheel,
        {
            **_base_members(tag=_OFFICIAL_WHEEL_METADATA_TAG),
            f"{_DIST_INFO}/WHEEL": (
                _base_members(tag=_OFFICIAL_WHEEL_METADATA_TAG)[f"{_DIST_INFO}/WHEEL"]
                + b"\n"
            ),
        },
    )
    policy = _make_policy(base_wheel, upstream_commit)

    artifacts = build_custom_wheel(
        _make_request(
            repo,
            source_commit,
            base_wheel,
            tmp_path / "dist",
            policy,
        )
    )

    members = _read_wheel(artifacts.wheel)
    assert artifacts.wheel.name.endswith(f"-{_TAG}.whl")
    assert (
        f"Tag: {_OFFICIAL_WHEEL_METADATA_TAG}\n".encode()
        in members[f"{_DIST_INFO}/WHEEL"]
    )


@pytest.mark.parametrize(
    ("host", "message"),
    [
        (HostPlatform(system="Darwin", machine="arm64"), "Linux"),
        (HostPlatform(system="Linux", machine="x86_64"), "aarch64"),
    ],
)
def test_build_refuses_wrong_host(
    tmp_path: Path,
    host: Any,
    message: str,
) -> None:
    repo = tmp_path / "repo"
    upstream_commit, source_commit = _initialize_source_repo(repo)
    base_wheel = tmp_path / f"vllm-0.20.2-{_TAG}.whl"
    _write_wheel(base_wheel, _base_members())
    policy = _make_policy(base_wheel, upstream_commit)

    with pytest.raises(BuildError, match=message):
        build_custom_wheel(
            _make_request(
                repo,
                source_commit,
                base_wheel,
                tmp_path / "dist",
                policy,
                host=host,
            )
        )


@pytest.mark.parametrize("dirty_path", ["vllm/__init__.py", "untracked.txt"])
def test_build_refuses_dirty_source_tree(
    tmp_path: Path,
    dirty_path: str,
) -> None:
    repo = tmp_path / "repo"
    upstream_commit, source_commit = _initialize_source_repo(repo)
    base_wheel = tmp_path / f"vllm-0.20.2-{_TAG}.whl"
    _write_wheel(base_wheel, _base_members())
    policy = _make_policy(base_wheel, upstream_commit)
    _write(repo / dirty_path, "dirty\n")

    with pytest.raises(BuildError, match="dirty"):
        build_custom_wheel(
            _make_request(
                repo,
                source_commit,
                base_wheel,
                tmp_path / "dist",
                policy,
            )
        )


def test_build_refuses_source_commit_other_than_clean_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    upstream_commit, source_commit = _initialize_source_repo(repo)
    base_wheel = tmp_path / f"vllm-0.20.2-{_TAG}.whl"
    _write_wheel(base_wheel, _base_members())
    policy = _make_policy(base_wheel, upstream_commit)
    assert source_commit != upstream_commit

    with pytest.raises(BuildError, match="HEAD"):
        build_custom_wheel(
            _make_request(
                repo,
                upstream_commit,
                base_wheel,
                tmp_path / "dist",
                policy,
            )
        )


def test_build_refuses_unrelated_upstream_base(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _, source_commit = _initialize_source_repo(repo)
    unrelated_repo = tmp_path / "unrelated"
    _, unrelated_commit = _initialize_source_repo(unrelated_repo)
    base_wheel = tmp_path / f"vllm-0.20.2-{_TAG}.whl"
    _write_wheel(base_wheel, _base_members())
    policy = _make_policy(base_wheel, unrelated_commit)

    with pytest.raises(BuildError, match="upstream base"):
        build_custom_wheel(
            _make_request(
                repo,
                source_commit,
                base_wheel,
                tmp_path / "dist",
                policy,
            )
        )


def test_build_refuses_wrong_base_wheel_sha256(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    upstream_commit, source_commit = _initialize_source_repo(repo)
    base_wheel = tmp_path / f"vllm-0.20.2-{_TAG}.whl"
    _write_wheel(base_wheel, _base_members())
    policy = _make_policy(
        base_wheel,
        upstream_commit,
        expected_base_sha256="0" * 64,
    )

    with pytest.raises(BuildError, match="SHA256"):
        build_custom_wheel(
            _make_request(
                repo,
                source_commit,
                base_wheel,
                tmp_path / "dist",
                policy,
            )
        )


def test_build_refuses_base_wheel_with_incompatible_tag(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    upstream_commit, source_commit = _initialize_source_repo(repo)
    wrong_tag = "cp310-cp310-manylinux_2_35_aarch64"
    base_wheel = tmp_path / f"vllm-0.20.2-{wrong_tag}.whl"
    _write_wheel(base_wheel, _base_members(tag=wrong_tag))
    policy = _make_policy(base_wheel, upstream_commit)

    with pytest.raises(BuildError, match="tag"):
        build_custom_wheel(
            _make_request(
                repo,
                source_commit,
                base_wheel,
                tmp_path / "dist",
                policy,
            )
        )


def test_build_refuses_base_wheel_without_native_extensions(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    upstream_commit, source_commit = _initialize_source_repo(repo)
    base_wheel = tmp_path / f"vllm-0.20.2-{_TAG}.whl"
    members = _base_members()
    del members["vllm/_C.abi3.so"]
    del members["vllm/_moe_C.abi3.so"]
    _write_wheel(base_wheel, members)
    policy = _make_policy(base_wheel, upstream_commit)

    with pytest.raises(BuildError, match="native extension"):
        build_custom_wheel(
            _make_request(
                repo,
                source_commit,
                base_wheel,
                tmp_path / "dist",
                policy,
            )
        )


def test_validation_rejects_source_overlay_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    upstream_commit, source_commit = _initialize_source_repo(repo)
    base_wheel = tmp_path / f"vllm-0.20.2-{_TAG}.whl"
    _write_wheel(base_wheel, _base_members())
    policy = _make_policy(base_wheel, upstream_commit)
    artifacts = build_custom_wheel(
        _make_request(
            repo,
            source_commit,
            base_wheel,
            tmp_path / "dist",
            policy,
        )
    )
    tampered_wheel = tmp_path / artifacts.wheel.name
    _rewrite_member_with_valid_record(
        artifacts.wheel,
        tampered_wheel,
        "vllm/model_executor/kernels/linear/mxfp8/flashinfer.py",
        b'IMPLEMENTATION = "tampered"\n',
    )

    with pytest.raises(BuildError, match="source commit"):
        validate_custom_wheel(
            tampered_wheel,
            repo_root=repo,
            source_commit=source_commit,
            base_wheel=base_wheel,
            policy=policy,
        )


def test_resolve_precompiled_wheel_requires_official_environment_contract(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / f"vllm-0.20.2-{_TAG}.whl"
    wheel.touch()

    assert (
        resolve_precompiled_wheel(
            {
                "VLLM_USE_PRECOMPILED": "1",
                "VLLM_PRECOMPILED_WHEEL_LOCATION": str(wheel),
            }
        )
        == wheel.resolve()
    )
    with pytest.raises(BuildError, match="VLLM_USE_PRECOMPILED=1"):
        resolve_precompiled_wheel({"VLLM_PRECOMPILED_WHEEL_LOCATION": str(wheel)})
    with pytest.raises(BuildError, match="local absolute path"):
        resolve_precompiled_wheel(
            {
                "VLLM_USE_PRECOMPILED": "1",
                "VLLM_PRECOMPILED_WHEEL_LOCATION": "https://example.com/vllm.whl",
            }
        )


def test_setup_declares_mxfp8_tactic_json_as_package_data() -> None:
    setup_source = (_REPO_ROOT / "setup.py").read_text(encoding="utf-8")

    assert '"model_executor/kernels/linear/mxfp8/tactic_configs/*.json"' in setup_source
