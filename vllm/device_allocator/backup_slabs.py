# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic bounded CPU-backup layout, independent of CUDA allocation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BackupSlab:
    device: int
    tag: str
    capacity: int
    views: tuple[tuple[int, int, int], ...]


def plan_backup_slabs(
    handles: list[tuple[int, int, int, str]], *, limit: int = 1 << 30
) -> list[BackupSlab]:
    """Pack (pointer, bytes, device, tag) without splitting any handle."""
    if limit <= 0 or limit & (limit - 1):
        raise ValueError("The slab limit must be a positive power of two")
    groups: dict[tuple[int, str], list[tuple[int, int]]] = {}
    seen: set[int] = set()
    for pointer, size, device, tag in handles:
        if size <= 0 or pointer in seen:
            raise ValueError("Handles must have positive sizes and unique pointers")
        seen.add(pointer)
        groups.setdefault((device, tag), []).append((pointer, size))

    result: list[BackupSlab] = []
    for (device, tag), entries in sorted(groups.items()):
        used: list[int] = []
        bins: list[list[tuple[int, int, int]]] = []
        for pointer, size in sorted(entries, key=lambda item: (-item[1], item[0])):
            index = next(
                (index for index, total in enumerate(used) if total + size <= limit),
                len(used),
            )
            if index == len(used):
                used.append(0)
                bins.append([])
            bins[index].append((pointer, used[index], size))
            used[index] += size
        for total, views in zip(used, bins):
            result.append(
                BackupSlab(device, tag, 1 << (total - 1).bit_length(), tuple(views))
            )
    return result
