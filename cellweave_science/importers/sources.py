"""Internal scientific import source set abstraction.

Represents a trusted bundle of verified local source files and their assigned
scientific roles for consumption by DatasetImporter implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cellweave_science.contracts import ImportFileRole, SourceFormat


@dataclass(frozen=True)
class ImportSourceFile:
    """A single verified participating source file in local trusted storage."""

    dataset_file_id: str
    role: ImportFileRole
    local_path: Path
    ordinal: int = 0


@dataclass(frozen=True)
class ImportSourceSet:
    """Group of participating source files bound to an import execution."""

    source_format: SourceFormat
    files: tuple[ImportSourceFile, ...]

    def get_primary_file(self) -> ImportSourceFile:
        """Retrieve the single PRIMARY file for single-file formats."""
        primaries = [f for f in self.files if f.role == ImportFileRole.PRIMARY]
        if len(primaries) != 1:
            raise ValueError(f"Expected exactly 1 PRIMARY file, found {len(primaries)}")
        return primaries[0]

    def get_files_by_role(self, role: ImportFileRole) -> list[ImportSourceFile]:
        """Retrieve all participating files assigned to a specific role, sorted by ordinal."""
        return sorted([f for f in self.files if f.role == role], key=lambda x: x.ordinal)

    def get_single_file_by_role(self, role: ImportFileRole) -> ImportSourceFile:
        """Retrieve exactly one participating file assigned to a specific role."""
        matching = self.get_files_by_role(role)
        if len(matching) != 1:
            raise ValueError(f"Expected exactly 1 file for role {role}, found {len(matching)}")
        return matching[0]


__all__ = ["ImportSourceFile", "ImportSourceSet"]
