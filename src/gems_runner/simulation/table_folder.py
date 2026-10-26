# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.

"""A simulation table stored on disk: the folder is the table.

::

    <folder>/
    ├── manifest.json            # written last: a folder without it is incomplete
    ├── scenario-common.<ext>    # rows shared by all scenarios, if any
    ├── scenario-0.<ext>
    └── scenario-1.<ext>

Every file has the columns and types of ``SIMULATION_TABLE_SCHEMA``, and keeps
the ``scenario_index`` column (empty in the common file), so the files can also
be read together by any tool.
"""

import json
import operator
import os
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from gems_craft.study.parsing import OutputFormat
from gems_runner.simulation.table_schema import (
    _SCENARIO,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_ROW_GROUP_SIZE,
    SIMULATION_TABLE_SCHEMA,
    _apply_schema,
    _empty_frame,
)
from gems_runner.simulation.table_store import _RowFilter

MANIFEST_FILE = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
COMMON_FILE_STEM = "scenario-common"

_COLUMNS = [
    {"name": field.name, "type": str(field.type)} for field in SIMULATION_TABLE_SCHEMA
]


def _file_stem(scenario: Optional[int]) -> str:
    return COMMON_FILE_STEM if scenario is None else f"scenario-{scenario}"


@dataclass(frozen=True)
class _Manifest:
    """Content of ``manifest.json``."""

    schema_version: int
    run_id: str
    format: str
    scenarios: List[int]
    has_common: bool
    columns: List[Dict[str, str]]

    @property
    def output_format(self) -> OutputFormat:
        return OutputFormat(self.format)

    def file(self, folder: Path, scenario: Optional[int]) -> Path:
        """The file of *scenario*, or the common file for None."""
        return folder / f"{_file_stem(scenario)}.{self.format}"

    def write(self, folder: Path) -> None:
        """Write the manifest atomically, so that it is either absent or
        complete."""
        partial = folder / f"{MANIFEST_FILE}.tmp"
        partial.write_text(json.dumps(asdict(self), indent=2))
        os.replace(partial, folder / MANIFEST_FILE)

    @classmethod
    def read(cls, folder: Path) -> "_Manifest":
        path = folder / MANIFEST_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"{folder} is not a simulation table folder: it has no "
                f"{MANIFEST_FILE} (the run that wrote it may have been interrupted)."
            )
        try:
            content = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: not valid JSON ({error}).") from error
        if not isinstance(content, dict):
            raise ValueError(f"{path}: the manifest is not a JSON object.")
        version = content.get("schema_version")
        if type(version) is not int or version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: unsupported schema_version {version!r}, expected "
                f"{MANIFEST_SCHEMA_VERSION}."
            )
        # Keys unknown here are ignored: a field added later under the same
        # schema_version does not stop this version from reading the folder.
        names = [field.name for field in dataclass_fields(cls)]
        missing = [name for name in names if name not in content]
        if missing:
            raise ValueError(f"{path}: missing {', '.join(missing)}.")
        known = {name: content[name] for name in names}
        checks: Dict[str, Callable[[Any], bool]] = {
            "run_id": lambda v: isinstance(v, str),
            "format": lambda v: v in {f.value for f in OutputFormat},
            "scenarios": lambda v: isinstance(v, list)
            and all(type(s) is int for s in v)
            and v == sorted(set(v)),
            "has_common": lambda v: isinstance(v, bool),
        }
        invalid = [name for name, valid in checks.items() if not valid(known[name])]
        if invalid:
            raise ValueError(f"{path}: invalid {', '.join(invalid)}.")
        if known["columns"] != _COLUMNS:
            raise ValueError(
                f"{path}: the columns do not match the simulation table schema."
            )
        return cls(**known)


class _TableWriter:
    """Writes a simulation table folder, one scenario at a time.

    The folder is created if missing and refused if it is not empty. The
    manifest is written last, by ``finalize()``.
    """

    def __init__(self, folder: Path, run_id: str, output_format: OutputFormat) -> None:
        # Absolute: pyarrow reads a relative path such as "s3:x/..." as a URI.
        folder = Path(folder).resolve()
        output_format = OutputFormat(output_format)  # checked before any write
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            raise FileExistsError(
                f"{folder} is not empty, or is not a folder: a simulation table is "
                "written into a new or empty folder."
            )
        # The folders created here, deepest first, so that abort() removes them.
        self._created_folders = [p for p in [folder, *folder.parents] if not p.exists()]
        folder.mkdir(parents=True, exist_ok=True)
        self._folder = folder
        self._run_id = run_id
        self._format = output_format
        self._scenarios: List[int] = []
        self._has_common = False
        self._finalized = False
        self._written: List[Path] = []

    def write_scenario(self, df: pd.DataFrame, scenario: Optional[int] = None) -> Path:
        """Write the rows of one scenario, not written yet.

        Every row must have the same, non-empty ``scenario_index``. Pass
        *scenario* for a scenario of the run without rows of its own: an empty
        file is written, so that the scenario is still listed in the manifest.
        """
        self._check_open()
        if scenario is not None:
            # A plain int, so that the manifest can be written: a numpy integer
            # is not JSON serializable. A float or a string is refused.
            scenario = operator.index(scenario)
        in_rows = df[_SCENARIO] if _SCENARIO in df.columns else pd.Series(dtype=int)
        labels = {int(s) for s in in_rows.dropna().unique()}
        if in_rows.isna().any() or len(labels) > 1 or (not labels and scenario is None):
            raise ValueError(
                "write_scenario() takes the rows of exactly one scenario, all "
                "with a non-empty scenario_index (or no rows and the scenario)."
            )
        if scenario is None:
            scenario = labels.pop()
        elif labels and labels != {scenario}:
            raise ValueError(
                f"write_scenario(): the rows are of scenario {labels.pop()}, "
                f"not {scenario}."
            )
        if scenario in self._scenarios:
            raise ValueError(f"Scenario {scenario} is already written.")
        path = self._write_file(df, _file_stem(scenario))
        self._scenarios.append(scenario)
        return path

    def write_common(self, df: pd.DataFrame) -> Optional[Path]:
        """Write the rows shared by all scenarios, all with an empty
        ``scenario_index``. Nothing is written for no rows."""
        self._check_open()
        if self._has_common:
            raise ValueError("The rows shared by all scenarios are already written.")
        if df[_SCENARIO].notna().any():
            raise ValueError(
                "write_common() takes rows with an empty scenario_index only."
            )
        if df.empty:
            return None
        self._has_common = True
        return self._write_file(df, COMMON_FILE_STEM)

    def finalize(self) -> _Manifest:
        """Write the manifest, which marks the folder as complete."""
        self._check_open()
        manifest = _Manifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            run_id=self._run_id,
            format=self._format.value,
            scenarios=sorted(self._scenarios),
            has_common=self._has_common,
            columns=_COLUMNS,
        )
        manifest.write(self._folder)
        self._finalized = True
        return manifest

    def abort(self) -> None:
        """Remove what was written, so that the disk is as before: the files, a
        partial manifest, and the folders created here, parents included. No
        write is accepted after it."""
        for path in [*self._written, self._folder / f"{MANIFEST_FILE}.tmp"]:
            path.unlink(missing_ok=True)
        for created in self._created_folders:  # deepest first
            if created.is_dir() and not any(created.iterdir()):
                created.rmdir()
        self._finalized = True

    def _check_open(self) -> None:
        if self._finalized:
            raise ValueError("The simulation table folder is already finalized.")

    def _write_file(self, df: pd.DataFrame, stem: str) -> Path:
        path = self._folder / f"{stem}.{self._format.value}"
        self._written.append(path)  # before writing: a partial file is removed too
        df = _empty_frame() if df.empty else _apply_schema(df)
        if self._format == OutputFormat.CSV:
            df.to_csv(path, index=False)
            return path
        # Pandas metadata is dropped, so the file depends on the data and the
        # schema only. nthreads=1: pyarrow's default converts with a thread
        # pool, started on every call, which is slower for one file per scenario.
        table = pa.Table.from_pandas(
            df, schema=SIMULATION_TABLE_SCHEMA, preserve_index=False, nthreads=1
        ).replace_schema_metadata(None)
        pq.write_table(  # type: ignore[no-untyped-call]
            table,
            path,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            row_group_size=PARQUET_ROW_GROUP_SIZE,
        )
        return path


_PANDAS_TYPES = {pa.int64(): pd.Int64Dtype(), pa.string(): pd.StringDtype("python")}


def _to_frame(table: pa.Table) -> pd.DataFrame:
    """Arrow table, read with the schema, to a DataFrame with the schema's
    columns and dtypes. Each column is converted once, straight to its dtype."""
    return table.to_pandas(types_mapper=_PANDAS_TYPES.get)


class _FolderStore:
    """A simulation table read from its folder, through ``pyarrow.dataset``.

    Only the files of the requested scenarios are opened, plus the common file,
    and the component and output filters are passed down to the scan.
    """

    def __init__(self, folder: Path, manifest: _Manifest) -> None:
        self._folder = folder
        self._manifest = manifest
        self.table_id = manifest.run_id

    @property
    def scenarios(self) -> List[int]:
        return list(self._manifest.scenarios)

    @property
    def path(self) -> Optional[Path]:
        return self._folder

    def _files(self, scenarios: Optional[Any]) -> List[Path]:
        """Files of the requested scenarios (all if None), then the common
        file."""
        selected = (
            self._manifest.scenarios
            if scenarios is None
            else [s for s in self._manifest.scenarios if s in scenarios]
        )
        files = [self._manifest.file(self._folder, s) for s in selected]
        if self._manifest.has_common:
            files.append(self._manifest.file(self._folder, None))
        return files

    def _dataset(self, files: List[Path]) -> ds.Dataset:
        file_format: Any = "parquet"
        if self._manifest.output_format == OutputFormat.CSV:
            # Only an empty cell is missing, as to_csv writes it: pyarrow's
            # default would also read labels such as "NA" or "null" as missing.
            file_format = ds.CsvFileFormat(
                convert_options=pacsv.ConvertOptions(
                    column_types=SIMULATION_TABLE_SCHEMA,
                    strings_can_be_null=True,
                    null_values=[""],
                )
            )
        return ds.dataset(  # type: ignore[no-untyped-call]
            [str(f) for f in files], schema=SIMULATION_TABLE_SCHEMA, format=file_format
        )

    def _read_files(self, files: List[Path], row_filter: _RowFilter) -> pd.DataFrame:
        if not files:
            return _empty_frame()
        table = self._dataset(files).to_table(filter=row_filter.expression())
        return _to_frame(table)

    def read(self, row_filter: _RowFilter) -> pd.DataFrame:
        return self._read_files(self._files(row_filter.scenarios), row_filter)

    def iter_scenarios(
        self, row_filter: _RowFilter
    ) -> Iterator[Tuple[Optional[int], pd.DataFrame]]:
        if self._manifest.has_common:
            common = self._read_files(
                [self._manifest.file(self._folder, None)], row_filter
            )
            if not common.empty:
                yield None, common
        for scenario in self._manifest.scenarios:
            if row_filter.scenarios is None or scenario in row_filter.scenarios:
                file = self._manifest.file(self._folder, scenario)
                yield scenario, self._read_files([file], row_filter)

    def scan(self) -> ds.Dataset:
        return self._dataset(self._files(None))
