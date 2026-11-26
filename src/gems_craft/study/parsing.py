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

import argparse
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from pydantic import Field, ValidationError, field_validator, model_validator
from yaml import safe_dump, safe_load

from gems_craft.utils import ModifiedBaseModel


class PortConnectionsSchema(ModifiedBaseModel):
    component1: str
    port1: str
    component2: str
    port2: str


class ComponentParameterSchema(ModifiedBaseModel):
    id: str
    time_dependent: bool = False
    scenario_dependent: bool = False
    indexed_by: List[str] = Field(default_factory=list)
    # float: constant; str: series name; dict: inline set-indexed values (nested per set)
    value: Union[float, str, Dict[Union[str, int], Any]]
    scenario_group: Optional[str] = None


class ComponentPropertySchema(ModifiedBaseModel):
    id: str
    value: str


class IntegerStrategyId(str, Enum):
    EXACT = "exact"
    RELAXED = "relaxed"
    HEURISTIC = "heuristic"


class HeuristicId(str, Enum):
    FAST = "fast"
    ACCURATE = "accurate"


class OutputFormat(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"


class IntegerStrategy(ModifiedBaseModel):
    id: IntegerStrategyId = IntegerStrategyId.EXACT
    heuristic_id: Optional[HeuristicId] = None

    @model_validator(mode="after")
    def _heuristic_id_consistency(self) -> "IntegerStrategy":
        if self.id == IntegerStrategyId.HEURISTIC and self.heuristic_id is None:
            raise ValueError(
                "'heuristic-id' is required when integer-strategy is 'heuristic'"
            )
        if self.id != IntegerStrategyId.HEURISTIC and self.heuristic_id is not None:
            raise ValueError(
                f"'heuristic-id' is only allowed when integer-strategy is 'heuristic', "
                f"got integer-strategy '{self.id.value}'"
            )
        return self


class SetTreeNodeSchema(ModifiedBaseModel):
    """An element of a tree set and the subtrees of its children."""

    node: Union[str, int]
    children: List["SetTreeNodeSchema"] = Field(default_factory=list)


class SetInstanceSchema(ModifiedBaseModel):
    id: str
    # A list, an "a..b" range, or the tree of a tree set (from its root)
    elements: Union[List[Union[str, int]], str, SetTreeNodeSchema]

    @field_validator("elements", mode="after")
    @classmethod
    def _expand_range(
        cls, v: Union[List[Union[str, int]], str, SetTreeNodeSchema]
    ) -> Union[List[Union[str, int]], str, SetTreeNodeSchema]:
        if isinstance(v, str):
            m = re.fullmatch(r"(\d+)\.\.(\d+)", v.strip())
            if not m:
                raise ValueError(
                    f"Invalid elements range {v!r}: expected 'a..b' (e.g. '0..9')"
                )
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                raise ValueError(f"Range start must be <= end, got {v!r}")
            return list(range(a, b + 1))
        return v

    def _tree_walk(self) -> List[Tuple[Union[str, int], int]]:
        """(element, parent position) of a tree set, in pre-order (root first)."""
        assert isinstance(self.elements, SetTreeNodeSchema)
        result: List[Tuple[Union[str, int], int]] = []
        stack = [(self.elements, -1)]
        while stack:
            node, parent = stack.pop()
            position = len(result)
            result.append((node.node, parent))
            stack.extend((child, position) for child in reversed(node.children))
        return result

    def element_list(self) -> List[Union[str, int]]:
        """The elements, a tree being listed in pre-order (root first)."""
        if isinstance(self.elements, SetTreeNodeSchema):
            return [element for element, _ in self._tree_walk()]
        assert isinstance(self.elements, list)  # ranges are expanded at parsing
        return list(self.elements)

    def parent_positions(self) -> Optional[Tuple[int, ...]]:
        """Position of each element's parent (-1 for the root), or None if the
        elements are not a tree."""
        if not isinstance(self.elements, SetTreeNodeSchema):
            return None
        return tuple(parent for _, parent in self._tree_walk())


class ComponentSchema(ModifiedBaseModel):
    id: str
    model: str
    scenario_group: Optional[str] = None
    parameters: Optional[List[ComponentParameterSchema]] = None
    properties: Optional[List[ComponentPropertySchema]] = None
    integer_strategy: IntegerStrategy = Field(default_factory=IntegerStrategy)
    sets: Optional[List[SetInstanceSchema]] = None


class SystemSchema(ModifiedBaseModel):
    id: Optional[str] = None
    model_libraries: Optional[str] = None  # Parsed but unused for now
    components: List[ComponentSchema] = Field(default_factory=list)
    connections: Optional[List[PortConnectionsSchema]] = None
    sets: Optional[List[SetInstanceSchema]] = None


_S = TypeVar("_S", bound=SystemSchema)


@overload
def load_input_system(input_study: Path) -> SystemSchema: ...
@overload
def load_input_system(input_study: Path, schema: Type[_S]) -> _S: ...
def load_input_system(
    input_study: Path, schema: Type[SystemSchema] = SystemSchema
) -> SystemSchema:
    try:
        with input_study.open() as f:
            return schema.model_validate(safe_load(f))
    except ValidationError as e:
        raise ValueError(f"An error occurred during parsing: {e}")


@overload
def parse_yaml_system(input_study: TextIO) -> SystemSchema: ...
@overload
def parse_yaml_system(input_study: TextIO, schema: Type[_S]) -> _S: ...
def parse_yaml_system(
    input_study: TextIO, schema: Type[SystemSchema] = SystemSchema
) -> SystemSchema:
    tree = safe_load(input_study)
    return schema.model_validate(tree["system"])


def write_yaml_system(system: SystemSchema, path: Path) -> None:
    data = {"system": system.model_dump(by_alias=True, exclude_none=True, mode="json")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        safe_dump(data, f, allow_unicode=True, sort_keys=False)


@dataclass(frozen=True)
class ParsedArguments:
    study_dir: Path
    optim_config_path: Optional[Path] = None
    output_format: OutputFormat = OutputFormat.PARQUET


def parse_cli() -> ParsedArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        type=Path,
        required=True,
        help="path to the root directory of the study",
    )
    parser.add_argument(
        "--optim-config",
        type=Path,
        default=None,
        dest="optim_config",
        help="optional custom path to optim-config.yml (defaults to study_dir/input/optim-config.yml)",
    )
    parser.add_argument(
        "--output-format",
        choices=[output_format.value for output_format in OutputFormat],
        default=OutputFormat.PARQUET.value,
        dest="output_format",
        help="format of the simulation table files (default: parquet)",
    )

    args = parser.parse_args()
    return ParsedArguments(
        study_dir=args.study,
        optim_config_path=args.optim_config,
        output_format=OutputFormat(args.output_format),
    )
