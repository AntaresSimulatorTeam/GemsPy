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

"""
The network module defines the data model for an instance of network,
including nodes, links, and components (model instantations).
"""
import itertools
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, cast

from gems.model import PortField, PortType
from gems.model.model import Model
from gems.model.port import PortFieldId
from gems.utils import require_not_none


@dataclass(frozen=True)
class Component:
    """
    A component is an instance of a model, with specified parameter values.
    """

    model: Model
    id: str

    def is_variable_in_model(self, var_id: str) -> bool:
        return var_id in self.model.variables.keys()

    def replicate(self, /, **changes: Any) -> "Component":
        return replace(self, **changes)


def create_component(model: Model, id: str) -> Component:
    return Component(model=model, id=id)


@dataclass(frozen=True)
class Node(Component):
    """
    A node in the network.
    """

    pass


def create_node(model: Model, id: str) -> Node:
    return Node(model=model, id=id)


@dataclass(frozen=True)
class PortRef:
    component: Component
    port_id: str


@dataclass()
class PortsConnection:
    port1: PortRef
    port2: PortRef
    master_port: Dict[PortField, PortRef] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.__validate_ports()

    def __validate_ports(self) -> None:
        model1 = self.port1.component.model
        model2 = self.port2.component.model
        port_1 = model1.ports.get(self.port1.port_id)
        port_2 = model2.ports.get(self.port2.port_id)

        if port_1 is None or port_2 is None:
            raise ValueError(f"Missing port: {port_1} or {port_2} ")
        if port_1.port_type != port_2.port_type:
            raise ValueError(
                f"Incompatible portTypes {port_1.port_type} != {port_2.port_type}"
            )

        for field_name in [f.name for f in port_1.port_type.fields]:
            def1: bool = (
                PortFieldId(port_name=port_1.port_name, field_name=field_name)
                in model1.port_fields_definitions
            )
            def2: bool = (
                PortFieldId(port_name=port_2.port_name, field_name=field_name)
                in model2.port_fields_definitions
            )
            if not def1 and not def2:
                raise ValueError(
                    f"No definition for port field {field_name} on {port_1.port_name}."
                )
            if def1 and def2:
                raise ValueError(
                    f"Port field {field_name} on {port_1.port_name} has 2 definitions."
                )

            self.master_port[PortField(name=field_name)] = (
                self.port1 if def1 else self.port2
            )

    def get_port_type(self) -> PortType:
        port_1 = self.port1.component.model.ports.get(self.port1.port_id)

        if port_1 is None:
            raise ValueError(f"Missing port: {port_1}")
        return port_1.port_type

    def replicate(self, /, **changes: Any) -> "PortsConnection":
        # Shallow copy
        return replace(self, **changes)


@dataclass
class Network:
    """
    Network model: simply nodes, links, and components.
    """

    id: str
    _nodes: Dict[str, Node] = field(init=False, default_factory=dict)
    _components: Dict[str, Component] = field(init=False, default_factory=dict)
    _connections: List[PortsConnection] = field(init=False, default_factory=list)

    def _check_node_exists(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} does not exist in the network.")

    def add_component(self, component: Component) -> None:
        require_not_none(component)
        self._components[component.id] = component

    def get_component(self, component_id: str) -> Component:
        """
        Returns the component (possibly a node) corresponding to this ID.
        """
        res = self._components.get(component_id, None)
        return res if res else self._nodes[component_id]

    @property
    def components(self) -> Iterable[Component]:
        return self._components.values()

    def add_node(self, node: Node) -> None:
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> Node:
        return self._nodes[node_id]

    @property
    def nodes(self) -> Iterable[Node]:
        return self._nodes.values()

    @property
    def all_components(self) -> Iterable[Component]:
        """
        An iterable over both nodes and components.
        """
        return itertools.chain(self.nodes, self.components)

    def connect(self, port1: PortRef, port2: PortRef) -> None:
        ports_connection = PortsConnection(port1, port2)
        self._connections.append(ports_connection)

    @property
    def connections(self) -> Iterable[PortsConnection]:
        return self._connections

    def get_connection(self, idx: int) -> PortsConnection:
        return self._connections[idx]

    def is_empty(self) -> bool:
        return (not self._nodes) and (not self._components) and (not self._connections)

    def replicate(self, /, **changes: Any) -> "Network":
        replica = replace(self, **changes)

        for node in self.nodes:
            replica.add_node(cast(Node, node.replicate()))

        for component in self.components:
            replica.add_component(component.replicate())

        for connection in self.connections:
            replica._connections.append(connection.replicate())

        return replica
