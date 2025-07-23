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

import pytest

from gems.simulation import TimeBlock
from gems.simulation.decision_tree import (
    DecisionTreeNode,
    InterDecisionTimeScenarioConfig,
)
from gems.study.network import Network


def test_decision_tree_generation() -> None:
    scenarios = 1
    blocks = [TimeBlock(1, [0])]
    config = InterDecisionTimeScenarioConfig(blocks, scenarios)

    network = Network("network_id")
    root = DecisionTreeNode("root", config, network)

    assert root.id == "root"
    assert root.parent is None
    assert root.prob == 1.0
    assert not root.children  # No children

    child = DecisionTreeNode("child", config, network, parent=root, prob=0.8)

    assert child.parent == root
    assert child.prob == 0.8
    assert child in root.children

    grandchild = DecisionTreeNode("grandchild", config, network, parent=child, prob=0.6)

    assert grandchild.parent == child
    assert grandchild.prob == (0.8 * 0.6)
    assert (grandchild not in root.children) and (grandchild in child.children)

    with pytest.raises(ValueError, match="Probability must be a value in the range"):
        great_grandchild = DecisionTreeNode(
            "greatgrandchild", config, network, parent=grandchild, prob=2.0
        )

    with pytest.raises(ValueError, match="Probability must be a value in the range"):
        great_grandchild = DecisionTreeNode(
            "greatgrandchild", config, network, parent=grandchild, prob=-0.3
        )


def test_decision_tree_probabilities() -> None:
    scenarios = 1
    blocks = [TimeBlock(1, [0])]
    config = InterDecisionTimeScenarioConfig(blocks, scenarios)
    network = Network("network_id")

    """
    root (p = 1)
      |- l_child (p = 0.7)
      |     |- ll_child (p = 0.5)
      |     |      `- lll_child (p = 1)
      |     |
      |     `- lr_child (p = 0.5)
      |
      `- r_child (p = 0.3)
            |- rl_child (p = 0.4)
            `- rr_child (p = 0.5)
    """

    # Root
    root = DecisionTreeNode("root", config, network)

    # 1st level
    l_child = DecisionTreeNode("l_child", config, network, parent=root, prob=0.7)
    r_child = DecisionTreeNode("r_child", config, network, parent=root, prob=0.3)

    # 2nd level
    ll_child = DecisionTreeNode("ll_child", config, network, parent=l_child, prob=0.5)
    lr_child = DecisionTreeNode("lr_child", config, network, parent=l_child, prob=0.5)

    rl_child = DecisionTreeNode("rl_child", config, network, parent=r_child, prob=0.4)
    rr_child = DecisionTreeNode("rr_child", config, network, parent=r_child, prob=0.5)

    # 3rd level
    lll_child = DecisionTreeNode("lll_child", config, network, parent=ll_child, prob=1)

    assert ll_child.is_leaves_prob_sum_one()  # One child with p = 1

    assert l_child.is_leaves_prob_sum_one()  # Two children w/ p1 = p2 = 0.5

    assert not r_child.is_leaves_prob_sum_one()  # Two children w/ p1 + p2 != 1

    assert not root.is_leaves_prob_sum_one()
