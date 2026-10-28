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

"""Peak memory and time of a run, with and without an output_dir.

Not part of the test suite. Builds a synthetic study with many scenarios and
runs it in a fresh process per configuration, so that each peak is its own:

    python tests/e2e/functional/perf_simulation_table_memory.py
    python tests/e2e/functional/perf_simulation_table_memory.py --scenarios 100

With an output_dir, each scenario is written as soon as it is solved (frontal:
one scenario at a time from the solution), so the peak should not grow with
the size of the whole simulation table.
"""

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

_LIBRARY = """\
library:
  id: lib
  models:
    - id: m
      parameters:
        - id: demand
          time-dependent: true
          scenario-dependent: true
        - id: cost
          time-dependent: false
          scenario-dependent: false
      variables:
        - id: x
          lower-bound: 0
      constraints:
        - id: meet_demand
          expression: x >= demand
      extra-outputs:
        - id: spend
          expression: cost * x
      objective-contributions:
        - id: objective
          expression: expec(sum(cost * x))
"""


def make_study(folder: Path, components: int, time_steps: int, scenarios: int) -> Path:
    (folder / "input" / "model-libraries").mkdir(parents=True)
    (folder / "input" / "data-series").mkdir()
    (folder / "input" / "model-libraries" / "lib.yml").write_text(_LIBRARY)
    system = "system:\n  id: s\n  components:\n"
    for c in range(components):
        system += (
            f"    - id: c{c}\n      model: lib.m\n      parameters:\n"
            "        - id: demand\n          time-dependent: true\n"
            "          scenario-dependent: true\n          value: demand_ts\n"
            "        - id: cost\n          time-dependent: false\n"
            f"          scenario-dependent: false\n          value: {c + 1}\n"
        )
    (folder / "input" / "system.yml").write_text(system)
    rows = (
        "\t".join(str((t + s) % 10) for s in range(scenarios))
        for t in range(time_steps)
    )
    (folder / "input" / "data-series" / "demand_ts.tsv").write_text(
        "\n".join(rows) + "\n"
    )
    return folder


def _config(mode: str, time_steps: int, scenarios: int, block: int) -> str:
    resolution = (
        ""
        if mode == "frontal"
        else f"resolution:\n  mode: {mode}\n  block-length: {block}\n"
    )
    return (
        f"time-scope:\n  first-time-step: 0\n  last-time-step: {time_steps - 1}\n"
        "solver-options:\n  name: highs\n  logs: false\n"
        f"scenario-scope:\n  include: [{', '.join(map(str, range(scenarios)))}]\n"
        + resolution
    )


def _measure(study: str, mode: str, output_dir: str) -> None:
    """Run once in this process and print time, peak memory and row count."""
    from gems_craft.optim_config.parsing import load_optim_config
    from gems_craft.study.folder import load_study
    from gems_runner.session.session import SimulationSession

    config = load_optim_config(Path(study) / "input" / f"optim-config-{mode}.yml")
    start = time.perf_counter()
    table = SimulationSession(
        load_study(Path(study)),
        config,
        output_dir=Path(output_dir) if output_dir else None,
    ).run()
    elapsed = time.perf_counter() - start
    rows = sum(len(part) for _, part in table.iter_scenario_frames())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
    print(json.dumps({"time": elapsed, "peak": peak, "rows": rows}))


def _report(args: argparse.Namespace, results: Dict[Tuple[str, bool], Any]) -> None:
    rows = next(iter(results.values()))["rows"]
    print(
        f"\n\nStudy: {args.components} components x {args.time_steps} time steps x "
        f"{args.scenarios} scenarios ({rows:,} result rows)\n"
    )
    print(f"  {'mode':10} {'results':10} {'time':>9} {'peak memory':>12} {'change':>8}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 9} {'-' * 12} {'-' * 8}")
    for mode in ("frontal", "parallel-subproblems"):
        memory, disk = results[(mode, False)], results[(mode, True)]
        saving = f"{disk['peak'] / memory['peak'] - 1:+.0%}"
        for label, r, note in (("in memory", memory, ""), ("on disk", disk, saving)):
            print(
                f"  {mode.split('-')[0]:10} {label:10} {r['time']:7.2f} s "
                f"{r['peak']:9.0f} MB {note:>8}"
            )
    print(
        "\n  Peak memory is the whole process (Python, the problem, the solver and"
        "\n  the results); on disk, the results are not all held in memory at once."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, default=20)
    parser.add_argument("--time-steps", type=int, default=672)
    parser.add_argument("--scenarios", type=int, default=50)
    parser.add_argument("--block", type=int, default=168)
    parser.add_argument("--measure", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.measure:
        _measure(*args.measure)
        return

    results: Dict[Tuple[str, bool], Any] = {}
    print("Running 4 configurations, each in its own process ", end="", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        study = make_study(
            Path(tmp) / "study", args.components, args.time_steps, args.scenarios
        )
        for mode in ("frontal", "parallel-subproblems"):
            (study / "input" / f"optim-config-{mode}.yml").write_text(
                _config(mode, args.time_steps, args.scenarios, args.block)
            )
            for on_disk in (False, True):
                output_dir = str(Path(tmp) / f"out-{mode}") if on_disk else ""
                done = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--measure",
                        str(study),
                        mode,
                        output_dir,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                results[(mode, on_disk)] = json.loads(done.stdout.splitlines()[-1])
                print(".", end="", flush=True)
    _report(args, results)


if __name__ == "__main__":
    main()
