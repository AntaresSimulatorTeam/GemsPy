# Bundled GEMS input schemas

These Draft 2020-12 schemas are snapshotted from
[GEMS PR #284](https://github.com/AntaresSimulatorTeam/GEMS/pull/284) at commit
`4498d940d28878067ce5cc586bc818feb8f665f9`. The snapshot contains the shared definitions and
the schemas consumed by GemsPy's model-library, system, and optimization-config parsers.
`system-for-view.schema.yml` is not included because GemsPy does not parse that file type.

The schemas are drafts; `optim-config.schema.yml` is explicitly provisional. Runtime validation
requires the shared GEMS schema to pass; it does not use a permissive schema union with the
Pydantic schema. Pydantic then supplies typed parsing and additional business-rule validation.
If GemsPy needs to accept a feature the shared schema does not yet describe, add a narrowly
scoped extension to the relevant schema and its tests. When the GEMS schemas are revised or
released, update this snapshot and its conformance tests together rather than silently relaxing
the shared contract locally.

The schema snapshot intentionally follows GemsPy on documented discrepancies that affect parser
compatibility: component parameter values may be numeric regardless of dependency flags
(`system.schema.yml`, S05), and hybrid area-connection members are optional
(`model-library.schema.yml`, L07).
