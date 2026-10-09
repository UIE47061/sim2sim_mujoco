# WTW standalone integration provenance

- The command/phase/observation/control contract was ported from the local independent `wtw_mjlab` implementation (reference contract SHA-256: `f82461bced659f12ec2c07db128fc1c9db431e7e9f4950f5f96510950da0abcb`). The deployed policy includes history → adaptation → actor. Neither mjlab nor the training checkout is an execution dependency here.
- Gait/algorithm reference: Improbable AI Lab, Walk These Ways, revision `0e7236bdc81ce855cbe3d70345a7899452bdeb1c`, MIT License. Original license retained in `walk_these_ways.LICENSE`.
- Little White v3 XML/meshes: https://github.com/morrisx28/csl_mujoco/tree/0ab74ef5d6048345db7a541316019100416b57d6/unitree_robots/little_white_v3 . BSD 3-Clause, copyright 2016–2024 Unitree Robotics. License retained in `little_white_v3.LICENSE` and `assets/little_white_v3_wtw/upstream/LICENSE`.
- The copied asset manifest preserves the source URL, revision, transform version 3, and SHA-256 for all original and derived files. Derived collision/control settings are explicit; meshes remain relative paths.
- Each WTW export must include its paired `metadata.json`, which records policy/checkpoint hashes and the observation/control contract. Exports and machine-specific provenance are excluded from Git and must be supplied separately after cloning. Historical source paths in metadata are not execution dependencies.
