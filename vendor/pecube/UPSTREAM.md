# Pecube Vendor Source

- Upstream repository: https://github.com/jeanbraun/Pecube
- Upstream commit: `cf100c1fc542773ab829ce88ae600da18841f612`
- License: GNU General Public License v3.0, copyright 2003-2018 Jean Braun
- Vendored path: `vendor/pecube/source`

This directory contains a source snapshot of Pecube for use as the bundled
thermo-kinematic engine in GA-LEM-Inverter. The upstream source is kept separate
from GA-LEM-Inverter Python code. Build products are written to
`vendor/pecube/bin` and runtime Pecube projects are written to
`vendor/pecube/projects` or a run-local `pecube/PGB01` directory.

Local changes:

- None at vendoring time. Integration code is kept in `ga_lem_inverter/`.
