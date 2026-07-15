# DECISIONS.md — nba-replay

Append-only log of design decisions. Newest entries at the top of each dated section.
Format: **Added / Changed / Removed** bullets, each with a one-line *why*.

## Current State (as of 2026-07-15)

- v1 scaffold complete: TypeScript + Vite + Three.js app replaying one stubbed
  SportVU-shaped possession (25fps, 10 players x/y, ball x/y/z) through a first-person
  player-POV camera.
- Frozen data contract in `src/engine/types.ts` (`Frame`, `Play`, `SceneState`,
  `CameraPose`, `PlaybackState`); every module imports from it, none define rivals.
- Two-tier standard: `src/engine` + `src/cameras` + `src/data` are strict, pure, and
  unit-tested with no Three.js (enforced by `tests/tierBoundary.test.ts`);
  `src/render` + `src/ui` are typecheck-only glue and the only Three.js importers.
- CI (GitHub Actions) runs typecheck + Biome + Vitest on every push touching this app.
- Not yet built (deliberately): real SportVU data loader, tactical camera, timeline UI.

## 2026-07-15 — Initial scaffold

### Added

- **App lives at `nba-replay/` inside the host repository instead of its own new repo.**
  Why: the execution environment was scoped to the existing repo with a designated
  feature branch and no repo-creation access; the directory is fully self-contained so
  it can be extracted to a standalone repo later with zero changes.
- **`src/engine/types.ts` created first and frozen as the single data contract.**
  Why: every later feature (real loader, tactical cam, timeline) plugs into these shapes;
  fixing them up front is the anti-backtracking strategy — additive changes only.
- **Court-space convention: feet, x 0–94, y 0–50, z = height; yaw = atan2 radians.**
  Why: matches SportVU raw data directly, so the future real-data loader is a
  pass-through instead of a coordinate transform.
- **Render tier owns the court→Three.js mapping (`courtToWorld`), in exactly one place.**
  Why: nothing outside `src/render` should know Three.js axis conventions; one function
  to change if the mapping is ever wrong.
- **Two-tier standard with a mechanical guard test (`tierBoundary.test.ts`).**
  Why: conventions rot; a failing test doesn't.
- **Facing as yaw (a single angle) with 1D smoothing instead of quaternion slerp.**
  Why: a player's eye camera only yaws (plus a fixed pitch); yaw-space
  exponential-smoothing + rate clamp is mathematically the 1D restriction of a clamped
  slerp, is far easier to test, and keeps Three.js out of the camera tier. Revisit only
  if free-look (roll/pitch dynamics) is ever added.
- **Facing blend done on normalized vectors, then `atan2` — never on raw angles.**
  Why: averaging angles breaks at the ±π seam (e.g. 179° and -179° average to 0°).
- **Velocity via central difference over the interpolated spline, not raw frames.**
  Why: differencing raw 25fps samples stair-steps the facing target 25×/s; differencing
  the spline keeps the camera's input signal as smooth as its output.
- **Dead zone implemented as "trail on the boundary" (move only past the radius).**
  Why: sub-threshold tracking noise must produce exactly zero camera translation, while
  real movement still tracks with at most a fixed offset.
- **Stub data authored as sparse waypoints sampled to 25fps at build time.**
  Why: hand-writing 126 literal frames is unmaintainable; waypoints keep the play
  readable and editable while still emitting genuine SportVU-shaped frames.
- **Defenders derived (mark lerped 35% toward the hoop) rather than hand-authored.**
  Why: reads as plausible man defense for a demo at 1/5th the authoring cost; real data
  replaces it wholesale anyway.
- **npm as package manager.** Why: zero-setup default; nothing here needs workspaces.
- **Biome (single tool) instead of ESLint + Prettier.** Why: one config, one CI step,
  fast; recommended preset is strict enough for this codebase.
- **Vitest with tests in `tests/`, node environment.** Why: shares the Vite config;
  the strict tier is DOM-free by design so no browser environment is needed.
- **CI workflow is path-filtered to `nba-replay/**`.** Why: the host repo's ML project
  pushes shouldn't burn CI minutes on this app (and vice versa).
- **`strict` + `noUncheckedIndexedAccess` in tsconfig for ALL code.** Why: the "loose"
  tier is loose about *testing requirements*, not about types — frame/array indexing
  bugs are exactly what this app would otherwise ship.

### Changed

- **Defender sag 0.25 → 0.35 of the way to the hoop.** Why: at 0.25 the on-ball
  defender's capsule filled half the POV frame; 0.35 keeps the demo readable.

### Removed

- *(nothing yet)*
