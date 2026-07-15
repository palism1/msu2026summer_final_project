# Project Rules — check before generating or editing ANY code

Scope: everything under `nba-replay/` (this directory is a self-contained app inside a
host repository; the host's root CLAUDE.md describes an unrelated ML project).
All paths below are relative to `nba-replay/`.

## Rule 1: Frozen type contract (load-bearing)

- `Frame` and `SceneState` (and the rest of the contract: `Play`, `PlayerSnapshot`,
  `CameraPose`, `PlaybackState`, `Vec2`, `Vec3`) are defined ONLY in
  `src/engine/types.ts`. Every module imports them from there. Never redefine,
  extend, or shadow these shapes elsewhere.
- Before writing any code that touches tracking data, re-read `src/engine/types.ts`
  and conform to it.
- Changing `types.ts` is a formal event: it requires (1) explicit user approval in that
  session, (2) a dated **Changed** entry in `DECISIONS.md` with the why, (3) updating
  every consumer in the same commit. Additive optional fields still require (2).

## Rule 2: Two-tier standard

- `src/data`, `src/engine`, `src/cameras`: strict TS, pure functions, **no Three.js
  imports**, Vitest coverage for all math.
- `src/render` and `src/ui` are the ONLY directories that may import Three.js.

Both rules are mechanically enforced by `tests/architecture.test.ts`, which fails the
build if `Frame`/`SceneState` are declared outside `types.ts` or Three.js is imported
outside `render`/`ui`. Never weaken or delete that test to make a change pass; fix the
change instead.

## Rule 3: Repo hygiene

- Every file starts with a FILE MAP header (sections, approximate line ranges, purpose).
  Use `// TWEAK:`, `// CHANGE ME:`, `// DO NOT TOUCH:` tags.
- Conventional Commits. Never include a Co-Authored-By trailer.
- Any add/delete/change of architecture gets a `DECISIONS.md` entry before the commit.

## Before every task

State which of the rules above apply to the files in scope, then proceed.
Verification floor for any change here: `npm run typecheck && npm run lint && npm test`.
