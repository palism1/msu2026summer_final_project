# nba-replay — NBA 3D Replay Viewer

First-person **player-POV camera** replaying a basketball possession in 3D, from tracking
data shaped like 2015-16 NBA SportVU exports (25fps, 10 players x/y, ball x/y/z).
TypeScript + Vite + Three.js. v1 plays one stubbed possession; the module layout is
designed so a real SportVU loader, a tactical (top-down) camera, and a timeline UI slot
in without restructuring.

> This app lives as a self-contained directory inside a host repository; everything below
> assumes you are in `nba-replay/`. It shares nothing with the host project.

## Run it

```bash
npm install
npm run dev        # dev server; open the printed URL
```

You should see a court, 10 capsules + a ball running a 5-second play on loop, viewed
through the shooting guard's eyes. **Space** pauses/resumes.

```bash
npm run typecheck  # strict tsc over everything
npm run lint       # Biome (format + lint)
npm test           # Vitest (engine + camera math + data invariants)
npm run build      # production build
```

## Architecture: two tiers, one frozen contract

```
src/engine/types.ts        <- THE data contract. Everything imports shapes from here.
src/engine/interpolate.ts  <- Catmull-Rom sampling: 25fps frames -> smooth SceneState
src/engine/playback.ts     <- pure playback clock (play/pause/speed/loop)
src/data/stubPlay.ts       <- hand-authored SportVU-shaped possession (future: real loader)
src/cameras/povCamera.ts   <- all POV camera math as pure functions
src/render/scene.ts        <- Three.js: court, capsules, ball, camera mapping
src/ui/main.ts             <- entry point: DOM + requestAnimationFrame loop
```

| Tier | Dirs | Rules |
| --- | --- | --- |
| **Load-bearing** | `src/engine`, `src/cameras`, `src/data` | Strict TS, pure functions, Vitest unit tests, **no Three.js imports** (mechanically enforced by `tests/architecture.test.ts`) |
| **Loose** | `src/render`, `src/ui` | Typecheck-only glue; the **only** places allowed to import Three.js |

`src/engine/types.ts` is the anti-backtracking anchor: `Frame`/`Play` (raw samples),
`SceneState` (interpolated world), `CameraPose`, `PlaybackState`. Additive changes are
fine; renames/removals require a `DECISIONS.md` entry. No module may define its own
competing data shapes — `tests/architecture.test.ts` fails the build if a contract type
is declared anywhere else. Project rules live in [`CLAUDE.md`](CLAUDE.md).

## POV camera math (src/cameras/povCamera.ts)

Each behavior is a separately tested pure function:

- **`eyePosition`** — court (x, y) lifted to eye height.
- **`catmullRom*`** (engine) — spline through the 25fps samples so motion glides instead
  of stair-stepping.
- **`playerVelocity`** — central difference over the *interpolated* track.
- **`facingYaw`** — facing = weighted blend of velocity direction and ball direction
  (blended as vectors, then `atan2`, so the ±π seam can't break it).
- **`smoothYaw`** — exponential smoothing (frame-rate independent) + hard rotation-speed
  clamp: sudden target flips become bounded pans, never snaps.
- **`applyDeadZone`** — camera position freezes for sub-threshold movement; zero shimmer.
- **`updatePovCamera`** — composes the above: prev pose → next pose.

## Comment conventions

Every file opens with a `FILE MAP` header (sections, approximate line ranges, purpose).
Inline tags: `// TWEAK:` safe to adjust · `// CHANGE ME:` must set before running
(none needed for the stub demo) · `// DO NOT TOUCH:` load-bearing, reason given inline.

## What's next (designed-for, not built)

- Real SportVU loader: emit `Play` from actual game JSON; `stubPlay.ts` is the template.
- Tactical camera: new module in `src/cameras/` returning the same `CameraPose`.
- Timeline UI: drive `PlaybackState` via `seek`/`setSpeed` from DOM controls.
