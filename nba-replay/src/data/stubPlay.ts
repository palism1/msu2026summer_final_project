// FILE MAP
// ~L1-20    header + imports + court constants
// ~L21-55   waypoint helpers (piecewise-linear path sampling, lerp, parabolic arc)
// ~L56-115  offensive player paths (hand-authored waypoints for a wing pick-and-pop play)
// ~L116-170 ball script: dribble -> pass -> hold -> jump shot -> made basket
// ~L171-215 buildStubPlay: assembles 25fps Frames (offense from paths, defense derived)
// Purpose: stand-in for real SportVU tracking. Same shape as 2015-16 SportVU exports
// (25fps, 10 players x/y, ball x/y/z) so the future real-data loader is a drop-in swap.
// Load-bearing tier: strict TS, no Three.js, invariants tested in tests/stubPlay.test.ts.

import type { Frame, Play, PlayerSnapshot, Vec2, Vec3 } from "../engine/types";

const FPS = 25; // DO NOT TOUCH: SportVU sample rate; the whole pipeline is shaped around it.
const DURATION_S = 5;
const COURT_LENGTH_FT = 94;
const COURT_WIDTH_FT = 50;
/** Attacking basket: rim center is 5.25ft from the right baseline, mid-court width. */
const HOOP: Vec2 = { x: COURT_LENGTH_FT - 5.25, y: 25 };
const RIM_HEIGHT_FT = 10;

/** A hand-authored path key: be at (x, y) at time t (seconds). */
interface Waypoint {
  t: number;
  x: number;
  y: number;
}

function lerp(a: number, b: number, u: number): number {
  return a + (b - a) * u;
}

/** Piecewise-linear sample of a waypoint path at time t (clamped to the path's span). */
function samplePath(path: readonly Waypoint[], t: number): Vec2 {
  const first = path[0];
  const last = path[path.length - 1];
  if (first === undefined || last === undefined) {
    throw new Error("samplePath: empty path");
  }
  if (t <= first.t) return { x: first.x, y: first.y };
  if (t >= last.t) return { x: last.x, y: last.y };
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i];
    const b = path[i + 1];
    if (a !== undefined && b !== undefined && t >= a.t && t <= b.t) {
      const u = (t - a.t) / (b.t - a.t);
      return { x: lerp(a.x, b.x, u), y: lerp(a.y, b.y, u) };
    }
  }
  return { x: last.x, y: last.y };
}

/** Height of a ballistic arc from z0 to z1 peaking at `peak`, u in [0,1]. */
function arcHeight(z0: number, z1: number, peak: number, u: number): number {
  // Quadratic through (0, z0), (0.5, peak), (1, z1).
  const a = 2 * z0 + 2 * z1 - 4 * peak;
  const b = -3 * z0 - z1 + 4 * peak;
  return a * u * u + b * u + z0;
}

// --- Offensive paths (team "home", attacking +x toward HOOP) ------------------------
// TWEAK: these waypoints ARE the play. Reshape them freely; the pipeline only needs
// each path to stay inside the court and to span t=0..5.
// The play: PG walks the ball up and hits the wing; the wing takes one dribble into a
// mid-range jumper at t=3.6 while the bigs crash and the corner shooter spaces.

const OFFENSE_PATHS: Record<string, readonly Waypoint[]> = {
  "home-pg": [
    { t: 0.0, x: 58, y: 25 },
    { t: 2.0, x: 70, y: 27 },
    { t: 3.0, x: 68, y: 32 },
    { t: 5.0, x: 66, y: 34 },
  ],
  // The shooter — and the default POV player (see src/ui/main.ts).
  "home-sg": [
    { t: 0.0, x: 68, y: 44 },
    { t: 1.6, x: 74, y: 41 },
    { t: 2.4, x: 76, y: 38 },
    { t: 3.6, x: 78, y: 35 },
    { t: 4.2, x: 78.5, y: 34.5 },
    { t: 5.0, x: 79, y: 34 },
  ],
  "home-sf": [
    { t: 0.0, x: 84, y: 6 },
    { t: 2.5, x: 86, y: 8 },
    { t: 5.0, x: 85, y: 12 },
  ],
  "home-pf": [
    { t: 0.0, x: 76, y: 30 },
    { t: 2.2, x: 82, y: 28 },
    { t: 3.6, x: 84, y: 26 },
    { t: 5.0, x: 86, y: 24 },
  ],
  "home-c": [
    { t: 0.0, x: 80, y: 16 },
    { t: 3.6, x: 84, y: 19 },
    { t: 5.0, x: 87, y: 23 },
  ],
};

const OFFENSE_IDS = Object.keys(OFFENSE_PATHS);

// --- Ball script ---------------------------------------------------------------------
// Timeline (seconds):        holder / flight
//   0.0 - 2.0   PG dribbles up            (ball follows PG, bouncing)
//   2.0 - 2.4   pass PG -> SG             (low arc)
//   2.4 - 3.6   SG holds / gathers        (triple-threat height)
//   3.6 - 4.5   jump shot SG -> rim       (high arc, ends at rim height)
//   4.5 - 5.0   made basket               (drops through the net)

const PASS_START = 2.0;
const PASS_END = 2.4;
const SHOT_START = 3.6;
const SHOT_END = 4.5;
const HOLD_HEIGHT_FT = 4.5; // TWEAK: where a held ball sits (waist/chest height).
const SHOT_PEAK_FT = 16; // TWEAK: shot-arc apex height.

function ballAt(t: number): Vec3 {
  const pg = samplePath(OFFENSE_PATHS["home-pg"] ?? [], t);
  const sg = samplePath(OFFENSE_PATHS["home-sg"] ?? [], t);

  if (t < PASS_START) {
    // Dribble: bounce between hand (~4ft) and floor at ~2 bounces/second.
    const bounce = Math.abs(Math.sin(Math.PI * 2 * t));
    return { x: pg.x + 1, y: pg.y, z: 0.8 + 3.2 * bounce };
  }
  if (t < PASS_END) {
    const u = (t - PASS_START) / (PASS_END - PASS_START);
    const from = samplePath(OFFENSE_PATHS["home-pg"] ?? [], PASS_START);
    const to = samplePath(OFFENSE_PATHS["home-sg"] ?? [], PASS_END);
    return {
      x: lerp(from.x, to.x, u),
      y: lerp(from.y, to.y, u),
      z: arcHeight(4, HOLD_HEIGHT_FT, 6.5, u),
    };
  }
  if (t < SHOT_START) {
    return { x: sg.x, y: sg.y, z: HOLD_HEIGHT_FT };
  }
  if (t < SHOT_END) {
    const u = (t - SHOT_START) / (SHOT_END - SHOT_START);
    const from = samplePath(OFFENSE_PATHS["home-sg"] ?? [], SHOT_START);
    return {
      x: lerp(from.x, HOOP.x, u),
      y: lerp(from.y, HOOP.y, u),
      z: arcHeight(7, RIM_HEIGHT_FT, SHOT_PEAK_FT, u),
    };
  }
  // Made basket: ball falls from the rim, slowing horizontally.
  const u = Math.min((t - SHOT_END) / 0.5, 1);
  return { x: HOOP.x, y: HOOP.y, z: RIM_HEIGHT_FT - 7 * u * u };
}

// --- Assembly ------------------------------------------------------------------------

/**
 * Defender position: on the line between their mark and the hoop, one quarter of the
 * way back. Crude but reads as plausible man defense without authoring 5 more paths.
 */
function defenderFor(mark: Vec2): Vec2 {
  // TWEAK: 0.35 = how far defenders sag off toward the hoop (higher = more open looks).
  return { x: lerp(mark.x, HOOP.x, 0.35), y: lerp(mark.y, HOOP.y, 0.35) };
}

/** Build the full stub possession: 126 frames (t = 0..5s inclusive) at 25fps. */
export function buildStubPlay(): Play {
  const frameCount = DURATION_S * FPS + 1;
  const frames: Frame[] = [];

  for (let i = 0; i < frameCount; i++) {
    const t = i / FPS;
    const players: PlayerSnapshot[] = [];

    // DO NOT TOUCH: offense then defense, in stable OFFENSE_IDS order, every frame —
    // the interpolator's index-alignment invariant (types.ts) depends on this ordering.
    for (const id of OFFENSE_IDS) {
      players.push({ playerId: id, team: "home", pos: samplePath(OFFENSE_PATHS[id] ?? [], t) });
    }
    for (const id of OFFENSE_IDS) {
      const mark = samplePath(OFFENSE_PATHS[id] ?? [], t);
      players.push({
        playerId: id.replace("home-", "away-d"),
        team: "away",
        pos: defenderFor(mark),
      });
    }

    frames.push({ t, players, ball: ballAt(t) });
  }

  return {
    fps: FPS,
    frames,
    courtLengthFt: COURT_LENGTH_FT,
    courtWidthFt: COURT_WIDTH_FT,
  };
}

/** The id the app follows by default — the shooter, who moves AND turns during the play. */
export const DEFAULT_POV_PLAYER_ID = "home-sg";
