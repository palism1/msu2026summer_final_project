// FILE MAP
// ~L1-15   header + imports
// ~L16-45  scalar Catmull-Rom spline + Vec2/Vec3 wrappers
// ~L46-60  playDuration + safe frame lookup
// ~L61-110 sampleSceneState: Play + continuous t -> interpolated SceneState
// Purpose: pure interpolation math. Turns 25fps samples into smooth continuous motion.
// Load-bearing tier: strict TS, no Three.js, fully unit-tested (tests/interpolate.test.ts).

import type { Frame, Play, SceneState, Vec2, Vec3 } from "./types";

/**
 * Uniform Catmull-Rom spline for one scalar channel.
 * Interpolates BETWEEN p1 (u=0) and p2 (u=1); p0 and p3 only shape the tangents,
 * which is what makes the motion glide through every sample instead of kinking at it.
 *
 * DO NOT TOUCH: coefficient set is the standard uniform Catmull-Rom basis; it guarantees
 * C1 continuity across segments. If motion looks wrong, tune the *inputs*, not these.
 */
export function catmullRom(p0: number, p1: number, p2: number, p3: number, u: number): number {
  const u2 = u * u;
  const u3 = u2 * u;
  return (
    0.5 *
    (2 * p1 +
      (-p0 + p2) * u +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * u3)
  );
}

/** Catmull-Rom applied per-axis to court-plane points. */
export function catmullRomVec2(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, u: number): Vec2 {
  return {
    x: catmullRom(p0.x, p1.x, p2.x, p3.x, u),
    y: catmullRom(p0.y, p1.y, p2.y, p3.y, u),
  };
}

/** Catmull-Rom applied per-axis to 3D points (ball: court plane + height). */
export function catmullRomVec3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, u: number): Vec3 {
  return {
    x: catmullRom(p0.x, p1.x, p2.x, p3.x, u),
    y: catmullRom(p0.y, p1.y, p2.y, p3.y, u),
    z: catmullRom(p0.z, p1.z, p2.z, p3.z, u),
  };
}

/** Total duration of a play in seconds (last frame's timestamp). */
export function playDuration(play: Play): number {
  return (play.frames.length - 1) / play.fps;
}

/** Frame at index i, clamped to the play's range (duplicates endpoints for spline ends). */
function frameAt(frames: readonly Frame[], i: number): Frame {
  const clamped = Math.min(Math.max(i, 0), frames.length - 1);
  const frame = frames[clamped];
  if (frame === undefined) {
    throw new Error("sampleSceneState: play has no frames");
  }
  return frame;
}

/**
 * The engine's main output: the interpolated world at continuous time t (seconds).
 * t is clamped to [0, playDuration]. Each player and the ball are splined independently
 * through the four frames surrounding t (endpoint frames are duplicated at the edges,
 * which degrades gracefully to a clamped spline).
 *
 * DO NOT TOUCH invariant: relies on Frame.players being index-aligned across frames
 * (same ids, same order) — see the contract note in types.ts.
 */
export function sampleSceneState(play: Play, t: number): SceneState {
  const duration = playDuration(play);
  const clampedT = Math.min(Math.max(t, 0), duration);
  const exact = clampedT * play.fps;
  const i = Math.min(Math.floor(exact), play.frames.length - 2);
  const u = exact - i;

  const f0 = frameAt(play.frames, i - 1);
  const f1 = frameAt(play.frames, i);
  const f2 = frameAt(play.frames, i + 1);
  const f3 = frameAt(play.frames, i + 2);

  const players = f1.players.map((p, idx) => {
    const p0 = f0.players[idx]?.pos ?? p.pos;
    const p2 = f2.players[idx]?.pos ?? p.pos;
    const p3 = f3.players[idx]?.pos ?? p.pos;
    return {
      playerId: p.playerId,
      team: p.team,
      pos: catmullRomVec2(p0, p.pos, p2, p3, u),
    };
  });

  return {
    t: clampedT,
    players,
    ball: catmullRomVec3(f0.ball, f1.ball, f2.ball, f3.ball, u),
  };
}
