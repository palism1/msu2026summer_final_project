// FILE MAP
// ~L1-20   header + coordinate-system contract
// ~L21-40  primitive vectors (Vec2, Vec3)
// ~L41-75  raw tracking data types: PlayerSnapshot, Frame, Play (SportVU-shaped, 25fps)
// ~L76-95  derived per-render-tick types: SceneState (interpolated), CameraPose
// ~L96-110 PlaybackState (pure playback clock state)
// Purpose: THE frozen data contract of the whole app. Every module — engine, data,
//          cameras, render, ui — imports these shapes from here and nowhere else.
//
// DO NOT TOUCH: this file is the anti-backtracking anchor. Changing a shape here is
// the one expensive edit in the codebase (every tier depends on it). Additive changes
// (new optional fields, new types) are fine; renames/removals need a DECISIONS.md entry.
//
// Coordinate system (SportVU convention, units = feet):
//   x: 0..94 along the court length (baseline to baseline)
//   y: 0..50 along the court width  (sideline to sideline)
//   z: height above the floor (ball only in raw data; cameras add their own)
// Yaw: radians, 0 = facing +x, increasing counter-clockwise toward +y (atan2 convention).
// The render tier owns the mapping from this court space into Three.js world space;
// nothing outside src/render may assume a Three.js axis convention.

/** 2D point/vector on the court plane, in feet. */
export interface Vec2 {
  readonly x: number;
  readonly y: number;
}

/** 3D point/vector: court plane (x, y) plus height z, in feet. */
export interface Vec3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

/** Which bench a player belongs to. Rendering may color by this; logic must not care. */
export type TeamId = "home" | "away";

/** One player's sample within a single Frame. */
export interface PlayerSnapshot {
  /** Stable id, unique within a Play, identical across all frames of the Play. */
  readonly playerId: string;
  readonly team: TeamId;
  /** Court position in feet (players are tracked in 2D; SportVU has no player z). */
  readonly pos: Vec2;
}

/**
 * One raw tracking sample. SportVU 2015-16 shape: 10 players (x, y) + ball (x, y, z)
 * at 25 samples per second.
 *
 * DO NOT TOUCH invariant: `players` has the same length AND the same playerId order in
 * every Frame of a Play. The interpolator relies on index alignment between frames.
 */
export interface Frame {
  /** Seconds since the start of the play (frame i has t = i / fps). */
  readonly t: number;
  readonly players: readonly PlayerSnapshot[];
  readonly ball: Vec3;
}

/** A full possession worth of frames plus the constants needed to interpret them. */
export interface Play {
  /** Samples per second of `frames`. SportVU = 25. */
  readonly fps: number;
  readonly frames: readonly Frame[];
  /** Court dimensions in feet (NBA: 94 x 50). Render sizes the floor from these. */
  readonly courtLengthFt: number;
  readonly courtWidthFt: number;
}

/**
 * The world at one *continuous* time t — the engine's interpolated output for a render
 * tick. Same shapes as Frame on purpose: SceneState is "Frame, but between samples".
 * Render consumes this and only this; it never touches raw frames.
 */
export interface SceneState {
  readonly t: number;
  readonly players: readonly PlayerSnapshot[];
  readonly ball: Vec3;
}

/**
 * Where the camera is and where it looks, still in court space.
 * position.z is the eye height in feet. Render maps this to a Three.js camera.
 */
export interface CameraPose {
  readonly position: Vec3;
  /** Radians, atan2 convention (see coordinate contract above). */
  readonly yaw: number;
  /** Radians, positive looks up. Kept separate from yaw so smoothing stays 1D per axis. */
  readonly pitch: number;
}

/** Pure playback clock state; advanced by src/engine/playback.ts, owned by ui. */
export interface PlaybackState {
  /** Current playhead in seconds, always within [0, play duration]. */
  readonly time: number;
  readonly playing: boolean;
  /** Time multiplier; 1 = real time. */
  readonly speed: number;
  /** Wrap around at the end instead of stopping. */
  readonly loop: boolean;
}
