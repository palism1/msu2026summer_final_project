// FILE MAP
// ~L1-12   header + imports
// ~L13-30  createPlayback: initial state factory
// ~L31-70  advancePlayback: the pure clock reducer (play/pause/speed/loop/clamp)
// ~L71-90  small pure state helpers (togglePlay, setSpeed, seek)
// Purpose: pure playback clock. ui owns a PlaybackState and feeds real dt in; nothing
// here reads wall time or touches the DOM, so every branch is unit-testable.
// Load-bearing tier: strict TS, no Three.js, tested in tests/playback.test.ts.

import type { PlaybackState } from "./types";

/** Fresh clock at t=0. Defaults: playing, real-time speed, looping. */
export function createPlayback(
  options: Partial<Pick<PlaybackState, "playing" | "speed" | "loop">> = {},
): PlaybackState {
  return {
    time: 0,
    // TWEAK: default autoplay/loop behavior of the app lives here.
    playing: options.playing ?? true,
    speed: options.speed ?? 1,
    loop: options.loop ?? true,
  };
}

/**
 * Advance the clock by dtSeconds of wall time against a play of `duration` seconds.
 * - paused        -> unchanged state (same reference, so callers can cheap-compare)
 * - loop          -> time wraps modulo duration
 * - no loop + end -> time clamps to duration and playback stops
 * Negative/zero duration degenerates safely to time=0.
 */
export function advancePlayback(
  state: PlaybackState,
  dtSeconds: number,
  duration: number,
): PlaybackState {
  if (!state.playing || dtSeconds <= 0) {
    return state;
  }
  if (duration <= 0) {
    return { ...state, time: 0 };
  }

  const rawTime = state.time + dtSeconds * state.speed;

  if (state.loop) {
    // DO NOT TOUCH: double-modulo keeps time in [0, duration) even if speed is negative.
    const wrapped = ((rawTime % duration) + duration) % duration;
    return { ...state, time: wrapped };
  }

  if (rawTime >= duration) {
    return { ...state, time: duration, playing: false };
  }
  return { ...state, time: Math.max(rawTime, 0) };
}

/** Flip play/pause. Restarts from 0 when resuming at the very end of a non-looping play. */
export function togglePlay(state: PlaybackState, duration: number): PlaybackState {
  if (!state.playing && !state.loop && state.time >= duration) {
    return { ...state, playing: true, time: 0 };
  }
  return { ...state, playing: !state.playing };
}

/** Set the speed multiplier (clamped to a sane positive range). */
export function setSpeed(state: PlaybackState, speed: number): PlaybackState {
  // TWEAK: allowed speed range.
  return { ...state, speed: Math.min(Math.max(speed, 0.1), 8) };
}

/** Jump the playhead to an absolute time, clamped to the play. */
export function seek(state: PlaybackState, time: number, duration: number): PlaybackState {
  return { ...state, time: Math.min(Math.max(time, 0), Math.max(duration, 0)) };
}
