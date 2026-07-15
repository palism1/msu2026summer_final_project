// FILE MAP
// ~L1-10   header + imports
// ~L11-45  advancePlayback: speed, pause, loop wrap, end clamp+stop
// ~L46-70  togglePlay / setSpeed / seek helpers
// Purpose: pins the pure playback clock's semantics.

import { describe, expect, it } from "vitest";
import {
  advancePlayback,
  createPlayback,
  seek,
  setSpeed,
  togglePlay,
} from "../src/engine/playback";

const DURATION = 5;

describe("advancePlayback", () => {
  it("advances time by dt * speed", () => {
    const state = setSpeed(createPlayback(), 2);
    expect(advancePlayback(state, 0.5, DURATION).time).toBeCloseTo(1, 10);
  });

  it("does not advance while paused (and returns the same reference)", () => {
    const paused = togglePlay(createPlayback(), DURATION);
    expect(paused.playing).toBe(false);
    expect(advancePlayback(paused, 1, DURATION)).toBe(paused);
  });

  it("wraps around the duration when looping", () => {
    const state = { ...createPlayback({ loop: true }), time: 4.8 };
    expect(advancePlayback(state, 0.5, DURATION).time).toBeCloseTo(0.3, 10);
  });

  it("clamps at the end and stops when not looping", () => {
    const state = { ...createPlayback({ loop: false }), time: 4.9 };
    const done = advancePlayback(state, 1, DURATION);
    expect(done.time).toBe(DURATION);
    expect(done.playing).toBe(false);
  });

  it("degenerates safely on a zero-duration play", () => {
    expect(advancePlayback(createPlayback(), 1, 0).time).toBe(0);
  });
});

describe("helpers", () => {
  it("togglePlay restarts a finished non-looping play from 0", () => {
    const finished = { ...createPlayback({ loop: false }), time: DURATION, playing: false };
    const resumed = togglePlay(finished, DURATION);
    expect(resumed.playing).toBe(true);
    expect(resumed.time).toBe(0);
  });

  it("setSpeed clamps to the allowed range", () => {
    expect(setSpeed(createPlayback(), 100).speed).toBeLessThanOrEqual(8);
    expect(setSpeed(createPlayback(), 0).speed).toBeGreaterThan(0);
  });

  it("seek clamps into [0, duration]", () => {
    expect(seek(createPlayback(), -2, DURATION).time).toBe(0);
    expect(seek(createPlayback(), 99, DURATION).time).toBe(DURATION);
    expect(seek(createPlayback(), 2.5, DURATION).time).toBe(2.5);
  });
});
