// FILE MAP
// ~L1-15   header + imports + tiny Play fixture builder
// ~L16-45  catmullRom scalar properties (endpoints, linearity preservation)
// ~L46-95  sampleSceneState: exact-frame hits, midpoints, clamping, id preservation
// Purpose: pins the interpolation contract the render loop depends on.

import { describe, expect, it } from "vitest";
import { catmullRom, playDuration, sampleSceneState } from "../src/engine/interpolate";
import type { Frame, Play } from "../src/engine/types";

/** Play where every tracked channel moves linearly — the easiest case to reason about. */
function linearPlay(frameCount: number, fps = 25): Play {
  const frames: Frame[] = [];
  for (let i = 0; i < frameCount; i++) {
    frames.push({
      t: i / fps,
      players: [
        { playerId: "p1", team: "home", pos: { x: i, y: 2 * i } },
        { playerId: "p2", team: "away", pos: { x: 50 - i, y: 25 } },
      ],
      ball: { x: 3 * i, y: 10, z: 5 },
    });
  }
  return { fps, frames, courtLengthFt: 94, courtWidthFt: 50 };
}

describe("catmullRom", () => {
  it("returns p1 at u=0 and p2 at u=1 (passes through samples)", () => {
    expect(catmullRom(1, 4, 9, 16, 0)).toBe(4);
    expect(catmullRom(1, 4, 9, 16, 1)).toBe(9);
  });

  it("reproduces linear data exactly (no overshoot on straight-line motion)", () => {
    // Collinear, evenly spaced control points: the spline must stay on the line.
    for (const u of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(catmullRom(0, 10, 20, 30, u)).toBeCloseTo(10 + 10 * u, 10);
    }
  });
});

describe("playDuration", () => {
  it("is (frameCount - 1) / fps", () => {
    expect(playDuration(linearPlay(126, 25))).toBeCloseTo(5, 10);
  });
});

describe("sampleSceneState", () => {
  const play = linearPlay(126);

  it("returns exact frame data at exact sample times", () => {
    const state = sampleSceneState(play, 10 / 25);
    expect(state.players[0]?.pos.x).toBeCloseTo(10, 10);
    expect(state.players[0]?.pos.y).toBeCloseTo(20, 10);
    expect(state.ball.x).toBeCloseTo(30, 10);
  });

  it("returns the halfway point between frames for linear motion", () => {
    const state = sampleSceneState(play, 10.5 / 25);
    expect(state.players[0]?.pos.x).toBeCloseTo(10.5, 10);
    expect(state.players[1]?.pos.x).toBeCloseTo(50 - 10.5, 10);
    expect(state.ball.x).toBeCloseTo(31.5, 10);
  });

  it("clamps t below 0 and beyond the play's end", () => {
    const before = sampleSceneState(play, -1);
    const after = sampleSceneState(play, 999);
    expect(before.t).toBe(0);
    expect(before.players[0]?.pos.x).toBeCloseTo(0, 10);
    expect(after.t).toBeCloseTo(playDuration(play), 10);
    expect(after.players[0]?.pos.x).toBeCloseTo(125, 10);
  });

  it("preserves playerId and team through interpolation", () => {
    const state = sampleSceneState(play, 1.234);
    expect(state.players.map((p) => p.playerId)).toEqual(["p1", "p2"]);
    expect(state.players.map((p) => p.team)).toEqual(["home", "away"]);
  });

  it("throws on an empty play instead of returning garbage", () => {
    const empty: Play = { fps: 25, frames: [], courtLengthFt: 94, courtWidthFt: 50 };
    expect(() => sampleSceneState(empty, 0)).toThrow();
  });
});
