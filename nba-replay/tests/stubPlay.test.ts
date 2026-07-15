// FILE MAP
// ~L1-10   header + imports
// ~L11-55  buildStubPlay: SportVU shape, frame-alignment invariant, court bounds
// Purpose: pins the data-contract invariants the interpolator relies on.

import { describe, expect, it } from "vitest";
import { buildStubPlay, DEFAULT_POV_PLAYER_ID } from "../src/data/stubPlay";

describe("buildStubPlay", () => {
  const play = buildStubPlay();

  it("is SportVU-shaped: 25fps, ~5 seconds, 10 players, ball with height", () => {
    expect(play.fps).toBe(25);
    expect(play.frames.length).toBe(126); // t = 0..5s inclusive
    expect(play.frames[0]?.players.length).toBe(10);
    expect(typeof play.frames[0]?.ball.z).toBe("number");
  });

  it("keeps the same playerIds in the same order in every frame (interpolator invariant)", () => {
    const first = play.frames[0];
    if (first === undefined) throw new Error("no frames");
    const ids = first.players.map((p) => p.playerId);
    for (const frame of play.frames) {
      expect(frame.players.map((p) => p.playerId)).toEqual(ids);
    }
  });

  it("includes the default POV player and 5 per team", () => {
    const first = play.frames[0];
    if (first === undefined) throw new Error("no frames");
    expect(first.players.some((p) => p.playerId === DEFAULT_POV_PLAYER_ID)).toBe(true);
    expect(first.players.filter((p) => p.team === "home").length).toBe(5);
    expect(first.players.filter((p) => p.team === "away").length).toBe(5);
  });

  it("keeps everyone on the court and the ball above the floor", () => {
    for (const frame of play.frames) {
      for (const p of frame.players) {
        expect(p.pos.x).toBeGreaterThanOrEqual(0);
        expect(p.pos.x).toBeLessThanOrEqual(play.courtLengthFt);
        expect(p.pos.y).toBeGreaterThanOrEqual(0);
        expect(p.pos.y).toBeLessThanOrEqual(play.courtWidthFt);
      }
      expect(frame.ball.z).toBeGreaterThanOrEqual(0);
    }
  });

  it("stamps frame timestamps at exactly i / fps", () => {
    play.frames.forEach((frame, i) => {
      expect(frame.t).toBeCloseTo(i / play.fps, 10);
    });
  });
});
