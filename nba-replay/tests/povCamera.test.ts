// FILE MAP
// ~L1-20    header + imports + straight-line Play fixture
// ~L21-40   wrapAngle + eyePosition
// ~L41-80   facingYaw: pure-velocity, pure-ball, blend, fallbacks
// ~L81-120  smoothYaw: convergence, rate clamp, shortest arc across the +/-PI seam
// ~L121-140 applyDeadZone: inside = frozen, outside = trails on the boundary
// ~L141-185 updatePovCamera: composition (eye height, first tick, follow, purity)
// Purpose: pins every pure function of the POV camera math.

import { describe, expect, it } from "vitest";
import {
  applyDeadZone,
  DEFAULT_POV_PARAMS,
  eyePosition,
  facingYaw,
  playerVelocity,
  smoothYaw,
  updatePovCamera,
  wrapAngle,
} from "../src/cameras/povCamera";
import type { Frame, Play } from "../src/engine/types";

/** One player marching +x at exactly 10 ft/s; ball parked far up the +y sideline. */
function marchingPlay(): Play {
  const fps = 25;
  const frames: Frame[] = [];
  for (let i = 0; i <= 125; i++) {
    frames.push({
      t: i / fps,
      players: [{ playerId: "p1", team: "home", pos: { x: 20 + (10 * i) / fps, y: 25 } }],
      ball: { x: 40, y: 45, z: 5 },
    });
  }
  return { fps, frames, courtLengthFt: 94, courtWidthFt: 50 };
}

describe("wrapAngle", () => {
  it("maps into (-PI, PI]", () => {
    expect(wrapAngle(0)).toBe(0);
    expect(wrapAngle(Math.PI)).toBeCloseTo(Math.PI, 10);
    expect(wrapAngle(-Math.PI)).toBeCloseTo(Math.PI, 10);
    expect(wrapAngle(3 * Math.PI)).toBeCloseTo(Math.PI, 10);
    expect(wrapAngle(2 * Math.PI + 0.25)).toBeCloseTo(0.25, 10);
    expect(wrapAngle(-2 * Math.PI - 0.25)).toBeCloseTo(-0.25, 10);
  });
});

describe("eyePosition", () => {
  it("lifts the court position to eye height", () => {
    expect(eyePosition({ x: 10, y: 20 }, 5.9)).toEqual({ x: 10, y: 20, z: 5.9 });
  });
});

describe("playerVelocity", () => {
  it("recovers constant velocity from the interpolated track", () => {
    const v = playerVelocity(marchingPlay(), "p1", 2.5);
    expect(v.x).toBeCloseTo(10, 5);
    expect(v.y).toBeCloseTo(0, 5);
  });

  it("throws on an unknown playerId", () => {
    expect(() => playerVelocity(marchingPlay(), "nope", 1)).toThrow();
  });
});

describe("facingYaw", () => {
  const east = { x: 1, y: 0 };
  const north = { x: 0, y: 1 };

  it("faces pure velocity at ballWeight 0 and pure ball at ballWeight 1", () => {
    expect(facingYaw(east, north, 0, 99)).toBeCloseTo(0, 10);
    expect(facingYaw(east, north, 1, 99)).toBeCloseTo(Math.PI / 2, 10);
  });

  it("blends perpendicular unit directions to the diagonal at weight 0.5", () => {
    expect(facingYaw(east, north, 0.5, 99)).toBeCloseTo(Math.PI / 4, 10);
  });

  it("falls back to the ball when standing still, and to velocity when on the ball", () => {
    expect(facingYaw({ x: 0, y: 0 }, north, 0.5, 99)).toBeCloseTo(Math.PI / 2, 10);
    expect(facingYaw(east, { x: 0, y: 0 }, 0.5, 99)).toBeCloseTo(0, 10);
  });

  it("keeps the previous yaw when there is no signal at all", () => {
    expect(facingYaw({ x: 0, y: 0 }, { x: 0, y: 0 }, 0.5, 1.23)).toBe(1.23);
  });

  it("lets the ball win when velocity and ball direction exactly cancel", () => {
    const west = { x: -1, y: 0 };
    expect(facingYaw(east, west, 0.5, 99)).toBeCloseTo(Math.PI, 10);
  });
});

describe("smoothYaw", () => {
  it("moves toward the target without overshooting", () => {
    const next = smoothYaw(0, 1, 0.016, 6, Math.PI * 4);
    expect(next).toBeGreaterThan(0);
    expect(next).toBeLessThan(1);
  });

  it("clamps the per-tick step to maxTurnRate * dt", () => {
    const dt = 0.016;
    const maxRate = 1; // rad/s -> max step 0.016 rad this tick
    // Huge smoothing would want to jump nearly the whole PI delta; the clamp must win.
    const next = smoothYaw(0, Math.PI, dt, 1000, maxRate);
    expect(next).toBeCloseTo(maxRate * dt, 10);
  });

  it("takes the shortest arc across the +/-PI seam", () => {
    // 170deg -> -170deg is 20deg through the seam, not 340deg back through zero.
    const current = (170 * Math.PI) / 180;
    const target = (-170 * Math.PI) / 180;
    const next = smoothYaw(current, target, 0.016, 6, Math.PI * 4);
    // Must have rotated positively (through +PI), i.e. wrapped past the seam or grown.
    expect(wrapAngle(next - current)).toBeGreaterThan(0);
  });

  it("converges to the target over many ticks", () => {
    let yaw = 0;
    for (let i = 0; i < 300; i++) yaw = smoothYaw(yaw, 2, 1 / 60, 6, Math.PI * 4);
    expect(yaw).toBeCloseTo(2, 3);
  });
});

describe("applyDeadZone", () => {
  it("does not move the anchor while the target is inside the radius", () => {
    const anchor = { x: 10, y: 10 };
    expect(applyDeadZone(anchor, { x: 10.2, y: 10.1 }, 0.35)).toBe(anchor);
  });

  it("trails the target on the dead-zone boundary once outside", () => {
    const moved = applyDeadZone({ x: 0, y: 0 }, { x: 2, y: 0 }, 0.5);
    expect(moved.x).toBeCloseTo(1.5, 10);
    expect(moved.y).toBeCloseTo(0, 10);
    // Distance from moved anchor to target equals exactly the radius.
    expect(Math.hypot(2 - moved.x, 0 - moved.y)).toBeCloseTo(0.5, 10);
  });
});

describe("updatePovCamera", () => {
  const play = marchingPlay();

  it("positions the eye at eye height above the followed player", () => {
    const pose = updatePovCamera(null, play, "p1", 2.5, 0.016);
    expect(pose.position.z).toBe(DEFAULT_POV_PARAMS.eyeHeightFt);
    expect(pose.position.x).toBeCloseTo(45, 5);
    expect(pose.position.y).toBeCloseTo(25, 5);
  });

  it("snaps directly to the target yaw on the first tick (no prev pose)", () => {
    const pose = updatePovCamera(null, play, "p1", 2.5, 0.016);
    // Moving east with the ball up-court to the north-ish: yaw must be in (0, PI/2).
    expect(pose.yaw).toBeGreaterThan(0);
    expect(pose.yaw).toBeLessThan(Math.PI / 2);
  });

  it("follows the player over successive ticks", () => {
    let pose = updatePovCamera(null, play, "p1", 0, 0);
    for (let step = 1; step <= 50; step++) {
      pose = updatePovCamera(pose, play, "p1", step * 0.05, 0.05);
    }
    const expected = 20 + 10 * 2.5; // 2.5s at 10 ft/s from x=20.
    expect(pose.position.x).toBeCloseTo(expected, 0);
  });

  it("is pure: identical inputs give identical poses", () => {
    const a = updatePovCamera(null, play, "p1", 1.7, 0.016);
    const b = updatePovCamera(null, play, "p1", 1.7, 0.016);
    expect(a).toEqual(b);
  });

  it("throws on an unknown playerId", () => {
    expect(() => updatePovCamera(null, play, "ghost", 1, 0.016)).toThrow();
  });
});
