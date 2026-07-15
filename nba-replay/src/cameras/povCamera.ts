// FILE MAP
// ~L1-25    header + imports + PovParams (all tunables in one place)
// ~L26-45   wrapAngle: shortest-arc angle normalization
// ~L46-60   eyePosition: 2D court position -> 3D eye point
// ~L61-85   playerVelocity: central-difference velocity from the interpolated track
// ~L86-125  facingYaw: blend of velocity direction and ball direction -> target yaw
// ~L126-160 smoothYaw: exponential smoothing + rotation-speed clamp (the anti-jitter core)
// ~L161-180 applyDeadZone: position dead zone (ignore sub-threshold position noise)
// ~L181-225 updatePovCamera: composes all of the above into prev pose -> next pose
// Purpose: all first-person camera math, as pure functions over the frozen types.
// Load-bearing tier: strict TS, no Three.js, tested in tests/povCamera.test.ts.

import { sampleSceneState } from "../engine/interpolate";
import type { CameraPose, Play, Vec2, Vec3 } from "../engine/types";

/** Every tunable of the POV camera. Pass a modified copy to experiment. */
export interface PovParams {
  /** Camera height above the floor, feet. ~5.9 approximates a 6'4" player's eyes. */
  readonly eyeHeightFt: number;
  /** 0 = face pure velocity (where I'm going), 1 = face pure ball (where the play is). */
  readonly ballWeight: number;
  /** Exponential smoothing rate for yaw, 1/s. Higher = snappier head turns. */
  readonly yawSmoothing: number;
  /** Hard cap on head-turn speed, rad/s. Prevents whip-pans when the target yaw jumps. */
  readonly maxTurnRate: number;
  /** Player movement below this radius (feet) does not move the camera at all. */
  readonly deadZoneFt: number;
  /** Fixed downward/upward tilt, radians. Slight down-tilt reads naturally on court. */
  readonly pitch: number;
}

// TWEAK: safe to adjust any of these; they only change camera feel, never correctness.
export const DEFAULT_POV_PARAMS: PovParams = {
  eyeHeightFt: 5.9,
  ballWeight: 0.65,
  yawSmoothing: 6,
  maxTurnRate: Math.PI * 1.25,
  deadZoneFt: 0.35,
  pitch: -0.06,
};

/**
 * Normalize an angle to (-PI, PI].
 * DO NOT TOUCH: every yaw delta below must pass through this, otherwise interpolating
 * from +179deg to -179deg spins 358deg the wrong way instead of 2deg the short way.
 */
export function wrapAngle(a: number): number {
  let r = a % (2 * Math.PI);
  if (r <= -Math.PI) r += 2 * Math.PI;
  if (r > Math.PI) r -= 2 * Math.PI;
  return r;
}

/** Lift a 2D court position to the 3D eye point. */
export function eyePosition(playerPos: Vec2, eyeHeightFt: number): Vec3 {
  return { x: playerPos.x, y: playerPos.y, z: eyeHeightFt };
}

/**
 * Player velocity (ft/s) at time t via central difference over the *interpolated*
 * track — sampling the spline instead of raw frames keeps velocity as smooth as
 * position, so the facing target doesn't stair-step 25 times a second.
 */
export function playerVelocity(play: Play, playerId: string, t: number, h = 0.04): Vec2 {
  const before = sampleSceneState(play, t - h);
  const after = sampleSceneState(play, t + h);
  const pBefore = before.players.find((p) => p.playerId === playerId);
  const pAfter = after.players.find((p) => p.playerId === playerId);
  if (pBefore === undefined || pAfter === undefined) {
    throw new Error(`playerVelocity: unknown playerId "${playerId}"`);
  }
  // Actual elapsed time between the two samples (clamping at play edges shrinks it).
  const dt = after.t - before.t;
  if (dt <= 0) return { x: 0, y: 0 };
  return { x: (pAfter.pos.x - pBefore.pos.x) / dt, y: (pAfter.pos.y - pBefore.pos.y) / dt };
}

/** Below this speed/distance (ft/s, ft) a direction is too noisy to trust. */
const DIRECTION_EPSILON = 0.25;

/**
 * Where the player should look: a weighted blend of "where I'm moving" and "where the
 * ball is". The blend happens on normalized *vectors*, then converts to an angle —
 * blending raw angles would break when the two directions straddle the +/-PI seam.
 * Falls back gracefully: no velocity -> face the ball; ball on top of you -> face
 * velocity; neither -> keep `fallbackYaw` (the previous facing).
 */
export function facingYaw(
  velocity: Vec2,
  toBall: Vec2,
  ballWeight: number,
  fallbackYaw: number,
): number {
  const speed = Math.hypot(velocity.x, velocity.y);
  const ballDist = Math.hypot(toBall.x, toBall.y);
  const hasVel = speed > DIRECTION_EPSILON;
  const hasBall = ballDist > DIRECTION_EPSILON;

  if (!hasVel && !hasBall) return fallbackYaw;
  if (!hasVel && hasBall) return Math.atan2(toBall.y, toBall.x);
  if (hasVel && !hasBall) return Math.atan2(velocity.y, velocity.x);

  const vx = velocity.x / speed;
  const vy = velocity.y / speed;
  const bx = toBall.x / ballDist;
  const by = toBall.y / ballDist;
  const mixX = vx * (1 - ballWeight) + bx * ballWeight;
  const mixY = vy * (1 - ballWeight) + by * ballWeight;
  // Opposite directions can cancel to ~zero; the ball wins that tie (players ball-watch).
  if (Math.hypot(mixX, mixY) < 1e-6) return Math.atan2(by, bx);
  return Math.atan2(mixY, mixX);
}

/**
 * Move `current` yaw toward `target` yaw over one tick of `dt` seconds.
 * Two stages, both essential to "smooth with no jitter":
 *   1. exponential smoothing: step = delta * (1 - e^(-smoothing*dt)) — frame-rate
 *      independent easing, so 30fps and 144fps clients turn identically per second;
 *   2. rotation-speed clamp: |step| <= maxTurnRate * dt — even a legitimate 180deg
 *      target flip (e.g. ball passed behind you) becomes a bounded pan, not a snap.
 * This is the 1D (yaw-only) equivalent of a clamped quaternion slerp; see DECISIONS.md
 * for why full quaternions were deliberately not used.
 */
export function smoothYaw(
  current: number,
  target: number,
  dt: number,
  smoothing: number,
  maxTurnRate: number,
): number {
  if (dt <= 0) return wrapAngle(current);
  const delta = wrapAngle(target - current);
  let step = delta * (1 - Math.exp(-smoothing * dt));
  const maxStep = maxTurnRate * dt;
  if (step > maxStep) step = maxStep;
  if (step < -maxStep) step = -maxStep;
  return wrapAngle(current + step);
}

/**
 * Position dead zone: the camera anchor only moves once the target escapes a radius
 * around it, and then only far enough to sit on that radius — so sub-threshold tracking
 * noise produces exactly zero camera translation instead of a constant shimmer.
 */
export function applyDeadZone(anchor: Vec2, target: Vec2, radiusFt: number): Vec2 {
  const dx = target.x - anchor.x;
  const dy = target.y - anchor.y;
  const dist = Math.hypot(dx, dy);
  if (dist <= radiusFt) return anchor;
  const pull = (dist - radiusFt) / dist;
  return { x: anchor.x + dx * pull, y: anchor.y + dy * pull };
}

/**
 * One camera tick: previous pose (null on the first tick) -> next pose.
 * Pipeline: sample world -> dead-zone the anchor -> eye height -> velocity+ball blend
 * -> smoothed, rate-clamped yaw. Pure: same inputs, same pose.
 */
export function updatePovCamera(
  prev: CameraPose | null,
  play: Play,
  playerId: string,
  t: number,
  dt: number,
  params: PovParams = DEFAULT_POV_PARAMS,
): CameraPose {
  const state = sampleSceneState(play, t);
  const player = state.players.find((p) => p.playerId === playerId);
  if (player === undefined) {
    throw new Error(`updatePovCamera: unknown playerId "${playerId}"`);
  }

  const prevAnchor: Vec2 = prev === null ? player.pos : { x: prev.position.x, y: prev.position.y };
  const anchor = applyDeadZone(prevAnchor, player.pos, params.deadZoneFt);

  const velocity = playerVelocity(play, playerId, t);
  const toBall: Vec2 = { x: state.ball.x - player.pos.x, y: state.ball.y - player.pos.y };
  const targetYaw = facingYaw(velocity, toBall, params.ballWeight, prev?.yaw ?? 0);
  const yaw =
    prev === null
      ? targetYaw
      : smoothYaw(prev.yaw, targetYaw, dt, params.yawSmoothing, params.maxTurnRate);

  return {
    position: eyePosition(anchor, params.eyeHeightFt),
    yaw,
    pitch: params.pitch,
  };
}
