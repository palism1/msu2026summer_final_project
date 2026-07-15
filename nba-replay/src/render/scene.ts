// FILE MAP
// ~L1-20    header + imports + SceneHandles type
// ~L21-40   courtToWorld: THE court-space -> Three.js-world mapping (single source)
// ~L41-110  createScene: renderer, lights, floor, hoops, player capsules, ball
// ~L111-140 applySceneState: push an interpolated SceneState into the meshes
// ~L141-170 applyCameraPose: push a CameraPose into the Three.js camera
// Purpose: all Three.js lives in src/render + src/ui and NOWHERE else (two-tier rule).
// This module renders whatever the engine hands it; it contains zero game/camera logic.

import * as THREE from "three";
import type { CameraPose, Play, SceneState } from "../engine/types";

/** Everything ui needs to drive a frame: mutate via applySceneState/applyCameraPose. */
export interface SceneHandles {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly playerMeshes: ReadonlyMap<string, THREE.Mesh>;
  readonly ballMesh: THREE.Mesh;
}

/**
 * Map court space (x: 0..94, y: 0..50, z: height, feet) to Three.js world space
 * (1 world unit = 1 foot): court x -> world X, court y -> world Z, height -> world Y.
 * DO NOT TOUCH: every position AND the camera math below assume this exact mapping;
 * changing it here without changing applyCameraPose flips the world.
 */
export function courtToWorld(x: number, y: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, z, y);
}

const PLAYER_HEIGHT_FT = 6.4; // TWEAK: capsule visual height.
const PLAYER_RADIUS_FT = 0.8; // TWEAK: capsule visual radius.
const BALL_RADIUS_FT = 0.39; // Regulation ball is ~9.4in diameter.

function makeHoop(x: number): THREE.Group {
  const group = new THREE.Group();
  const poleMat = new THREE.MeshStandardMaterial({ color: 0x555555 });
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 12, 12), poleMat);
  pole.position.copy(courtToWorld(x < 47 ? x - 4 : x + 4, 25, 6));
  const board = new THREE.Mesh(
    new THREE.BoxGeometry(0.2, 3.5, 6),
    new THREE.MeshStandardMaterial({ color: 0xdddddd }),
  );
  board.position.copy(courtToWorld(x < 47 ? x - 1.25 : x + 1.25, 25, 11));
  const rim = new THREE.Mesh(
    new THREE.TorusGeometry(0.75, 0.06, 8, 24),
    new THREE.MeshStandardMaterial({ color: 0xdd4422 }),
  );
  rim.position.copy(courtToWorld(x, 25, 10));
  rim.rotation.x = Math.PI / 2;
  group.add(pole, board, rim);
  return group;
}

/** Build the whole static scene + one mesh per player + the ball; attach to container. */
export function createScene(container: HTMLElement, play: Play): SceneHandles {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x10131a); // TWEAK: arena background color.

  const camera = new THREE.PerspectiveCamera(
    70, // TWEAK: POV field of view in degrees.
    container.clientWidth / container.clientHeight,
    0.1,
    500,
  );

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(40, 80, 60);
  scene.add(sun);

  // Court floor, sized from the Play so real data with other dimensions still fits.
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(play.courtLengthFt, play.courtWidthFt),
    new THREE.MeshStandardMaterial({ color: 0xb98a4d }), // TWEAK: hardwood color.
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.copy(courtToWorld(play.courtLengthFt / 2, play.courtWidthFt / 2, 0));
  scene.add(floor);

  const centerLine = new THREE.Mesh(
    new THREE.PlaneGeometry(0.4, play.courtWidthFt),
    new THREE.MeshBasicMaterial({ color: 0xf5f0e6 }),
  );
  centerLine.rotation.x = -Math.PI / 2;
  centerLine.position.copy(courtToWorld(play.courtLengthFt / 2, play.courtWidthFt / 2, 0.01));
  scene.add(centerLine);

  scene.add(makeHoop(5.25), makeHoop(play.courtLengthFt - 5.25));

  // One capsule per player, keyed by the frozen playerId from the data contract.
  const playerMeshes = new Map<string, THREE.Mesh>();
  const firstFrame = play.frames[0];
  const capsuleGeo = new THREE.CapsuleGeometry(
    PLAYER_RADIUS_FT,
    PLAYER_HEIGHT_FT - 2 * PLAYER_RADIUS_FT,
    4,
    12,
  );
  for (const p of firstFrame?.players ?? []) {
    const mat = new THREE.MeshStandardMaterial({
      color: p.team === "home" ? 0x2b6fd6 : 0xd63b3b, // TWEAK: team colors.
    });
    const mesh = new THREE.Mesh(capsuleGeo, mat);
    playerMeshes.set(p.playerId, mesh);
    scene.add(mesh);
  }

  const ballMesh = new THREE.Mesh(
    new THREE.SphereGeometry(BALL_RADIUS_FT, 20, 20),
    new THREE.MeshStandardMaterial({ color: 0xe8722a }),
  );
  scene.add(ballMesh);

  window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  return { scene, camera, renderer, playerMeshes, ballMesh };
}

/** Move every player capsule + the ball to an interpolated SceneState. */
export function applySceneState(handles: SceneHandles, state: SceneState): void {
  for (const p of state.players) {
    const mesh = handles.playerMeshes.get(p.playerId);
    if (mesh === undefined) continue; // Unknown id: skip rather than crash the frame.
    // Capsule origin is its center, so lift by half the height to stand on the floor.
    mesh.position.copy(courtToWorld(p.pos.x, p.pos.y, PLAYER_HEIGHT_FT / 2));
  }
  handles.ballMesh.position.copy(courtToWorld(state.ball.x, state.ball.y, state.ball.z));
}

/**
 * Point the Three.js camera per a court-space CameraPose.
 * Forward vector from yaw/pitch in court space, mapped through courtToWorld's axes:
 * court (cos yaw, sin yaw) lands on world (X, Z), pitch lifts world Y.
 */
export function applyCameraPose(camera: THREE.PerspectiveCamera, pose: CameraPose): void {
  const eye = courtToWorld(pose.position.x, pose.position.y, pose.position.z);
  camera.position.copy(eye);
  const cosP = Math.cos(pose.pitch);
  const forward = new THREE.Vector3(
    cosP * Math.cos(pose.yaw),
    Math.sin(pose.pitch),
    cosP * Math.sin(pose.yaw),
  );
  camera.lookAt(eye.clone().add(forward));
}
