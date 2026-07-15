// FILE MAP
// ~L1-15   header + imports
// ~L16-35  bootstrap: build stub play, scene, playback clock, HUD overlay
// ~L36-55  input: space = play/pause
// ~L56-95  requestAnimationFrame loop: advance clock -> sample -> camera -> render
// Purpose: app entry point; the only module that owns wall-clock time and the DOM.
// Loose tier: typecheck-only glue — all logic it calls is tested in the strict tier.

import { updatePovCamera } from "../cameras/povCamera";
import { buildStubPlay, DEFAULT_POV_PLAYER_ID } from "../data/stubPlay";
import { playDuration, sampleSceneState } from "../engine/interpolate";
import { advancePlayback, createPlayback, togglePlay } from "../engine/playback";
import type { CameraPose } from "../engine/types";
import { applyCameraPose, applySceneState, createScene } from "../render/scene";

const container = document.getElementById("app");
if (container === null) throw new Error("main: #app container missing from index.html");

const play = buildStubPlay();
const duration = playDuration(play);
// TWEAK: change which player the camera rides (any playerId from stubPlay.ts).
const povPlayerId = DEFAULT_POV_PLAYER_ID;

const handles = createScene(container, play);
// Hide the followed player's own capsule so the camera isn't inside a blue tube.
const selfMesh = handles.playerMeshes.get(povPlayerId);
if (selfMesh !== undefined) selfMesh.visible = false;

const hud = document.createElement("div");
hud.id = "hud";
container.appendChild(hud);

let playback = createPlayback();
let pose: CameraPose | null = null;
let lastMs: number | null = null;

window.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    playback = togglePlay(playback, duration);
  }
});

function tick(nowMs: number): void {
  // Clamp dt so a background tab doesn't produce one giant catch-up jump.
  const dt = lastMs === null ? 0 : Math.min((nowMs - lastMs) / 1000, 0.05);
  lastMs = nowMs;

  playback = advancePlayback(playback, dt, duration);
  const state = sampleSceneState(play, playback.time);
  pose = updatePovCamera(pose, play, povPlayerId, playback.time, dt);

  applySceneState(handles, state);
  applyCameraPose(handles.camera, pose);
  handles.renderer.render(handles.scene, handles.camera);

  hud.textContent = `POV: ${povPlayerId}  ·  t = ${playback.time.toFixed(2)}s / ${duration.toFixed(2)}s  ·  ${playback.playing ? "▶" : "⏸"} (space)`;
  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
