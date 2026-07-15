// FILE MAP
// ~L1-20   header + imports + src file walker
// ~L21-40  Rule 1 guard: Frame/SceneState (and friends) declared ONLY in types.ts
// ~L41-60  Rule 2 guard: Three.js imported ONLY under src/render and src/ui
// Purpose: mechanical enforcement of CLAUDE.md Rules 1 and 2. Conventions rot; a
// failing CI test doesn't. If this test blocks you, the change is wrong — fix the
// change, never this test (see CLAUDE.md).

import { readdirSync, readFileSync } from "node:fs";
import { join, sep } from "node:path";
import { describe, expect, it } from "vitest";

/** All .ts files under src/, as repo-relative paths. */
function srcFiles(): string[] {
  return readdirSync("src", { recursive: true, encoding: "utf8" })
    .filter((f) => f.endsWith(".ts"))
    .map((f) => join("src", f));
}

const CONTRACT_HOME = join("src", "engine", "types.ts");
// The full frozen contract of types.ts — redefining ANY of these elsewhere is a leak.
const CONTRACT_TYPES = [
  "Frame",
  "SceneState",
  "Play",
  "PlayerSnapshot",
  "CameraPose",
  "PlaybackState",
  "Vec2",
  "Vec3",
];
const CONTRACT_DECL = new RegExp(
  `\\b(?:interface|type|class|enum)\\s+(?:${CONTRACT_TYPES.join("|")})\\b`,
);

const THREE_IMPORT = /from\s+["']three["']|import\s+["']three["']|require\(["']three["']\)/;
const THREE_ALLOWED = [join("src", "render") + sep, join("src", "ui") + sep];

describe("architecture rules (CLAUDE.md)", () => {
  it("Rule 1: contract types are declared only in src/engine/types.ts", () => {
    for (const file of srcFiles()) {
      if (file === CONTRACT_HOME) continue;
      const source = readFileSync(file, "utf8");
      expect(
        source,
        `${file} declares a contract type — import it from src/engine/types.ts instead`,
      ).not.toMatch(CONTRACT_DECL);
    }
  });

  it("Rule 2: Three.js is imported only under src/render and src/ui", () => {
    for (const file of srcFiles()) {
      if (THREE_ALLOWED.some((dir) => file.startsWith(dir))) continue;
      const source = readFileSync(file, "utf8");
      expect(
        source,
        `${file} imports three — Three.js is allowed only in src/render and src/ui`,
      ).not.toMatch(THREE_IMPORT);
    }
  });
});
