// FILE MAP
// ~L1-10   header + imports
// ~L11-40  the two-tier import rule, enforced mechanically
// Purpose: CI-enforces the architecture rule that src/engine, src/cameras, and src/data
// never import Three.js. If this test fails, logic is leaking into the render tier's
// dependency — move the Three.js usage into src/render or src/ui instead.

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const LOAD_BEARING_DIRS = ["src/engine", "src/cameras", "src/data"];

function tsFilesUnder(dir: string): string[] {
  return readdirSync(dir, { recursive: true, encoding: "utf8" })
    .filter((f) => f.endsWith(".ts"))
    .map((f) => join(dir, f));
}

describe("two-tier boundary", () => {
  it("keeps Three.js out of the load-bearing tier", () => {
    for (const dir of LOAD_BEARING_DIRS) {
      for (const file of tsFilesUnder(dir)) {
        const source = readFileSync(file, "utf8");
        expect(source, `${file} must not import three`).not.toMatch(
          /from\s+["']three["']|import\s+["']three["']|require\(["']three["']\)/,
        );
      }
    }
  });
});
