// FILE MAP
// ~L1-8    imports + config export
// ~L9-14   vitest config (node environment; tests live in tests/)
// Purpose: single Vite config shared by dev server, build, and Vitest.

import { defineConfig } from "vitest/config";

export default defineConfig({
  // TWEAK: base/dev-server options (port, open) can be added here freely.
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
