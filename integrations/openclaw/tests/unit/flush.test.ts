import { describe, expect, it } from "vitest";

import { parseConfig } from "../../src/config.js";
import { resolveFlushPlan } from "../../src/flush.js";

describe("pre-compaction flush plan", () => {
  it("is absent without a valid configuration", () => {
    expect(resolveFlushPlan({ config: null })).toBeNull();
  });

  it("persists through memory_store only, with disjoint lookup tools", () => {
    const parsed = parseConfig({});
    if (!parsed.ok) throw new Error(parsed.error);
    const plan = resolveFlushPlan({ config: parsed.config, nowMs: 0 });
    expect(plan).toMatchObject({
      persistenceToolNames: ["memory_store"],
      lookupToolNames: ["memory_search", "memory_get"],
      prompt: expect.stringContaining("NO_REPLY"),
      systemPrompt: expect.stringContaining("never memory files"),
    });
  });
});
