import { describe, expect, it } from "vitest";

import { buildMemoryPromptSection } from "../../src/prompt.js";
import { MEMORY_TOOL_NAMES } from "../../src/tools.js";

describe("memory prompt section", () => {
  it("adds nothing when no memory tool is available", () => {
    expect(buildMemoryPromptSection({ availableTools: new Set(["web_search"]) })).toEqual([]);
  });

  it("guides only the tools the agent can call", () => {
    const text = buildMemoryPromptSection({ availableTools: new Set(["memory_search", "memory_get"]) }).join("\n");
    expect(text).toContain("use memory_search");
    expect(text).toContain("memory_get");
    expect(text).not.toContain("memory_store");
    expect(text).not.toContain("memory_forget");
  });

  it("steers memory away from files and treats recall as untrusted", () => {
    const lines = buildMemoryPromptSection({ availableTools: new Set(MEMORY_TOOL_NAMES) });
    expect(lines[0]).toBe("## Memory");
    expect(lines.join("\n")).toContain("not MEMORY.md");
    expect(lines.join("\n")).toContain("untrusted historical data");
    for (const name of MEMORY_TOOL_NAMES) {
      expect(lines.join("\n")).toContain(name);
    }
  });
});
