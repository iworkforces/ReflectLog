import type { AnyAgentTool, OpenClawConfig, OpenClawPluginToolContext } from "openclaw/plugin-sdk/plugin-entry";
import { describe, expect, it, vi } from "vitest";

import { MemoryAccessDeniedError, ownerScope } from "../../src/access.js";
import { PLUGIN_ID, parseConfig, type ReflectLogPluginConfig } from "../../src/config.js";
import { UNTRUSTED_MEMORY_NOTICE } from "../../src/policy.js";
import { ReflectLogError } from "../../src/reflectlog/types.js";
import { MemoryInputError, MemoryService } from "../../src/service.js";
import {
  createMemoryForgetTool, createMemoryGetTool, createMemoryListTool, createMemoryPreviewTool,
  createMemorySearchTool, createMemoryStoreTool,
} from "../../src/tools.js";
import { FakeReflectLogClient } from "../support/fake-client.js";

function config(raw: unknown): ReflectLogPluginConfig {
  const parsed = parseConfig(raw);
  if (!parsed.ok) throw new Error(parsed.error);
  return parsed.config;
}

const smart = config({ smartReplace: true });
const owner = ownerScope(smart, "main");
const configured: OpenClawConfig = { plugins: { slots: { memory: PLUGIN_ID } } };
const ownerContext: OpenClawPluginToolContext = { agentId: "main", memoryAudience: { kind: "owner-private", agentId: "main" } };

type Create = typeof createMemorySearchTool;

function setup(options: { ctx?: OpenClawPluginToolContext; host?: OpenClawConfig; config?: ReflectLogPluginConfig } = {}) {
  const pluginConfig = options.config ?? smart;
  const client = new FakeReflectLogClient();
  const service = new MemoryService(client, pluginConfig);
  const deps = { api: { config: options.host ?? configured }, config: pluginConfig, getService: () => service };
  const build = (create: Create): AnyAgentTool | null => create(deps)(options.ctx ?? ownerContext);
  const tool = (create: Create): AnyAgentTool => {
    const built = build(create);
    if (!built) throw new Error("tool unavailable");
    return built;
  };
  return { client, build, tool };
}

async function run(tool: AnyAgentTool, params: unknown) {
  const result = await tool.execute("call-1", params);
  const text = result.content.map((block) => (block.type === "text" ? block.text : "")).join("");
  return { text, details: result.details };
}

describe("memory tools", () => {
  it("are offered only to an agent while this plugin owns the memory slot", () => {
    expect(setup({ ctx: { memoryAudience: { kind: "owner-private", agentId: "main" } } }).build(createMemorySearchTool)).toBeNull();
    expect(setup({ host: { plugins: { slots: { memory: "memory-core" } } } }).build(createMemorySearchTool)).toBeNull();
    const runtime = setup({
      host: { plugins: { slots: { memory: "memory-core" } } },
      ctx: { ...ownerContext, getRuntimeConfig: () => configured },
    });
    expect(runtime.build(createMemorySearchTool)?.name).toBe("memory_search");
  });

  it("frames search hits as untrusted data and accepts snake_case aliases", async () => {
    const { client, tool } = setup();
    client.seed(owner.workspaceId, ["Prefers green tea", "Drinks tea at noon"]);
    const { text, details } = await run(tool(createMemorySearchTool), { query: "green tea", max_results: 1, min_score: 0.5 });
    expect(text).toContain(UNTRUSTED_MEMORY_NOTICE);
    expect(text).toContain(`1. ref=${owner.tag}:1 score=1 Prefers green tea`);
    expect(text).toMatch(/^<relevant-memories>[\s\S]*<\/relevant-memories>$/);
    expect(details).toMatchObject({ count: 1, scoreKind: "reranker" });
    expect(client.callsTo("searchRecords")[0]?.params).toMatchObject({ query: "green tea", limit: 1 });
  });

  it("says so when nothing matches", async () => {
    expect((await run(setup().tool(createMemorySearchTool), { query: "tea" })).text).toContain("No matching memories.");
  });

  it("reads a memory by reference and reports missing ones", async () => {
    const { client, tool } = setup();
    client.seed(owner.workspaceId, ["one\ntwo"]);
    const get = tool(createMemoryGetTool);
    expect((await run(get, { ref: `${owner.tag}:1`, from: 2 })).text).toBe(`${UNTRUSTED_MEMORY_NOTICE}\ntwo`);
    expect((await run(get, { ref: `${owner.tag}:9` })).text).toContain("not found");
  });

  it.each([
    ["requested with smart replacement enabled", smart, true, true],
    ["not requested", smart, undefined, false],
    ["requested while smart replacement is disabled", config({}), true, false],
  ] as const)("stores with replacement only when %s", async (_label, pluginConfig, replace, allowReplace) => {
    const { client, tool } = setup({ config: pluginConfig });
    const { text } = await run(tool(createMemoryStoreTool), { memories: ["Prefers tea"], ...(replace === undefined ? {} : { replace }) });
    expect(text).toContain("Stored: 1; skipped: 0; replaced: 0; deduplicated: false.");
    expect(client.callsTo("add")[0]?.params).toMatchObject({ dryRun: false, allowReplace });
    expect(client.records(owner.workspaceId)).toHaveLength(1);
  });

  it("previews a write without storing it", async () => {
    const { client, tool } = setup();
    const { text } = await run(tool(createMemoryPreviewTool), { memories: ["Prefers tea"] });
    expect(text).toMatch(/^Preview only, nothing stored\./);
    expect(client.callsTo("add")[0]?.params).toMatchObject({ dryRun: true });
    expect(client.records(owner.workspaceId)).toEqual([]);
  });

  it("deduplicates a retried flush write and never replaces during a flush", async () => {
    const { client, tool } = setup({ ctx: { ...ownerContext, memoryFlush: { flushId: "flush-dedup" } } });
    const write = tool(createMemoryStoreTool);
    const params = { memories: ["Decided to ship on Friday"], replace: true };
    expect((await run(write, params)).details).toMatchObject({ storedCount: 1, deduplicated: false });
    expect((await run(write, params)).details).toMatchObject({ storedCount: 1, deduplicated: true });
    expect(client.callsTo("add")).toHaveLength(1);
    expect(client.callsTo("add")[0]?.params).toMatchObject({ allowReplace: false });
  });

  it("forgets by reference and reports unknown references", async () => {
    const { client, tool } = setup();
    client.seed(owner.workspaceId, ["Prefers tea"]);
    const { text } = await run(tool(createMemoryForgetTool), { refs: [`${owner.tag}:1`, `${owner.tag}:7`] });
    expect(text).toBe(`Removed: ${owner.tag}:1\nMissing: ${owner.tag}:7`);
    expect(client.records(owner.workspaceId)).toEqual([]);
  });

  it("pages through memories", async () => {
    const { client, tool } = setup();
    client.seed(owner.workspaceId, ["first", "second", "third"]);
    const list = tool(createMemoryListTool);
    const page = await run(list, { limit: 2 });
    expect(page.text).toContain(`ref=${owner.tag}:2 second`);
    expect(page.text).toContain("Total: 3; next offset: 2");
    expect(page.details).toMatchObject({ nextOffset: 2 });
    const last = await run(list, { limit: 2, offset: 2 });
    expect(last.text).toMatch(/Total: 3$/);
    expect(last.details).not.toHaveProperty("nextOffset");
  });

  it.each([
    [createMemoryStoreTool, { memories: [] }],
    [createMemorySearchTool, { query: "" }],
    [createMemoryForgetTool, { refs: "x" }],
  ] as const)("rejects invalid input before any backend call", async (create, params) => {
    const { client, tool } = setup();
    await expect(run(tool(create), params)).rejects.toThrow(MemoryInputError);
    expect(client.calls).toEqual([]);
  });

  it("denies a call without a host-minted audience", async () => {
    const { client, tool } = setup({ ctx: { agentId: "main" } });
    await expect(run(tool(createMemorySearchTool), { query: "tea" })).rejects.toThrow(MemoryAccessDeniedError);
    expect(client.calls).toEqual([]);
  });

  it.each(["assertInvocationCurrent", "assertMemoryAudienceCurrent"] as const)(
    "makes no backend call once %s fails", async (assertion) => {
      const { client, tool } = setup({ ctx: { ...ownerContext, [assertion]: () => { throw new Error("stale"); } } });
      await expect(run(tool(createMemoryStoreTool), { memories: ["Prefers tea"] })).rejects.toThrow("stale");
      expect(client.calls).toEqual([]);
    },
  );

  it("withholds results once the memory slot moves to another plugin mid-call", async () => {
    let slotOwner = PLUGIN_ID;
    const { client, tool } = setup({
      ctx: { ...ownerContext, getRuntimeConfig: () => ({ plugins: { slots: { memory: slotOwner } } }) },
    });
    client.seed(owner.workspaceId, ["Prefers tea"]);
    const search = client.searchRecords.bind(client);
    vi.spyOn(client, "searchRecords").mockImplementation(async (params, options) => {
      const result = await search(params, options);
      slotOwner = "memory-core";
      return result;
    });
    await expect(run(tool(createMemorySearchTool), { query: "tea" })).rejects.toThrow(MemoryAccessDeniedError);
  });

  it("tells the agent to check before retrying a write with an unknown outcome", async () => {
    const { client, tool } = setup();
    client.failNext("add", new ReflectLogError("outcome_unknown", "ReflectLog call aborted"));
    await expect(run(tool(createMemoryStoreTool), { memories: ["Prefers tea"] })).rejects.toMatchObject({
      code: "outcome_unknown",
      message: expect.stringContaining("Check with memory_search before retrying"),
    });
  });
});
