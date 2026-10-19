import type { AnyAgentTool, OpenClawConfig, OpenClawPluginToolContext } from "openclaw/plugin-sdk/plugin-entry";
import { describe, expect, it, vi } from "vitest";

import { MemoryAccessDeniedError, ownerScope } from "../../src/access.js";
import { PLUGIN_ID, parseConfig, type ReflectLogPluginConfig } from "../../src/config.js";
import { UNTRUSTED_MEMORY_NOTICE } from "../../src/policy.js";
import { FlushCapacityError, FlushDisposedError } from "../../src/flush-ledger.js";
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

function setup(options: {
  ctx?: OpenClawPluginToolContext; host?: OpenClawConfig; config?: ReflectLogPluginConfig;
  flush?: { maxEntries?: number; ttlMs?: number; now?: () => number };
} = {}) {
  const pluginConfig = options.config ?? smart;
  const client = new FakeReflectLogClient();
  const service = new MemoryService(client, pluginConfig, { flush: options.flush ?? {} });
  const deps = { api: { config: options.host ?? configured }, config: pluginConfig, getService: () => service };
  const build = (create: Create, ctx?: OpenClawPluginToolContext): AnyAgentTool | null =>
    create(deps)(ctx ?? options.ctx ?? ownerContext);
  const tool = (create: Create, ctx?: OpenClawPluginToolContext): AnyAgentTool => {
    const built = build(create, ctx);
    if (!built) throw new Error("tool unavailable");
    return built;
  };
  return { client, service, build, tool };
}

async function run(tool: AnyAgentTool, params: unknown, signal?: AbortSignal) {
  const result = await tool.execute("call-1", params, signal);
  const text = result.content.map((block) => (block.type === "text" ? block.text : "")).join("");
  return { text, details: result.details };
}

describe("memory tools", () => {
  it("returns explanations only in opted-in structured hit details after filtering", async () => {
    const { client, tool } = setup();
    client.seed(owner.workspaceId, ["tea", "tea biscuits"]);
    const search = tool(createMemorySearchTool);
    expect(search.parameters).toHaveProperty("properties.explain.type", "boolean");
    const result = await run(search, { query: "tea biscuits", minScore: 0.9, explain: true });
    expect(result.details).toMatchObject({ count: 1, hits: [{ ref: `${owner.tag}:2`, score: 1,
      explanation: { rawFinalScore: 1, semantic: { rank: 1, score: 1 }, fulltext: null,
        preRerankScore: 1, preRerankScoreKind: "reranker", finalScoreKind: "reranker" } }],
      stageOutcomes: { semantic: "applied", fulltext: "disabled", reranker: "disabled" } });
    expect(client.callsTo("searchMemories")[0]?.params).toHaveProperty("explain", true);
  });

  it("keeps raw fusion evidence distinct from nullable comparable scores", async () => {
    const { client, tool } = setup();
    client.seed(owner.workspaceId, ["tea"]);
    client.scoreKind = "rrf";
    expect((await run(tool(createMemorySearchTool), { query: "tea", explain: true })).details)
      .toMatchObject({ hits: [{ score: null, explanation: { rawFinalScore: 1, finalScoreKind: "rrf" } }] });
  });

  it("retains empty-server stages but omits diagnostics for local blank searches", async () => {
    const { client, tool } = setup();
    const search = tool(createMemorySearchTool);
    expect((await run(search, { query: "tea", explain: true })).details)
      .toMatchObject({ hits: [], stageOutcomes: { semantic: "empty", fulltext: "disabled", reranker: "disabled" } });
    expect((await run(search, { query: " ", explain: true })).details).not.toHaveProperty("stageOutcomes");
    expect(client.callsTo("searchMemories")).toHaveLength(1);
  });

  it.each([undefined, false])("omits default diagnostics and successor flags when opt-in is %s", async (flag) => {
    const { client, tool } = setup();
    client.seed(owner.workspaceId, ["tea"]);
    const search = await run(tool(createMemorySearchTool), { query: "tea", explain: flag });
    const get = await run(tool(createMemoryGetTool), { ref: `${owner.tag}:99`, includeSuccessor: flag });
    expect(search.details).not.toHaveProperty("stageOutcomes");
    expect(search.details).toHaveProperty("hits", [{ ref: `${owner.tag}:1`, score: 1, text: "tea" }]);
    expect(get.details).toEqual({ status: "not_found" });
    expect(client.callsTo("searchMemories")[0]?.params).not.toHaveProperty("explain");
    expect(client.callsTo("getMemoryById")[0]?.params).not.toHaveProperty("includeSuccessor");
  });

  it("returns a scoped successor hint without fetching or deleting the hinted record", async () => {
    const { client, tool } = setup();
    client.seed(owner.workspaceId, ["old", "current"]);
    await client.deleteMemories({ workspaceId: owner.workspaceId, ids: [1] });
    client.successorHints.set(owner.workspaceId, new Map([[1, 2]]));
    const get = tool(createMemoryGetTool);
    expect(get.parameters).toHaveProperty("properties.includeSuccessor.type", "boolean");
    expect((await run(get, { ref: `${owner.tag}:1`, includeSuccessor: true })).details)
      .toEqual({ status: "not_found", successor: { ref: `${owner.tag}:2` } });
    expect(client.callsTo("getMemoryById").map(call => call.params))
      .toEqual([{ workspaceId: owner.workspaceId, id: 1, includeSuccessor: true }]);
    expect(client.callsTo("deleteMemories")).toHaveLength(1);
  });

  it("returns explicit null when a missing original has no current successor", async () => {
    const { tool } = setup();
    expect((await run(tool(createMemoryGetTool), { ref: `${owner.tag}:99`, includeSuccessor: true })).details)
      .toEqual({ status: "not_found", successor: null });
  });

  const optionalTools: readonly {
    create: typeof createMemorySearchTool | typeof createMemoryGetTool;
    method: "searchMemories" | "getMemoryById";
    params: Readonly<{ query: string; explain: true }> | Readonly<{ ref: string; includeSuccessor: true }>;
  }[] = [
    { create: createMemorySearchTool, method: "searchMemories", params: { query: "tea", explain: true } },
    { create: createMemoryGetTool, method: "getMemoryById", params: { ref: `${owner.tag}:99`, includeSuccessor: true } },
  ];
  it.each(optionalTools)("withholds optional $method metadata after authority revocation", async ({ create, method, params }) => {
    const { client, tool } = setup();
    const state = { current: true };
    const pendingCall = client.deferNext(method);
    const pending = run(tool(create, { ...ownerContext, assertMemoryAudienceCurrent() {
      if (!state.current) throw new MemoryAccessDeniedError("revoked");
    } }), params);
    await pendingCall.dispatched;
    state.current = false;
    pendingCall.resolve();
    await expect(pending).rejects.toThrow(MemoryAccessDeniedError);
  });

  it.each(["true", 1, null])("rejects nonboolean opt-ins %s before backend dispatch", async (flag) => {
    const { client, tool } = setup();
    await expect(run(tool(createMemorySearchTool), { query: "tea", explain: flag })).rejects.toThrow(MemoryInputError);
    await expect(run(tool(createMemoryGetTool), { ref: `${owner.tag}:1`, includeSuccessor: flag })).rejects.toThrow(MemoryInputError);
    expect(client.calls).toEqual([]);
  });

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
    expect(text).toMatch(/^<relevant-memories>\n/);
    const [fenced, guidance] = text.split("</relevant-memories>");
    expect(fenced).toContain("Prefers green tea");
    expect(guidance).toContain("Search attempt 1 of 5.");
    expect(details).toMatchObject({ count: 1, scoreKind: "reranker" });
    expect(client.callsTo("searchMemories")[0]?.params).toMatchObject({ query: "green tea", limit: 1 });
  });

  it("says so when nothing matches", async () => {
    expect((await run(setup().tool(createMemorySearchTool), { query: "tea" })).text).toContain("No matching memories.");
  });

  it("starts at attempt 1 when omitted and keeps attempt out of backend parameters", async () => {
    const { client, tool } = setup();
    const { text, details } = await run(tool(createMemorySearchTool), { query: "tea" });
    expect(details).toMatchObject({ attempt: 1, maxAttempts: 5 });
    expect(text).toContain("Search attempt 1 of 5.");
    expect(text).toContain("call memory_search with attempt=2.");
    expect(client.callsTo("searchMemories")[0]?.params).not.toHaveProperty("attempt");
  });

  it.each([
    [1, false], [4, false], [1, true], [4, true],
  ] as const)("guides a rewrite at attempt %i with hits=%s", async (attempt, hasHits) => {
    const { client, tool } = setup();
    if (hasHits) client.seed(owner.workspaceId, ["Prefers tea"]);
    const { text, details } = await run(tool(createMemorySearchTool), { query: "tea", attempt });
    const guidance = text.split("</relevant-memories>\n")[1] ?? "";
    expect(guidance).toMatch(new RegExp(`^Search attempt ${attempt} of 5\\.`));
    expect(guidance.includes("Verify each memory is relevant")).toBe(hasHits);
    expect(guidance.toLowerCase()).toContain("rewrite the query");
    expect(guidance).toContain(`call memory_search with attempt=${attempt + 1}.`);
    expect(guidance).not.toContain("Search limit reached");
    expect(details).toMatchObject({ attempt, maxAttempts: 5, count: hasHits ? 1 : 0 });
    expect(client.callsTo("searchMemories")).toHaveLength(1);
    expect(client.callsTo("searchMemories")[0]?.params).not.toHaveProperty("attempt");
  });

  it.each([false, true])("stops at attempt 5 with hits=%s", async (hasHits) => {
    const { client, tool } = setup();
    if (hasHits) client.seed(owner.workspaceId, ["Prefers tea"]);
    const { text, details } = await run(tool(createMemorySearchTool), { query: "tea", attempt: 5 });
    const guidance = text.split("</relevant-memories>\n")[1] ?? "";
    expect(guidance).toMatch(/^Search attempt 5 of 5\. Search limit reached\./);
    expect(guidance.includes("Verify each memory is relevant")).toBe(hasHits);
    expect(guidance).toMatch(/Stop searching/);
    expect(text).not.toContain("attempt=6");
    expect(text.toLowerCase()).not.toContain("rewrite the query");
    expect(details).toMatchObject({ attempt: 5, maxAttempts: 5 });
  });

  it.each([0, 6, -1, 1.5, "2", true, false, null])("rejects invalid attempt %s before backend dispatch", async (attempt) => {
    const { client, tool } = setup();
    await expect(run(tool(createMemorySearchTool), { query: "tea", attempt })).rejects.toThrow(MemoryInputError);
    expect(client.callsTo("searchMemories")).toEqual([]);
  });

  it("keeps score warnings between the untrusted fence and search guidance", async () => {
    const { client, tool } = setup();
    client.seed(owner.workspaceId, ["Prefers tea"]);
    client.scoreKind = "rrf";
    const { text, details } = await run(tool(createMemorySearchTool), { query: "tea", minScore: 0.5 });
    expect(details).toHaveProperty("warning", expect.any(String));
    expect(text).toMatch(/<\/relevant-memories>\n[^\n]+\nSearch attempt 1 of 5\./);
    expect(text).toContain("minScore");
  });

  it("allows five caller-scripted rewritten searches and rejects a sixth", async () => {
    const { client, tool } = setup();
    const search = tool(createMemorySearchTool);
    const queries = ["tea", "drink preferences", "beverages", "green tea", "favorite drink"];
    for (const [index, query] of queries.entries()) {
      const { details } = await run(search, { query, attempt: index + 1 });
      expect(details).toMatchObject({ attempt: index + 1, maxAttempts: 5 });
    }
    expect(client.callsTo("searchMemories")).toHaveLength(5);
    expect(client.callsTo("searchMemories").map(call => call.params)).toEqual(
      queries.map(query => ({ workspaceId: owner.workspaceId, query, limit: smart.defaultMaxResults })),
    );
    await expect(run(search, { query: "another drink", attempt: 6 })).rejects.toThrow(MemoryInputError);
    expect(client.callsTo("searchMemories")).toHaveLength(5);
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
    expect(client.callsTo("addMemories")[0]?.params).toMatchObject({ dryRun: false, allowReplace });
    expect(client.records(owner.workspaceId)).toHaveLength(1);
  });

  it("previews a write without storing it", async () => {
    const { client, tool } = setup();
    const { text } = await run(tool(createMemoryPreviewTool), { memories: ["Prefers tea"] });
    expect(text).toMatch(/^Preview only, nothing stored\./);
    expect(client.callsTo("addMemories")[0]?.params).toMatchObject({ dryRun: true });
    expect(client.records(owner.workspaceId)).toEqual([]);
  });

  it("deduplicates a retried flush write and never replaces during a flush", async () => {
    const { client, tool } = setup({ ctx: { ...ownerContext, memoryFlush: { flushId: "flush-dedup" } } });
    const write = tool(createMemoryStoreTool);
    const params = { memories: ["Decided to ship on Friday"], replace: true };
    expect((await run(write, params)).details).toMatchObject({ storedCount: 1, deduplicated: false });
    expect((await run(write, params)).details).toMatchObject({ storedCount: 1, deduplicated: true });
    expect(client.callsTo("addMemories")).toHaveLength(1);
    expect(client.callsTo("addMemories")[0]?.params).toMatchObject({ allowReplace: false });
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
    const search = client.searchMemories.bind(client);
    vi.spyOn(client, "searchMemories").mockImplementation(async (params, options) => {
      const result = await search(params, options);
      slotOwner = "memory-core";
      return result;
    });
    await expect(run(tool(createMemorySearchTool), { query: "tea" })).rejects.toThrow(MemoryAccessDeniedError);
  });

  it("tells the agent to check before retrying a write with an unknown outcome", async () => {
    const { client, tool } = setup();
    client.failNext("addMemories", new ReflectLogError("outcome_unknown", "ReflectLog call aborted"));
    await expect(run(tool(createMemoryStoreTool), { memories: ["Prefers tea"] })).rejects.toMatchObject({
      code: "outcome_unknown",
      message: expect.stringContaining("Check with memory_search before retrying"),
    });
  });

  describe("flush writes", () => {
    const flushContext = (flushId = "flush-1"): OpenClawPluginToolContext => ({ ...ownerContext, memoryFlush: { flushId } });
    const params = { memories: ["Decided to ship on Friday"] };

    it("share one write between invocations built from different contexts and both see the result", async () => {
      const { client, tool } = setup();
      const write = client.deferNext("addMemories");
      const first = run(tool(createMemoryStoreTool, flushContext()), params);
      await write.dispatched;
      const second = run(tool(createMemoryStoreTool, flushContext()), params);
      write.resolve();
      const outcomes = await Promise.all([first, second]);
      expect(outcomes.map(({ details }) => details)).toEqual([
        expect.objectContaining({ storedCount: 1, deduplicated: false }),
        expect.objectContaining({ storedCount: 1, deduplicated: true }),
      ]);
      expect(outcomes[1]?.text).toContain("deduplicated: true.");
      expect(client.callsTo("addMemories")).toHaveLength(1);
      expect(client.callsTo("addMemories")[0]?.params).toMatchObject({ dryRun: false, allowReplace: false });
    });

    it("keeps dedup per service: a second service built for the same flush writes on its own", async () => {
      const first = setup({ ctx: flushContext() });
      const second = setup({ ctx: flushContext() });
      await run(first.tool(createMemoryStoreTool), params);
      expect((await run(second.tool(createMemoryStoreTool), params)).details).toMatchObject({ deduplicated: false });
      expect(first.client.callsTo("addMemories")).toHaveLength(1);
      expect(second.client.callsTo("addMemories")).toHaveLength(1);
    });

    it("withholds the result from a waiter whose audience is revoked at release while the other still gets it", async () => {
      const { client, tool } = setup();
      const state = { revoked: false };
      const write = client.deferNext("addMemories");
      const revokable: OpenClawPluginToolContext = {
        ...flushContext(), assertMemoryAudienceCurrent: () => { if (state.revoked) throw new Error("audience revoked"); },
      };
      const authorized = run(tool(createMemoryStoreTool, flushContext()), params);
      await write.dispatched;
      const revoked = run(tool(createMemoryStoreTool, revokable), params);
      state.revoked = true;
      write.resolve();
      await expect(revoked).rejects.toThrow("audience revoked");
      expect((await authorized).details).toMatchObject({ storedCount: 1 });
      expect(client.callsTo("addMemories")).toHaveLength(1);
    });

    it("does not cancel the write for an authorized waiter when the invocation that started it is cancelled", async () => {
      const { client, tool } = setup();
      const write = client.deferNext("addMemories");
      const controller = new AbortController();
      const cancelled = run(tool(createMemoryStoreTool, flushContext()), params, controller.signal);
      await write.dispatched;
      const other = run(tool(createMemoryStoreTool, flushContext()), params);
      controller.abort(new Error("host cancelled"));
      await expect(cancelled).rejects.toThrow("host cancelled");
      write.resolve();
      expect((await other).details).toMatchObject({ storedCount: 1, deduplicated: true });
      expect(client.callsTo("addMemories")).toHaveLength(1);
      expect(client.records(owner.workspaceId)).toHaveLength(1);
    });

    it("does not dispatch for a waiter revoked before the connection came up, and a later retry can", async () => {
      const { client, tool } = setup();
      const state = { revoked: false };
      const connecting = client.holdConnect("addMemories");
      const only = run(tool(createMemoryStoreTool, {
        ...flushContext(), assertInvocationCurrent: () => { if (state.revoked) throw new Error("invocation revoked"); },
      }), params);
      state.revoked = true;
      connecting.release();
      await expect(only).rejects.toThrow("invocation revoked");
      expect(client.calls).toEqual([]);
      expect((await run(tool(createMemoryStoreTool, flushContext()), params)).details).toMatchObject({ deduplicated: false });
      expect(client.callsTo("addMemories")).toHaveLength(1);
    });

    it("tells the agent not to retry an unconfirmed flush and never redispatches it", async () => {
      const { client, tool } = setup({ ctx: flushContext() });
      const write = client.deferNext("addMemories");
      const first = run(tool(createMemoryStoreTool), params);
      await write.dispatched;
      write.applyThenFail(new Error("response lost"));
      const check = { code: "outcome_unknown", message: expect.stringContaining("Check with memory_search before retrying") };
      await expect(first).rejects.toMatchObject(check);
      await expect(run(tool(createMemoryStoreTool), params)).rejects.toMatchObject(check);
      expect(client.callsTo("addMemories")).toHaveLength(1);
    });

    it("explains a full ledger without echoing memory text and sends nothing", async () => {
      const { client, tool } = setup({ ctx: flushContext(), flush: { maxEntries: 1 } });
      const write = client.deferNext("addMemories");
      const first = run(tool(createMemoryStoreTool), params);
      await write.dispatched;
      const rejected = run(tool(createMemoryStoreTool), { memories: ["Secret second fact"] });
      await expect(rejected).rejects.toThrow(FlushCapacityError);
      await expect(rejected).rejects.toThrow(/nothing was sent/);
      await expect(rejected).rejects.not.toThrow(/Secret second fact/);
      write.resolve();
      await first;
      expect(client.callsTo("addMemories")).toHaveLength(1);
    });

    it("explains a disposed backend and sends nothing", async () => {
      const { client, service, tool } = setup({ ctx: flushContext() });
      service.dispose();
      await expect(run(tool(createMemoryStoreTool), params)).rejects.toThrow(FlushDisposedError);
      await expect(run(tool(createMemoryStoreTool), params)).rejects.toThrow(/nothing was sent/);
      expect(client.calls).toEqual([]);
    });

    it("leaves non-flush and preview writes outside the flush ledger", async () => {
      const { client, tool } = setup();
      await run(tool(createMemoryStoreTool), params);
      await run(tool(createMemoryStoreTool), params);
      await run(tool(createMemoryPreviewTool, flushContext()), params);
      await run(tool(createMemoryPreviewTool, flushContext()), params);
      expect(client.callsTo("addMemories").map((call) => (call.params as { dryRun: boolean }).dryRun))
        .toEqual([false, false, true, true]);
    });
  });
});
