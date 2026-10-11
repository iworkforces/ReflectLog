import { describe, expect, it, vi } from "vitest";

import { MemoryAccessDeniedError, conversationScope, ownerScope } from "../../src/access.js";
import { PLUGIN_ID, parseConfig } from "../../src/config.js";
import type { MemoryCallerAuthority, MemoryProviderOpenParams } from "../../src/host-types.js";
import { createProviderRuntime } from "../../src/provider.js";
import { ReflectLogError } from "../../src/reflectlog/types.js";
import { MemoryService } from "../../src/service.js";
import { FakeReflectLogClient } from "../support/fake-client.js";

const parsed = parseConfig({});
if (!parsed.ok) throw new Error(parsed.error);
const config = parsed.config;
const owner = ownerScope(config, "main");
const sessionKey = "agent:main:telegram:group:9";
const conversation = conversationScope(config, "main", sessionKey);
const configured = { plugins: { slots: { memory: PLUGIN_ID }, entries: { "memory-core": { enabled: false } } } };

const ownerAuthority: MemoryCallerAuthority = {
  kind: "session", sessionKey: "agent:main:main", sandboxed: false,
  audience: { kind: "owner-private", agentId: "main" },
};
const conversationAuthority: MemoryCallerAuthority = {
  kind: "session", sessionKey, sandboxed: false,
  audience: { kind: "conversation", agentId: "main", sessionKey, sessionId: "s-1" },
};
const hostAuthority: MemoryCallerAuthority = { kind: "host", operation: "memory-wiki.query" };

function first<T>(items: readonly T[]): T {
  const item = items[0];
  if (item === undefined) throw new Error("expected at least one item");
  return item;
}

function setup() {
  const client = new FakeReflectLogClient();
  const service = new MemoryService(client, config);
  const closeClient = vi.fn(async () => {});
  const runtime = createProviderRuntime({ config, getService: () => service, closeClient });
  const open = async (options: {
    authority?: MemoryCallerAuthority; cfg?: MemoryProviderOpenParams["cfg"];
    isCurrent?: () => boolean; signal?: AbortSignal;
  } = {}) => {
    const result = await runtime.open({
      cfg: options.cfg ?? configured,
      agentId: "main",
      context: {
        authority: options.authority ?? ownerAuthority,
        assertCurrent() {
          if (options.isCurrent?.() === false) throw new Error("authority revoked");
        },
        ...(options.signal ? { signal: options.signal } : {}),
      },
    });
    if (!result.provider) throw new Error(result.error ?? "provider unavailable");
    return result.provider;
  };
  return { client, runtime, closeClient, open };
}

describe("ReflectLog provider runtime", () => {
  it("refuses to open with a rejected configuration", async () => {
    const runtime = createProviderRuntime({
      config: null, error: "transport: invalid option",
      getService: () => { throw new Error("service must not be built"); },
    });
    await expect(runtime.open({ cfg: configured, agentId: "main", context: { authority: ownerAuthority, assertCurrent() {} } }))
      .resolves.toEqual({ provider: null, error: "transport: invalid option" });
  });

  it("advertises memory-only capabilities without pagination, candidates or project filters", async () => {
    const provider = await setup().open();
    expect(provider.capabilities).toEqual({ sources: ["memory"], pagination: false, candidates: [], projectFilter: false });
  });

  it("returns plugin references that resolve through get", async () => {
    const { client, open } = setup();
    client.seed(owner.workspaceId, ["Prefers green tea", "Uses vim"]);
    const provider = await open();
    const page = await provider.search({ query: "green tea" });
    expect(page).toEqual({ hits: [{
      reference: { providerId: PLUGIN_ID, id: `${owner.tag}:1`, revision: expect.any(String) },
      excerpt: "Prefers green tea", source: "memory", score: 1,
    }] });
    const { reference } = first(page.hits);
    await expect(provider.get({ reference })).resolves.toEqual({ status: "ok", reference, text: "Prefers green tea" });
  });

  it("returns a bounded line window", async () => {
    const { client, open } = setup();
    client.seed(owner.workspaceId, ["one\ntwo\nthree"]);
    const provider = await open();
    const { reference } = first((await provider.search({ query: "two" })).hits);
    await expect(provider.get({ reference, from: 2, lines: 1 })).resolves.toEqual({
      status: "ok", reference, text: "two", from: 2, lines: 1, truncated: true, nextFrom: 3,
    });
  });

  it("truncates long excerpts", async () => {
    const { client, open } = setup();
    client.seed(owner.workspaceId, [`tea ${"x".repeat(1200)}`]);
    const { excerpt } = first((await (await open()).search({ query: "tea" })).hits);
    expect(excerpt).toHaveLength(1000);
    expect(excerpt.endsWith("…")).toBe(true);
  });

  it("omits incomparable scores and refuses to threshold them", async () => {
    const { client, open } = setup();
    client.seed(owner.workspaceId, ["Prefers tea"]);
    client.scoreKind = "rrf";
    const provider = await open();
    const unthresholded = await provider.search({ query: "tea" });
    expect(first(unthresholded.hits)).not.toHaveProperty("score");
    const thresholded = await provider.search({ query: "tea", minScore: 0.5 });
    expect(thresholded.hits).toEqual([]);
    expect(thresholded.warning).toContain("minScore");
  });

  it("answers non-memory sources with no hits and no backend call", async () => {
    const { client, open } = setup();
    await expect((await open()).search({ query: "tea", sources: ["sessions"] })).resolves.toEqual({ hits: [] });
    expect(client.calls).toEqual([]);
  });

  it("denies host operations memory content without touching ReflectLog", async () => {
    const { client, open } = setup();
    client.seed(owner.workspaceId, ["Prefers tea"]);
    const provider = await open({ authority: hostAuthority });
    await expect(provider.search({ query: "tea" })).rejects.toThrow(MemoryAccessDeniedError);
    await expect(provider.get({ reference: { providerId: PLUGIN_ID, id: `${owner.tag}:1` } }))
      .rejects.toThrow(MemoryAccessDeniedError);
    expect(client.calls).toEqual([]);
  });

  it("keeps conversation audiences out of owner memory", async () => {
    const { client, open } = setup();
    client.seed(owner.workspaceId, ["owner secret"]);
    client.seed(conversation.workspaceId, ["group fact"]);
    const provider = await open({ authority: conversationAuthority });
    expect((await provider.search({ query: "owner secret group fact" })).hits.map((hit) => hit.excerpt)).toEqual(["group fact"]);
    await expect(provider.get({ reference: { providerId: PLUGIN_ID, id: `${owner.tag}:1` } }))
      .resolves.toEqual({ status: "not_found" });
    expect(client.callsTo("getRecords")).toEqual([]);
  });

  it("does not resolve foreign, stale or removed references", async () => {
    const { client, open } = setup();
    client.seed(owner.workspaceId, ["Prefers tea"]);
    const provider = await open();
    const { reference } = first((await provider.search({ query: "tea" })).hits);
    await expect(provider.get({ reference: { ...reference, providerId: "memory-core" } })).resolves.toEqual({ status: "not_found" });
    await expect(provider.get({ reference: { ...reference, revision: "0".repeat(64) } })).resolves.toEqual({ status: "not_found" });
    await client.removeRecords({ workspaceId: owner.workspaceId, ids: [1] });
    await expect(provider.get({ reference })).resolves.toEqual({ status: "not_found" });
  });

  it("withholds results when authority is revoked during the backend call", async () => {
    const { client, open } = setup();
    client.seed(owner.workspaceId, ["Prefers tea"]);
    let current = true;
    const search = client.searchRecords.bind(client);
    vi.spyOn(client, "searchRecords").mockImplementation(async (params, options) => {
      const result = await search(params, options);
      current = false;
      return result;
    });
    const provider = await open({ isCurrent: () => current });
    await expect(provider.search({ query: "tea" })).rejects.toThrow("authority revoked");
  });

  it("honours the caller's abort signal", async () => {
    const { client, open } = setup();
    const provider = await open({ signal: AbortSignal.abort() });
    await expect(provider.search({ query: "tea" })).rejects.toMatchObject({ name: "AbortError" });
    expect(client.calls).toEqual([]);
  });

  it("closes only the handle it was called on", async () => {
    const { client, open } = setup();
    client.seed(owner.workspaceId, ["Prefers tea"]);
    const closed = await open();
    const live = await open();
    await closed.close();
    await expect(closed.search({ query: "tea" })).rejects.toMatchObject({ name: "AbortError" });
    expect((await live.search({ query: "tea" })).hits).toHaveLength(1);
  });

  it("closes the shared client when the host closes every manager", async () => {
    const { runtime, closeClient } = setup();
    await runtime.closeAllMemorySearchManagers?.();
    expect(closeClient).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["healthy", "ready"], ["degraded", "degraded"], ["unhealthy", "unavailable"],
  ] as const)("maps %s backend health to %s for any caller", async (backend, status) => {
    const { client, open } = setup();
    client.healthStatus = backend;
    await expect((await open({ authority: hostAuthority })).health()).resolves.toEqual({ status });
  });

  it("reports degraded health while memory-core is still enabled", async () => {
    const health = await (await setup().open({ cfg: { plugins: { slots: { memory: PLUGIN_ID } } } })).health();
    expect(health).toEqual({ status: "degraded", message: expect.stringContaining("memory-core.enabled=false") });
  });

  it("reports an unreachable backend as unavailable instead of throwing", async () => {
    const { client, open } = setup();
    client.failNext("healthCheck", new ReflectLogError("unavailable", "ReflectLog connection unavailable"));
    await expect((await open()).health()).resolves.toEqual({ status: "unavailable", message: "ReflectLog connection unavailable" });
  });
});
