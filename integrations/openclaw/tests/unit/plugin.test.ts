import { readFileSync } from "node:fs";

import type { MemoryPluginCapability } from "openclaw/plugin-sdk/memory-host-core";
import type { OpenClawConfig } from "openclaw/plugin-sdk/plugin-entry";
import { describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { ownerScope } from "../../src/access.js";
import { configJsonSchema, PLUGIN_ID } from "../../src/config.js";
import { FlushDisposedError } from "../../src/flush-ledger.js";
import { registerReflectLogMemory, type MemoryPluginApi } from "../../src/plugin.js";
import { MemoryInputError } from "../../src/service.js";
import { MEMORY_TOOL_NAMES } from "../../src/tools.js";
import { FakeReflectLogClient, allowAll } from "../support/fake-client.js";

const manifest = z.object({
  id: z.string(),
  kind: z.string(),
  contracts: z.object({ tools: z.array(z.string()) }),
  toolMetadata: z.record(z.string(), z.object({ sideEffecting: z.boolean().optional() })),
  configSchema: z.unknown(),
}).parse(JSON.parse(readFileSync(new URL("../../openclaw.plugin.json", import.meta.url), "utf8")));

const replacing: OpenClawConfig = {
  plugins: { slots: { memory: PLUGIN_ID }, entries: { "memory-core": { enabled: false } } },
};

/** A client whose close() the test can hold open, to race a stop against a new lifecycle. */
class SlowCloseClient extends FakeReflectLogClient {
  readonly closing = Promise.withResolvers<void>();

  override async close(): Promise<void> {
    await this.closing.promise;
    await super.close();
  }
}

function setup(pluginConfig: Record<string, unknown> = {}, config: OpenClawConfig = replacing,
  makeClient: () => FakeReflectLogClient = () => new FakeReflectLogClient()) {
  const capabilities: MemoryPluginCapability[] = [];
  const services: Parameters<MemoryPluginApi["registerService"]>[0][] = [];
  const registerTool = vi.fn<MemoryPluginApi["registerTool"]>();
  const logger = { info: vi.fn(), warn: vi.fn(), error: vi.fn() };
  const api: MemoryPluginApi = {
    config, pluginConfig, logger, registerTool,
    registerMemoryCapability: (capability) => { capabilities.push(capability); },
    registerService: (service) => { services.push(service); },
  };
  const client = makeClient();
  const clients: FakeReflectLogClient[] = [];
  const createClient = vi.fn(() => {
    const created = clients.length === 0 ? client : makeClient();
    clients.push(created);
    return created;
  });
  const registered = registerReflectLogMemory(api, { createClient });
  return {
    registered, capabilities, services, registerTool, logger, client, createClient, clients,
  };
}

describe("plugin registration", () => {
  it("declares every registered tool and the config schema in the manifest", () => {
    expect(manifest.id).toBe(PLUGIN_ID);
    expect(manifest.kind).toBe("memory");
    expect([...manifest.contracts.tools].sort()).toEqual([...MEMORY_TOOL_NAMES].sort());
    expect(Object.entries(manifest.toolMetadata).filter(([, meta]) => meta.sideEffecting).map(([name]) => name).sort())
      .toEqual(["memory_forget", "memory_store"]);
    expect(manifest.configSchema).toEqual(configJsonSchema);
  });

  it("owns the native provider, prompt, flush and recall surfaces", () => {
    const { capabilities } = setup();
    expect(capabilities).toHaveLength(1);
    expect(capabilities[0]).toMatchObject({
      providerRuntime: expect.any(Object),
      promptBuilder: expect.any(Function),
      providerFlushPlanResolver: expect.any(Function),
      recallToolNames: ["memory_search", "memory_get"],
      deterministicRecallToolName: "memory_search",
      supportsPrivateTranscriptRecall: false,
    });
    expect(capabilities[0]?.providerFlushPlanResolver?.({})).toMatchObject({ persistenceToolNames: ["memory_store"] });
  });

  it("registers each memory tool under its declared name", () => {
    const { registerTool } = setup();
    expect(registerTool.mock.calls.map(([, options]) => options)).toEqual(MEMORY_TOOL_NAMES.map((name) => ({ name })));
  });

  it("connects lazily and closes the client when the service stops", async () => {
    const { registered, services, client, createClient } = setup();
    expect(createClient).not.toHaveBeenCalled();
    expect(registered.getService()).toBe(registered.getService());
    expect(createClient).toHaveBeenCalledTimes(1);
    expect(services.map((service) => service.id)).toEqual([PLUGIN_ID]);
    const service = services[0];
    if (service === undefined || service.apiVersion === 2) throw new Error("expected a version 1 service");
    await service.stop?.({ config: replacing, stateDir: "/tmp", logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() } });
    expect(client.closeCount).toBe(1);
  });

  describe("backend lifecycle", () => {
    const stopContext = { config: replacing, stateDir: "/tmp", logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() } };
    const stopOf = (services: ReturnType<typeof setup>["services"]) => {
      const service = services[0];
      if (service === undefined || service.apiVersion === 2) throw new Error("expected a version 1 service");
      return () => service.stop?.(stopContext);
    };
    const owner = (registered: ReturnType<typeof setup>["registered"]) => {
      const config = registered.config;
      if (config === null) throw new Error("expected a valid configuration");
      return ownerScope(config, "main");
    };

    it("disposes the service and closes the client on stop, and builds a fresh lifecycle afterwards", async () => {
      const { registered, services, clients, createClient } = setup();
      const old = registered.getService();
      const scope = owner(registered);
      await stopOf(services)();
      expect(clients[0]?.closeCount).toBe(1);
      await expect(old.flush(scope, "f1", ["Decided to ship"], allowAll)).rejects.toBeInstanceOf(FlushDisposedError);
      expect(clients[0]?.calls).toEqual([]);
      const fresh = registered.getService();
      expect(fresh).not.toBe(old);
      expect(createClient).toHaveBeenCalledTimes(2);
      await expect(fresh.flush(scope, "f1", ["Decided to ship"], allowAll)).resolves.toMatchObject({ deduplicated: false });
      expect(clients[1]?.callsTo("addMemories")).toHaveLength(1);
    });

    it("does not carry flush dedup across lifecycles", async () => {
      const { registered, services, clients } = setup();
      const scope = owner(registered);
      await registered.getService().flush(scope, "f1", ["Decided to ship"], allowAll);
      await stopOf(services)();
      await expect(registered.getService().flush(scope, "f1", ["Decided to ship"], allowAll))
        .resolves.toMatchObject({ deduplicated: false });
      expect(clients.map((client) => client.callsTo("addMemories").length)).toEqual([1, 1]);
    });

    it("settles a pending flush as outcome_unknown when the lifecycle ends", async () => {
      const { registered, services, clients } = setup();
      const scope = owner(registered);
      const service = registered.getService();
      const deferred = clients[0]?.deferNext("addMemories");
      const pending = service.flush(scope, "f1", ["Decided to ship"], allowAll);
      await deferred?.dispatched;
      await stopOf(services)();
      await expect(pending).rejects.toMatchObject({ code: "outcome_unknown" });
    });

    it("does not let a stop racing with getService erase the new lifecycle", async () => {
      const { registered, services, clients, createClient } = setup({}, replacing, () => new SlowCloseClient());
      const old = registered.getService();
      const stopping = stopOf(services)();
      const fresh = registered.getService();
      expect(fresh).not.toBe(old);
      expect(createClient).toHaveBeenCalledTimes(2);
      const [oldClient] = clients;
      if (!(oldClient instanceof SlowCloseClient)) throw new Error("expected the slow client");
      oldClient.closing.resolve();
      await stopping;
      expect(registered.getService()).toBe(fresh);
      expect(createClient).toHaveBeenCalledTimes(2);
      expect(clients[1]?.closeCount).toBe(0);
      expect(oldClient.closeCount).toBe(1);
      await expect(fresh.flush(owner(registered), "f1", ["Decided to ship"], allowAll)).resolves.toMatchObject({ deduplicated: false });
    });

    it("ends the lifecycle when the host retires the memory runtime, not only on service stop", async () => {
      const { registered, capabilities, clients } = setup();
      const old = registered.getService();
      await capabilities[0]?.providerRuntime?.closeAllMemorySearchManagers?.();
      expect(clients[0]?.closeCount).toBe(1);
      await expect(old.flush(owner(registered), "f1", ["Decided to ship"], allowAll)).rejects.toBeInstanceOf(FlushDisposedError);
      expect(registered.getService()).not.toBe(old);
    });

    it("tolerates a stop before any service was created", async () => {
      const { services, createClient } = setup();
      await expect(stopOf(services)()).resolves.toBeUndefined();
      expect(createClient).not.toHaveBeenCalled();
    });
  });

  it("warns until memory-core is disabled and the slot selects this plugin", () => {
    expect(setup().logger.warn).not.toHaveBeenCalled();
    const { logger } = setup({}, { plugins: { slots: { memory: "memory-core" } } });
    expect(logger.warn.mock.calls.map(([message]) => String(message))).toEqual([
      expect.stringContaining("memory-core.enabled=false"),
      expect.stringContaining("plugins.slots.memory=memory-reflectlog"),
    ]);
  });

  it("keeps the slot but offers nothing usable when the configuration is rejected", async () => {
    const { registered, capabilities, registerTool, services, logger, createClient } = setup({ transport: "carrier-pigeon" });
    expect(registered.config).toBeNull();
    expect(logger.error).toHaveBeenCalledWith(expect.stringContaining("ReflectLog configuration rejected"));
    expect(registerTool).not.toHaveBeenCalled();
    expect(services).toEqual([]);
    expect(capabilities[0]?.promptBuilder?.({ availableTools: new Set(MEMORY_TOOL_NAMES) })).toEqual([]);
    expect(capabilities[0]).not.toHaveProperty("recallToolNames");
    const opened = await capabilities[0]?.providerRuntime?.open({
      cfg: replacing, agentId: "main",
      context: { authority: { kind: "operator", scopes: [] }, assertCurrent() {} },
    });
    expect(opened).toMatchObject({ provider: null, error: expect.stringContaining("transport") });
    expect(() => registered.getService()).toThrow(MemoryInputError);
    expect(createClient).not.toHaveBeenCalled();
  });
});
