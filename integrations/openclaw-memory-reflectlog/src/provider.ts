import { ownerScope, resolveProviderScope } from "./access.js";
import { PLUGIN_ID, type ReflectLogPluginConfig } from "./config.js";
import type { MemoryHealth, MemoryProviderRuntime } from "./host-types.js";
import type { MemoryService } from "./service.js";

export function createProviderRuntime(deps: {
  readonly config: ReflectLogPluginConfig | null;
  readonly error?: string;
  readonly getService: () => MemoryService;
  readonly closeClient?: () => Promise<void>;
}): MemoryProviderRuntime {
  return {
    async open(params) {
      const config = deps.config;
      if (!config) return { provider: null, error: deps.error ?? "Invalid ReflectLog configuration" };
      const controller = new AbortController();
      const signal = params.context.signal
        ? AbortSignal.any([controller.signal, params.context.signal]) : controller.signal;
      const guard = {
        signal,
        assertCurrent() { signal.throwIfAborted(); params.context.assertCurrent(); },
      };
      return { provider: {
        capabilities: { sources: ["memory"], pagination: false, candidates: [], projectFilter: false },
        async search(request) {
          guard.assertCurrent();
          const scope = resolveProviderScope(config, params.agentId, params.context.authority);
          if (request.sources && !request.sources.includes("memory")) return { hits: [] };
          const result = await deps.getService().search(scope, request, guard);
          guard.assertCurrent();
          return {
            hits: result.hits.map(hit => ({
              reference: { providerId: PLUGIN_ID, id: hit.ref, revision: hit.revision },
              excerpt: hit.text.length > 1000 ? `${hit.text.slice(0, 999)}…` : hit.text,
              source: "memory" as const,
              ...(hit.score === null ? {} : { score: hit.score }),
            })),
            ...(result.warning ? { warning: result.warning } : {}),
          };
        },
        async get(request) {
          guard.assertCurrent();
          const scope = resolveProviderScope(config, params.agentId, params.context.authority);
          if (request.reference.providerId !== PLUGIN_ID) return { status: "not_found" };
          const result = await deps.getService().get(scope, {
            ref: request.reference.id,
            ...(request.reference.revision === undefined ? {} : { revision: request.reference.revision }),
          }, request, guard);
          guard.assertCurrent();
          switch (result.status) {
            case "not_found": return result;
            case "ok": {
              const { entry, ...page } = result;
              return { ...page, reference: { providerId: PLUGIN_ID, id: entry.ref, revision: entry.revision } };
            }
          }
        },
        async health() {
          try {
            guard.assertCurrent();
            const result = await deps.getService().health(ownerScope(config, params.agentId), guard);
            guard.assertCurrent();
            const statuses = { healthy: "ready", degraded: "degraded", unhealthy: "unavailable" } as const;
            const status = statuses[result.status];
            if (params.cfg.plugins?.entries?.["memory-core"]?.enabled !== false) {
              return { status: status === "unavailable" ? status : "degraded",
                message: "Set plugins.entries.memory-core.enabled=false to avoid conflicting memory guidance." };
            }
            return { status } satisfies MemoryHealth;
          } catch (error) {
            // Health is a text-free diagnostic boundary, including for host callers.
            return { status: "unavailable", message: error instanceof Error ? error.message : "ReflectLog health check failed" };
          }
        },
        async close() { controller.abort(); },
      } };
    },
    async closeAllMemorySearchManagers() { await deps.closeClient?.(); },
  };
}
