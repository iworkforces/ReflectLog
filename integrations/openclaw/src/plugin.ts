import type { OpenClawPluginApi } from "openclaw/plugin-sdk/plugin-entry";
import { parseConfig, PLUGIN_ID, type ReflectLogPluginConfig } from "./config.js";
import { resolveFlushPlan } from "./flush.js";
import { buildMemoryPromptSection } from "./prompt.js";
import { createProviderRuntime } from "./provider.js";
import type { ReflectLogClient } from "./reflectlog/types.js";
import { MemoryInputError, MemoryService } from "./service.js";
import { createMemorySearchTool, createMemoryGetTool, createMemoryStoreTool, createMemoryPreviewTool, createMemoryForgetTool, createMemoryListTool, MEMORY_TOOL_NAMES } from "./tools.js";

export type RegisteredMemory = { readonly config: ReflectLogPluginConfig | null; getService(): MemoryService };
export type MemoryPluginApi = Pick<OpenClawPluginApi,
  "config" | "pluginConfig" | "logger" | "registerMemoryCapability" | "registerTool" | "registerService">;

export function registerReflectLogMemory(api: MemoryPluginApi, deps: {
  readonly createClient: (config: ReflectLogPluginConfig) => ReflectLogClient;
}): RegisteredMemory {
  const parsed = parseConfig(api.pluginConfig);
  const config = parsed.ok ? parsed.config : null;
  const error = parsed.ok ? undefined : parsed.error;
  let client: ReflectLogClient | undefined;
  let service: MemoryService | undefined;
  const getService = () => {
    if (!config) throw new MemoryInputError(error ?? "Invalid ReflectLog configuration");
    if (!service) { client = deps.createClient(config); service = new MemoryService(client, config); }
    return service;
  };
  const closeClient = async () => { await client?.close(); };
  const providerRuntime = createProviderRuntime({ config, ...(error ? { error } : {}), getService, closeClient });
  if (!config) {
    api.logger.error(`ReflectLog configuration rejected: ${error}`);
    api.registerMemoryCapability({ providerRuntime, promptBuilder: () => [] });
    return { config, getService };
  }
  api.registerMemoryCapability({ providerRuntime, promptBuilder: buildMemoryPromptSection,
    providerFlushPlanResolver: params => resolveFlushPlan({ ...params, config }),
    recallToolNames: ["memory_search", "memory_get"], deterministicRecallToolName: "memory_search", supportsPrivateTranscriptRecall: false });
  const toolDeps = { api, config, getService };
  const factories = [createMemorySearchTool, createMemoryGetTool, createMemoryStoreTool, createMemoryPreviewTool, createMemoryForgetTool, createMemoryListTool];
  factories.forEach((create, index) => {
    const name = MEMORY_TOOL_NAMES[index];
    if (name) api.registerTool(create(toolDeps), { name });
  });
  api.registerService({ id: PLUGIN_ID, start() {}, stop: closeClient });
  if (api.config.plugins?.entries?.["memory-core"]?.enabled !== false) api.logger.warn("ReflectLog: set plugins.entries.memory-core.enabled=false to avoid conflicting memory guidance.");
  if (api.config.plugins?.slots?.memory !== PLUGIN_ID) api.logger.warn("ReflectLog: set plugins.slots.memory=memory-reflectlog to enable memory tools.");
  return { config, getService };
}
