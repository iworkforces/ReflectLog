import { buildJsonPluginConfigSchema, definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

import { registerReflectLogCli } from "./cli.js";
import { configJsonSchema, PLUGIN_ID } from "./config.js";
import { registerReflectLogMemory } from "./plugin.js";
import { createReflectLogClient } from "./reflectlog/mcp-client.js";

export default definePluginEntry({
  id: PLUGIN_ID,
  name: "ReflectLog Memory",
  description: "Replaces OpenClaw's built-in memory with the ReflectLog MCP server.",
  configSchema: () => buildJsonPluginConfigSchema(configJsonSchema),
  register(api) {
    const memory = registerReflectLogMemory(api, { createClient: (config) => createReflectLogClient(config) });
    if (memory.config) {
      registerReflectLogCli(api, { config: memory.config, getService: memory.getService });
    }
  },
});
