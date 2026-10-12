import { z } from "zod";

export const PLUGIN_ID = "memory-reflectlog";

export const DEFAULT_URL = "http://127.0.0.1:9103/mcp";
export const DEFAULT_AUTH_TOKEN_ENV = "REFLECTLOG_MCP_AUTH_TOKEN";
export const MAX_SEARCH_RESULTS = 50;

const AGENT_ID_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;
const NAMESPACE_RE = /^[a-z0-9][a-z0-9_.-]{0,31}$/;
const WORKSPACE_ID_RE = /^[a-z0-9_.-]{1,64}$/;
/** Shape of derived workspace ids; overrides must never collide with it. */
export const DERIVED_WORKSPACE_ID_RE = /^[pc]-[a-z0-9_-]{1,32}-[0-9a-f]{24}$/;
const ENV_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

export function isValidAgentId(value: string): boolean {
  return AGENT_ID_RE.test(value);
}

const configSchema = z
  .strictObject({
    transport: z.enum(["http"]).default("http"),
    url: z
      .url({ protocol: /^https?$/ })
      .default(DEFAULT_URL),
    authTokenEnv: z.string().regex(ENV_NAME_RE).default(DEFAULT_AUTH_TOKEN_ENV),
    namespace: z.string().regex(NAMESPACE_RE).default("openclaw"),
    workspaceOverrides: z
      .record(z.string().regex(AGENT_ID_RE), z.string().regex(WORKSPACE_ID_RE))
      .default({}),
    defaultMaxResults: z.int().min(1).max(MAX_SEARCH_RESULTS).default(6),
    connectTimeoutMs: z.int().min(1_000).max(600_000).default(120_000),
    requestTimeoutMs: z.int().min(1_000).max(600_000).default(60_000),
    smartReplace: z.boolean().default(false),
  })
  .superRefine((value, ctx) => {
    const seen = new Map<string, string>();
    for (const [agentId, workspaceId] of Object.entries(value.workspaceOverrides)) {
      if (workspaceId === "." || workspaceId === "..") {
        ctx.addIssue({
          code: "custom",
          path: ["workspaceOverrides", agentId],
          message: "workspace id must not be '.' or '..'",
        });
      }
      if (DERIVED_WORKSPACE_ID_RE.test(workspaceId)) {
        ctx.addIssue({
          code: "custom",
          path: ["workspaceOverrides", agentId],
          message: "workspace id must not use the reserved derived-id shape",
        });
      }
      const previous = seen.get(workspaceId);
      if (previous !== undefined) {
        ctx.addIssue({
          code: "custom",
          path: ["workspaceOverrides", agentId],
          message: `workspace id is already assigned to agent '${previous}'`,
        });
      }
      seen.set(workspaceId, agentId);
    }
  });

export type ReflectLogPluginConfig = Readonly<z.output<typeof configSchema>>;

export type ConfigParseResult =
  | { readonly ok: true; readonly config: ReflectLogPluginConfig }
  | { readonly ok: false; readonly error: string };

export function parseConfig(raw: unknown): ConfigParseResult {
  const parsed = configSchema.safeParse(raw ?? {});
  if (!parsed.success) {
    return { ok: false, error: z.prettifyError(parsed.error) };
  }
  return { ok: true, config: Object.freeze(parsed.data) };
}

/** JSON Schema mirrored in `openclaw.plugin.json`; kept here for runtime registration. */
export const configJsonSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    transport: { type: "string", enum: ["http"], default: "http" },
    url: { type: "string", default: DEFAULT_URL },
    authTokenEnv: { type: "string", pattern: ENV_NAME_RE.source, default: DEFAULT_AUTH_TOKEN_ENV },
    namespace: { type: "string", pattern: NAMESPACE_RE.source, default: "openclaw" },
    workspaceOverrides: {
      type: "object",
      propertyNames: { pattern: AGENT_ID_RE.source },
      additionalProperties: { type: "string", pattern: WORKSPACE_ID_RE.source },
    },
    defaultMaxResults: { type: "integer", minimum: 1, maximum: MAX_SEARCH_RESULTS, default: 6 },
    connectTimeoutMs: { type: "integer", minimum: 1000, maximum: 600000, default: 120000 },
    requestTimeoutMs: { type: "integer", minimum: 1000, maximum: 600000, default: 60000 },
    smartReplace: { type: "boolean", default: false },
  },
} as const;
