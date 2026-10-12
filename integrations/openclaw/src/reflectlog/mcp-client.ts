import {
  Client, ProtocolError, SdkError, SdkErrorCode, SdkHttpError,
  StreamableHTTPClientTransport, UnauthorizedError,
} from "@modelcontextprotocol/client";
import type { z } from "zod";
import type { ReflectLogPluginConfig } from "../config.js";
import {
  addMemoriesSchema, getMemoryByIdSchema, getAllMemoriesSchema,
  parseToolResponse, deleteMemoriesSchema, searchMemoriesSchema,
} from "./parse.js";
import type { ToolResponse } from "./parse.js";
import { ReflectLogError, REQUIRED_REFLECTLOG_TOOLS } from "./types.js";
import type { CallOptions, ReflectLogClient, ReflectLogErrorCode } from "./types.js";
import { waitFor } from "./wait-for.js";

export interface McpConnection {
  listTools(options: { readonly signal: AbortSignal; readonly timeout: number }): Promise<{
    readonly tools: readonly { readonly name: string; readonly inputSchema: {
      readonly properties?: Record<string, unknown> | undefined;
    } }[];
  }>;
  callTool(params: { readonly name: string; readonly arguments: Record<string, unknown> },
    options: { readonly timeout: number; readonly signal?: AbortSignal }): Promise<ToolResponse>;
  close(): Promise<void>;
}
export type ConnectFn = (config: ReflectLogPluginConfig, options: {
  readonly signal: AbortSignal; readonly timeout: number; readonly headers: Record<string, string>;
}) => Promise<McpConnection>;
export type ReflectLogClientDeps = {
  readonly connect?: ConnectFn;
  readonly env?: Readonly<Record<string, string | undefined>>;
};

const connectSdk: ConnectFn = async (config, options) => {
  const client = new Client({ name: "openclaw-memory-reflectlog", version: "0.1.0" });
  const transport = new StreamableHTTPClientTransport(
    new URL(config.url), { requestInit: { headers: options.headers } });
  try {
    await client.connect(transport, { signal: options.signal, timeout: options.timeout });
    return {
      listTools: (requestOptions) => client.listTools(undefined, requestOptions),
      callTool: (params, requestOptions) => client.callTool(params, requestOptions),
      close: () => client.close(),
    };
  } catch (cause) {
    await Promise.allSettled([client.close()]);
    throw cause;
  }
};

type Live = { readonly generation: number; readonly connection: McpConnection };

export function createReflectLogClient(
  config: ReflectLogPluginConfig, deps: ReflectLogClientDeps = {},
): ReflectLogClient {
  const connect = deps.connect ?? connectSdk;
  const env = deps.env ?? process.env;
  let generation = 0;
  let live: Live | undefined;
  let pending: Promise<Live> | undefined;
  let controller: AbortController | undefined;

  async function invalidate(expected: number): Promise<void> {
    if (expected !== generation) return;
    generation++;
    const connection = live?.connection;
    live = undefined;
    pending = undefined;
    controller?.abort(new SdkError(SdkErrorCode.ConnectionClosed, "Connection closed"));
    controller = undefined;
    if (connection) await Promise.allSettled([connection.close()]);
  }

  function acquire(): Promise<Live> {
    if (live) return Promise.resolve(live);
    if (pending) return pending;
    const token = env[config.authTokenEnv];
    if (!token?.trim()) {
      return Promise.reject(new ReflectLogError("unauthorized",
        `set ${config.authTokenEnv} to the ReflectLog bearer token`));
    }
    const headers = { Authorization: `Bearer ${token}` };
    const current = ++generation;
    const abort = new AbortController();
    controller = abort;
    const timer = setTimeout(() => abort.abort(
      new SdkError(SdkErrorCode.RequestTimeout, "Connection timed out")), config.connectTimeoutMs);
    const opening = (async (): Promise<Live> => {
      const connection = await connect(config, {
        headers, signal: abort.signal, timeout: config.connectTimeoutMs,
      });
      try {
        abort.signal.throwIfAborted();
        const { tools } = await waitFor(connection.listTools({
          signal: abort.signal, timeout: config.connectTimeoutMs,
        }), abort.signal);
        const missing: string[] = REQUIRED_REFLECTLOG_TOOLS.filter((name) =>
          !tools.some((tool) => tool.name === name));
        const add = tools.find((tool) => tool.name === "add_memories");
        if (!add?.inputSchema.properties || !Object.hasOwn(add.inputSchema.properties, "allow_replace")) {
          missing.push("add_memories.allow_replace");
        }
        if (missing.length) throw new ReflectLogError("incompatible",
          `ReflectLog is missing: ${missing.join(", ")}`, { cause: tools });
        abort.signal.throwIfAborted();
        if (current !== generation) throw new SdkError(SdkErrorCode.ConnectionClosed, "Connection closed");
        live = { generation: current, connection };
        return live;
      } catch (cause) {
        await Promise.allSettled([connection.close()]);
        throw cause;
      }
    })();
    pending = waitFor(opening, abort.signal).finally(() => {
      clearTimeout(timer);
      if (current === generation) { pending = undefined; controller = undefined; }
    });
    return pending;
  }

  async function invoke<T>(request: {
    readonly name: string; readonly args: Record<string, unknown>;
    readonly mutation?: boolean; readonly schema: z.ZodType<T>;
  }, options?: CallOptions): Promise<T> {
    for (let attempt = 0; attempt < 2; attempt++) {
      let sent = false;
      let current = generation;
      let veto: { readonly error: unknown } | undefined;
      try {
        options?.signal?.throwIfAborted();
        const acquiring = acquire();
        current = generation;
        const acquired = await waitFor(acquiring, options?.signal);
        current = acquired.generation;
        options?.signal?.throwIfAborted();
        try {
          options?.beforeDispatch?.();
        } catch (error) {
          veto = { error };
          throw error;
        }
        // SDK has no delivery acknowledgement: invocation is conservatively possibly sent.
        sent = true;
        const response = await waitFor(acquired.connection.callTool({
          name: request.name, arguments: request.args,
        }, { timeout: config.requestTimeoutMs,
          ...(options?.signal ? { signal: options.signal } : {}),
        }), options?.signal);
        return parseToolResponse(response, request.schema);
      } catch (cause) {
        if (veto) throw veto.error;
        // Once a mutation may have been sent, every failure leaves its effect unknown.
        const uncertain = request.mutation === true && sent;
        const raise: (code: ReflectLogErrorCode, message: string) => never = (code, message) => {
          throw new ReflectLogError(uncertain ? "outcome_unknown" : code, message, { cause });
        };
        if (cause instanceof ReflectLogError) {
          if (!uncertain) throw cause;
          raise("outcome_unknown", "ReflectLog did not confirm the write");
        }
        if (options?.signal?.aborted) raise("aborted", "ReflectLog call aborted");
        if (cause instanceof UnauthorizedError || (cause instanceof SdkHttpError && cause.status === 401)) {
          raise("unauthorized", "ReflectLog authentication failed");
        }
        if (cause instanceof ProtocolError) raise("tool_error", "ReflectLog returned a protocol error");
        if (cause instanceof SdkError && cause.code === SdkErrorCode.InvalidResult) {
          raise("malformed", "ReflectLog returned an invalid result");
        }
        if (sent && cause instanceof SdkError && cause.code === SdkErrorCode.RequestTimeout) {
          raise("timeout", "ReflectLog request timed out");
        }
        await invalidate(current);
        if (request.mutation || attempt === 1) raise("unavailable", "ReflectLog connection unavailable");
      }
    }
    throw new ReflectLogError("unavailable", "ReflectLog connection unavailable");
  }

  return {
    searchMemories: (p, o) => invoke({ name: "search_memories", schema: searchMemoriesSchema,
      args: { workspace_id: p.workspaceId, query: p.query, limit: p.limit } }, o),
    getMemoryById: (p, o) => invoke({ name: "get_memory_by_id", schema: getMemoryByIdSchema,
      args: { workspace_id: p.workspaceId, id: p.id } }, o),
    getAllMemories: (p, o) => invoke({ name: "get_all_memories", schema: getAllMemoriesSchema,
      args: { workspace_id: p.workspaceId, limit: p.limit, offset: p.offset } }, o),
    deleteMemories: (p, o) => invoke({ name: "delete_memories", schema: deleteMemoriesSchema,
      mutation: true, args: { workspace_id: p.workspaceId, ids: p.ids } }, o),
    addMemories: (p, o) => invoke({ name: "add_memories", schema: addMemoriesSchema, mutation: !p.dryRun,
      args: { workspace_id: p.workspaceId, memories: p.memories,
        dry_run: p.dryRun, allow_replace: p.allowReplace } }, o),
    close: () => invalidate(generation),
  };
}
