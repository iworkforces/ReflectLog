import { ProtocolError, SdkError, SdkErrorCode, SdkHttpError, UnauthorizedError } from "@modelcontextprotocol/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { parseConfig, type ReflectLogPluginConfig } from "../../src/config.js";
import { createReflectLogClient, type ConnectFn, type McpConnection } from "../../src/reflectlog/mcp-client.js";
import type { ToolResponse } from "../../src/reflectlog/parse.js";
import { REQUIRED_REFLECTLOG_TOOLS, type ReflectLogClient } from "../../src/reflectlog/types.js";

type Tool = Awaited<ReturnType<McpConnection["listTools"]>>["tools"][number];
type Respond = (name: string) => Promise<ToolResponse>;

const TOOLS: readonly Tool[] = REQUIRED_REFLECTLOG_TOOLS.map((name) => ({
  name,
  inputSchema: name === "add_memories" ? { properties: { allow_replace: { type: "boolean" } } } : {},
}));

const RESULTS: Readonly<Record<string, unknown>> = {
  search_memories: { results: [], score_kind: "none", limit: 5 },
  get_memory_by_id: { record: null },
  get_all_memories: { records: [], total: 0, offset: 0, limit: 0, truncated: false },
  delete_memories: { removed: [1], missing: [] },
  add_memories: { stored_count: 1, skipped_count: 0, replaced_count: 0, dry_run: false, replacements: [] },
};

const ok: Respond = async (name) => ({ structuredContent: RESULTS[name] });
const fail = (error: Error): Respond => async () => {
  throw error;
};
const hang: Respond = () => new Promise<ToolResponse>(() => {});

class FakeConnection implements McpConnection {
  readonly calls: { name: string; arguments: Record<string, unknown> }[] = [];
  closed = 0;
  readonly #respond: Respond;
  readonly #tools: readonly Tool[];

  constructor(respond: Respond = ok, tools: readonly Tool[] = TOOLS) {
    this.#respond = respond;
    this.#tools = tools;
  }

  async listTools(): Promise<{ readonly tools: readonly Tool[] }> {
    return { tools: this.#tools };
  }

  callTool(params: { readonly name: string; readonly arguments: Record<string, unknown> }): Promise<ToolResponse> {
    this.calls.push({ name: params.name, arguments: params.arguments });
    return this.#respond(params.name);
  }

  async close(): Promise<void> {
    this.closed += 1;
  }
}

function config(raw: unknown = {}): ReflectLogPluginConfig {
  const parsed = parseConfig(raw);
  if (!parsed.ok) {
    throw new Error(parsed.error);
  }
  return parsed.config;
}

type Opening = McpConnection | Error | Promise<McpConnection>;

function setup(
  openings: readonly Opening[],
  raw: unknown = {},
  env: Readonly<Record<string, string | undefined>> = { REFLECTLOG_MCP_TOKEN: "token" },
) {
  const queue = [...openings];
  const connect = vi.fn<ConnectFn>(async () => {
    const next = queue.shift();
    if (next === undefined) {
      throw new Error("unexpected connect");
    }
    if (next instanceof Error) {
      throw next;
    }
    return next;
  });
  return { connect, client: createReflectLogClient(config(raw), { connect, env }) };
}

const health = (client: ReflectLogClient, signal?: AbortSignal) =>
  client.getAllMemories({ workspaceId: "w", limit: 0, offset: 0 }, signal ? { signal } : undefined);
const store = (client: ReflectLogClient, dryRun = false, signal?: AbortSignal) =>
  client.addMemories(
    { workspaceId: "w", memories: ["Prefers tea"], dryRun, allowReplace: false },
    signal ? { signal } : undefined,
  );

describe("ReflectLog MCP client", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("maps typed calls onto ReflectLog tool arguments", async () => {
    const connection = new FakeConnection();
    const { client } = setup([connection]);
    await client.searchMemories({ workspaceId: "w", query: "tea", limit: 3 });
    await client.getMemoryById({ workspaceId: "w", id: 4 });
    await client.getAllMemories({ workspaceId: "w", limit: 10, offset: 20 });
    await store(client);
    await client.deleteMemories({ workspaceId: "w", ids: [1] });
    expect(connection.calls).toEqual([
      { name: "search_memories", arguments: { workspace_id: "w", query: "tea", limit: 3 } },
      { name: "get_memory_by_id", arguments: { workspace_id: "w", id: 4 } },
      { name: "get_all_memories", arguments: { workspace_id: "w", limit: 10, offset: 20 } },
      {
        name: "add_memories",
        arguments: { workspace_id: "w", memories: ["Prefers tea"], dry_run: false, allow_replace: false },
      },
      { name: "delete_memories", arguments: { workspace_id: "w", ids: [1] } },
    ]);
  });

  it("authenticates HTTP with the token from the configured environment variable", async () => {
    const { client, connect } = setup([new FakeConnection()]);
    await health(client);
    expect(connect.mock.calls[0]?.[1].headers).toEqual({ Authorization: "Bearer token" });
  });

  it("refuses HTTP without a token before connecting", async () => {
    const { client, connect } = setup([new FakeConnection()], {}, { REFLECTLOG_MCP_TOKEN: " " });
    await expect(health(client)).rejects.toMatchObject({ code: "unauthorized" });
    expect(connect).not.toHaveBeenCalled();
  });

  it("accepts only the HTTP transport", () => {
    expect(parseConfig({ transport: "http" }).ok).toBe(true);
    expect(parseConfig({ transport: "stdio" }).ok).toBe(false);
    expect(parseConfig({ stdio: { command: "reflectlog" } }).ok).toBe(false);
  });

  it("shares one connection between concurrent callers", async () => {
    const connection = new FakeConnection();
    const { client, connect } = setup([connection]);
    await Promise.all([health(client), health(client)]);
    expect(connect).toHaveBeenCalledTimes(1);
    expect(connection.calls).toHaveLength(2);
  });

  it.each([
    ...REQUIRED_REFLECTLOG_TOOLS.map((name) => [
      `a missing ${name} tool`, TOOLS.filter((tool) => tool.name !== name), name,
    ] as const),
    [
      "an add_memories tool without allow_replace",
      TOOLS.map((tool) => (tool.name === "add_memories" ? { name: "add_memories", inputSchema: {} } : tool)),
      "add_memories.allow_replace",
    ] as const,
  ])("rejects a server with %s and closes the connection", async (_label, tools, missing) => {
    const connection = new FakeConnection(ok, tools);
    const { client } = setup([connection]);
    await expect(health(client)).rejects.toMatchObject({
      code: "incompatible",
      message: expect.stringContaining(missing),
    });
    expect(connection.closed).toBe(1);
    expect(connection.calls).toEqual([]);
  });

  it("reconnects and retries a read once after the connection drops", async () => {
    const dropped = new FakeConnection(fail(new SdkError(SdkErrorCode.ConnectionClosed, "closed")));
    const fresh = new FakeConnection();
    const { client, connect } = setup([dropped, fresh]);
    await expect(client.getAllMemories({ workspaceId: "w", limit: 10, offset: 0 })).resolves.toMatchObject({ total: 0 });
    expect(connect).toHaveBeenCalledTimes(2);
    expect(dropped.closed).toBe(1);
    expect(fresh.calls).toHaveLength(1);
  });

  it("gives up on a read after one retry", async () => {
    const broken = () => new FakeConnection(fail(new SdkError(SdkErrorCode.SendFailed, "send failed")));
    const { client, connect } = setup([broken(), broken()]);
    await expect(health(client)).rejects.toMatchObject({ code: "unavailable" });
    expect(connect).toHaveBeenCalledTimes(2);
  });

  it.each(["addMemories", "deleteMemories"] as const)("never replays %s after it may have reached the server", async (method) => {
    const dropped = new FakeConnection(fail(new SdkError(SdkErrorCode.ConnectionClosed, "closed")));
    const { client, connect } = setup([dropped, new FakeConnection()]);
    const pending = method === "addMemories" ? store(client) : client.deleteMemories({ workspaceId: "w", ids: [1] });
    await expect(pending).rejects.toMatchObject({ code: "outcome_unknown" });
    expect(connect).toHaveBeenCalledTimes(1);
    expect(dropped.calls).toHaveLength(1);
  });

  it("retries a dry-run add like a read", async () => {
    const dropped = new FakeConnection(fail(new SdkError(SdkErrorCode.ConnectionClosed, "closed")));
    const fresh = new FakeConnection();
    const { client } = setup([dropped, fresh]);
    await expect(store(client, true)).resolves.toMatchObject({ storedCount: 1 });
    expect(fresh.calls).toHaveLength(1);
  });

  it("reports a mutation that could not be sent as unavailable without retrying", async () => {
    const { client, connect } = setup([new Error("ECONNREFUSED"), new FakeConnection()]);
    await expect(client.deleteMemories({ workspaceId: "w", ids: [1] })).rejects.toMatchObject({
      code: "unavailable",
    });
    expect(connect).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["read", (client: ReflectLogClient) => health(client), "timeout"],
    ["mutation", (client: ReflectLogClient) => store(client), "outcome_unknown"],
  ] as const)("does not retry a %s that timed out after sending", async (_label, call, code) => {
    const slow = new FakeConnection(fail(new SdkError(SdkErrorCode.RequestTimeout, "timed out")));
    const { client, connect } = setup([slow, new FakeConnection()]);
    await expect(call(client)).rejects.toMatchObject({ code });
    expect(connect).toHaveBeenCalledTimes(1);
    expect(slow.calls).toHaveLength(1);
  });

  it.each([
    ["a tool error", async () => ({ isError: true, content: [{ type: "text", text: "StorageError: busy" }] }), "tool_error"],
    ["a bearer rejection", async () => ({ isError: true, content: [{ type: "text", text: "Unauthorized" }] }), "unauthorized"],
    ["an HTTP 401", fail(new SdkHttpError(SdkErrorCode.SendFailed, "Unauthorized", { status: 401 })), "unauthorized"],
    ["an OAuth rejection", fail(new UnauthorizedError("denied")), "unauthorized"],
    ["a JSON-RPC error", fail(new ProtocolError(-32602, "invalid params")), "tool_error"],
    ["an invalid result", fail(new SdkError(SdkErrorCode.InvalidResult, "invalid")), "malformed"],
    ["an unexpected shape", async () => ({ structuredContent: { status: "sideways" } }), "malformed"],
  ] as const)("classifies %s without retrying", async (_label, respond, code) => {
    const connection = new FakeConnection(respond);
    const { client, connect } = setup([connection, new FakeConnection()]);
    await expect(health(client)).rejects.toMatchObject({ code });
    expect(connect).toHaveBeenCalledTimes(1);
    expect(connection.calls).toHaveLength(1);
  });

  it("aborts before connecting when the caller signal is already aborted", async () => {
    const { client, connect } = setup([new FakeConnection()]);
    await expect(health(client, AbortSignal.abort())).rejects.toMatchObject({ code: "aborted" });
    expect(connect).not.toHaveBeenCalled();
  });

  it("reports an aborted in-flight mutation as outcome unknown", async () => {
    const stuck = new FakeConnection(hang);
    const { client } = setup([stuck]);
    const controller = new AbortController();
    const pending = store(client, false, controller.signal);
    await vi.waitFor(() => expect(stuck.calls).toHaveLength(1));
    controller.abort();
    await expect(pending).rejects.toMatchObject({ code: "outcome_unknown" });
  });

  it("lets one caller abort without failing others waiting on the same connection", async () => {
    const opening = Promise.withResolvers<McpConnection>();
    const { client, connect } = setup([opening.promise]);
    const controller = new AbortController();
    const aborted = health(client, controller.signal);
    const waiting = health(client);
    controller.abort();
    await expect(aborted).rejects.toMatchObject({ code: "aborted" });
    opening.resolve(new FakeConnection());
    await expect(waiting).resolves.toMatchObject({ records: [], total: 0, limit: 0 });
    expect(connect).toHaveBeenCalledTimes(1);
  });

  it("discards a connection that finishes opening after close", async () => {
    const opening = Promise.withResolvers<McpConnection>();
    const stale = new FakeConnection();
    const fresh = new FakeConnection();
    const { client, connect } = setup([opening.promise, fresh]);
    const pending = health(client);
    await vi.waitFor(() => expect(connect).toHaveBeenCalledTimes(1));
    await client.close();
    opening.resolve(stale);
    await expect(pending).resolves.toMatchObject({ records: [], total: 0, limit: 0 });
    await vi.waitFor(() => expect(stale.closed).toBe(1));
    expect(stale.calls).toEqual([]);
    expect(fresh.calls).toHaveLength(1);
  });

  it("closes the live connection and reconnects lazily", async () => {
    const first = new FakeConnection();
    const second = new FakeConnection();
    const { client, connect } = setup([first, second]);
    await health(client);
    await client.close();
    expect(first.closed).toBe(1);
    await health(client);
    expect(connect).toHaveBeenCalledTimes(2);
    expect(second.calls).toHaveLength(1);
  });

  it("times out a hung connect and reports the backend unavailable", async () => {
    vi.useFakeTimers();
    const never = () => new Promise<McpConnection>(() => {});
    const { client, connect } = setup([never(), never()], { connectTimeoutMs: 1000 });
    const outcome = health(client).catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(1000);
    await vi.advanceTimersByTimeAsync(1000);
    expect(await outcome).toMatchObject({ code: "unavailable" });
    expect(connect).toHaveBeenCalledTimes(2);
  });
});
