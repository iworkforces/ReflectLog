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
  inputSchema: name === "add" ? { properties: { allow_replace: { type: "boolean" } } } : {},
}));

const RESULTS: Readonly<Record<string, unknown>> = {
  search_records: { results: [], score_kind: "none", limit: 5 },
  get_records: { records: [], missing: [] },
  list_records: { records: [], total: 0, offset: 0, limit: 10, truncated: false },
  remove_records: { removed: [1], missing: [] },
  add: { stored_count: 1, skipped_count: 0, replaced_count: 0, dry_run: false, replacements: [] },
  health_check: { status: "healthy" },
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
  client.healthCheck({ workspaceId: "w" }, signal ? { signal } : undefined);
const store = (client: ReflectLogClient, dryRun = false, signal?: AbortSignal) =>
  client.add(
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
    await client.searchRecords({ workspaceId: "w", query: "tea", limit: 3 });
    await client.getRecords({ workspaceId: "w", ids: [4] });
    await client.listRecords({ workspaceId: "w", limit: 10, offset: 20 });
    await store(client);
    await client.removeRecords({ workspaceId: "w", ids: [1] });
    expect(connection.calls).toEqual([
      { name: "search_records", arguments: { workspace_id: "w", query: "tea", limit: 3 } },
      { name: "get_records", arguments: { workspace_id: "w", ids: [4] } },
      { name: "list_records", arguments: { workspace_id: "w", limit: 10, offset: 20 } },
      {
        name: "add",
        arguments: { workspace_id: "w", memories: ["Prefers tea"], dry_run: false, allow_replace: false },
      },
      { name: "remove_records", arguments: { workspace_id: "w", ids: [1] } },
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

  it("launches stdio with the inherited and configured environment and no token", async () => {
    const { client, connect } = setup(
      [new FakeConnection()],
      { transport: "stdio", stdio: { env: { EXTRA: "1" } } },
      { PATH: "/bin", UNSET: undefined },
    );
    await health(client);
    expect(connect.mock.calls[0]?.[1]).toMatchObject({ headers: {}, env: { PATH: "/bin", EXTRA: "1" } });
    expect(connect.mock.calls[0]?.[1].env).not.toHaveProperty("UNSET");
  });

  it("shares one connection between concurrent callers", async () => {
    const connection = new FakeConnection();
    const { client, connect } = setup([connection]);
    await Promise.all([health(client), health(client)]);
    expect(connect).toHaveBeenCalledTimes(1);
    expect(connection.calls).toHaveLength(2);
  });

  it.each([
    ["a missing tool", TOOLS.filter((tool) => tool.name !== "remove_records"), "remove_records"],
    [
      "an add tool without allow_replace",
      TOOLS.map((tool) => (tool.name === "add" ? { name: "add", inputSchema: {} } : tool)),
      "add.allow_replace",
    ],
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
    await expect(client.listRecords({ workspaceId: "w", limit: 10, offset: 0 })).resolves.toMatchObject({ total: 0 });
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

  it("never replays a mutation that may have reached the server", async () => {
    const dropped = new FakeConnection(fail(new SdkError(SdkErrorCode.ConnectionClosed, "closed")));
    const { client, connect } = setup([dropped, new FakeConnection()]);
    await expect(store(client)).rejects.toMatchObject({ code: "outcome_unknown" });
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
    await expect(client.removeRecords({ workspaceId: "w", ids: [1] })).rejects.toMatchObject({
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
    await expect(waiting).resolves.toMatchObject({ status: "healthy" });
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
    await expect(pending).resolves.toMatchObject({ status: "healthy" });
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
