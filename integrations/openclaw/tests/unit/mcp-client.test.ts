import { ProtocolError, SdkError, SdkErrorCode, SdkHttpError, UnauthorizedError } from "@modelcontextprotocol/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { parseConfig, type ReflectLogPluginConfig } from "../../src/config.js";
import { createReflectLogClient, type ConnectFn, type McpConnection } from "../../src/reflectlog/mcp-client.js";
import type { ToolResponse } from "../../src/reflectlog/parse.js";
import { REQUIRED_REFLECTLOG_TOOLS, ReflectLogError, type ReflectLogClient } from "../../src/reflectlog/types.js";

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
  env: Readonly<Record<string, string | undefined>> = { REFLECTLOG_MCP_AUTH_TOKEN: "token" },
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
    const { client, connect } = setup([new FakeConnection()], {}, { REFLECTLOG_MCP_AUTH_TOKEN: " " });
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

  describe("mutation dispatch classification", () => {
    const mutations = [
      ["addMemories", (client: ReflectLogClient, signal?: AbortSignal) => store(client, false, signal)],
      ["deleteMemories", (client: ReflectLogClient, signal?: AbortSignal) =>
        client.deleteMemories({ workspaceId: "w", ids: [1] }, signal ? { signal } : undefined)],
    ] as const;
    const postSend = [
      ["a tool error", async () => ({ isError: true, content: [{ type: "text", text: "StorageError: busy" }] }), "tool_error"],
      ["a bearer rejection", async () => ({ isError: true, content: [{ type: "text", text: "Unauthorized" }] }), "unauthorized"],
      ["a malformed result", async () => ({ structuredContent: { status: "sideways" } }), "malformed"],
      ["an HTTP 401", fail(new SdkHttpError(SdkErrorCode.SendFailed, "Unauthorized", { status: 401 })), undefined],
      ["an OAuth rejection", fail(new UnauthorizedError("denied")), undefined],
      ["a JSON-RPC error", fail(new ProtocolError(-32602, "invalid params")), undefined],
      ["an invalid result", fail(new SdkError(SdkErrorCode.InvalidResult, "invalid")), undefined],
      ["a request timeout", fail(new SdkError(SdkErrorCode.RequestTimeout, "timed out")), undefined],
      ["a closed connection", fail(new SdkError(SdkErrorCode.ConnectionClosed, "closed")), undefined],
      ["a generic failure", fail(new Error("boom")), undefined],
    ] as const;

    describe.each(mutations)("%s", (_method, call) => {
      it.each(postSend)("is outcome_unknown with its cause after %s was received post-send", async (_label, respond, originalCode) => {
        const connection = new FakeConnection(respond);
        const { client, connect } = setup([connection, new FakeConnection()]);
        const error = await call(client).catch((caught: unknown) => caught);
        expect(error).toBeInstanceOf(ReflectLogError);
        expect(error).toMatchObject({ code: "outcome_unknown", cause: expect.anything() });
        expect(String((error as Error).message)).not.toMatch(/busy|token|Prefers tea/);
        if (originalCode !== undefined) {
          expect((error as Error).cause).toMatchObject({ code: originalCode });
        }
        expect(connect).toHaveBeenCalledTimes(1);
        expect(connection.calls).toHaveLength(1);
      });

      it("is outcome_unknown when the caller aborts after dispatch", async () => {
        const stuck = new FakeConnection(hang);
        const { client } = setup([stuck]);
        const controller = new AbortController();
        const pending = call(client, controller.signal);
        await vi.waitFor(() => expect(stuck.calls).toHaveLength(1));
        controller.abort();
        await expect(pending).rejects.toMatchObject({ code: "outcome_unknown", cause: expect.anything() });
      });

      it("stays unavailable, not outcome_unknown, when the connection fails before dispatch", async () => {
        const { client, connect } = setup([new Error("ECONNREFUSED"), new FakeConnection()]);
        await expect(call(client)).rejects.toMatchObject({ code: "unavailable" });
        expect(connect).toHaveBeenCalledTimes(1);
      });

      it("stays unauthorized, not outcome_unknown, when no token exists before dispatch", async () => {
        const { client, connect } = setup([new FakeConnection()], {}, { REFLECTLOG_MCP_AUTH_TOKEN: "" });
        await expect(call(client)).rejects.toMatchObject({ code: "unauthorized" });
        expect(connect).not.toHaveBeenCalled();
      });

      it("stays incompatible, not outcome_unknown, when the handshake rejects the server before dispatch", async () => {
        const connection = new FakeConnection(ok, TOOLS.filter((tool) => tool.name !== "search_memories"));
        const { client } = setup([connection]);
        await expect(call(client)).rejects.toMatchObject({ code: "incompatible" });
        expect(connection.calls).toEqual([]);
      });

      it("stays aborted when the caller signal was already aborted before dispatch", async () => {
        const { client, connect } = setup([new FakeConnection()]);
        await expect(call(client, AbortSignal.abort())).rejects.toMatchObject({ code: "aborted" });
        expect(connect).not.toHaveBeenCalled();
      });
    });

    it("keeps dry-run add and read classification after the request was sent", async () => {
      const toolError = async () => ({ isError: true, content: [{ type: "text", text: "StorageError: busy" }] });
      const dryRun = new FakeConnection(toolError);
      const { client } = setup([dryRun, new FakeConnection()]);
      await expect(store(client, true)).rejects.toMatchObject({ code: "tool_error" });
      const timedOut = new FakeConnection(fail(new SdkError(SdkErrorCode.RequestTimeout, "timed out")));
      const second = setup([timedOut, new FakeConnection()]);
      await expect(store(second.client, true)).rejects.toMatchObject({ code: "timeout" });
      const controller = new AbortController();
      const stuck = new FakeConnection(hang);
      const third = setup([stuck]);
      const pending = store(third.client, true, controller.signal);
      await vi.waitFor(() => expect(stuck.calls).toHaveLength(1));
      controller.abort();
      await expect(pending).rejects.toMatchObject({ code: "aborted" });
    });
  });

  describe("beforeDispatch", () => {
    it("runs after the connection is acquired and immediately before the request is handed over", async () => {
      const order: string[] = [];
      const connection = new FakeConnection(async (name) => {
        order.push(`call:${name}`);
        return { structuredContent: RESULTS[name] };
      });
      const { client, connect } = setup([connection]);
      connect.mockImplementationOnce(async () => {
        order.push("connect");
        return connection;
      });
      await client.addMemories({ workspaceId: "w", memories: ["x"], dryRun: false, allowReplace: false },
        { beforeDispatch: () => { order.push("hook"); } });
      expect(order).toEqual(["connect", "hook", "call:add_memories"]);
    });

    it("propagates what it throws unchanged and never invokes the transport", async () => {
      const veto = new Error("all waiters revoked");
      const connection = new FakeConnection();
      const { client, connect } = setup([connection, new FakeConnection()]);
      const attempt = client.addMemories({ workspaceId: "w", memories: ["x"], dryRun: false, allowReplace: false },
        { beforeDispatch: () => { throw veto; } });
      await expect(attempt).rejects.toBe(veto);
      expect(connection.calls).toEqual([]);
      expect(connect).toHaveBeenCalledTimes(1);
      expect(connection.closed).toBe(0);
    });

    it("propagates a typed error thrown by a read hook without retrying", async () => {
      const veto = new ReflectLogError("aborted", "revoked");
      const connection = new FakeConnection();
      const { client } = setup([connection, new FakeConnection()]);
      await expect(client.getAllMemories({ workspaceId: "w", limit: 0, offset: 0 },
        { beforeDispatch: () => { throw veto; } })).rejects.toBe(veto);
      expect(connection.calls).toEqual([]);
    });

    it("runs exactly once for a mutation, including when it fails after dispatch", async () => {
      const hook = vi.fn();
      const { client, connect } = setup([new FakeConnection(fail(new SdkError(SdkErrorCode.ConnectionClosed, "closed"))), new FakeConnection()]);
      await expect(client.addMemories({ workspaceId: "w", memories: ["x"], dryRun: false, allowReplace: false },
        { beforeDispatch: hook })).rejects.toMatchObject({ code: "outcome_unknown" });
      expect(hook).toHaveBeenCalledTimes(1);
      expect(connect).toHaveBeenCalledTimes(1);
    });

    it("runs once per attempt when a read is retried", async () => {
      const hook = vi.fn();
      const dropped = new FakeConnection(fail(new SdkError(SdkErrorCode.ConnectionClosed, "closed")));
      const fresh = new FakeConnection();
      const { client } = setup([dropped, fresh]);
      await client.getAllMemories({ workspaceId: "w", limit: 0, offset: 0 }, { beforeDispatch: hook });
      expect(hook).toHaveBeenCalledTimes(2);
      expect(dropped.calls).toHaveLength(1);
      expect(fresh.calls).toHaveLength(1);
    });

    it("does not run when connecting fails or the signal is already aborted", async () => {
      const hook = vi.fn();
      const refused = setup([new Error("ECONNREFUSED")]);
      await expect(refused.client.addMemories({ workspaceId: "w", memories: ["x"], dryRun: false, allowReplace: false },
        { beforeDispatch: hook })).rejects.toMatchObject({ code: "unavailable" });
      const aborted = setup([new FakeConnection()]);
      await expect(aborted.client.addMemories({ workspaceId: "w", memories: ["x"], dryRun: false, allowReplace: false },
        { signal: AbortSignal.abort(), beforeDispatch: hook })).rejects.toMatchObject({ code: "aborted" });
      expect(hook).not.toHaveBeenCalled();
    });
  });
});
