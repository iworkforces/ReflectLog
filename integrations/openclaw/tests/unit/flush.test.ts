import { describe, expect, it } from "vitest";

import { parseConfig } from "../../src/config.js";
import { FlushCoordinator, type SharedFlushRun } from "../../src/flush-coordinator.js";
import { FlushCapacityError, FlushDisposedError } from "../../src/flush-ledger.js";
import { resolveFlushPlan } from "../../src/flush.js";

describe("pre-compaction flush plan", () => {
  it("is absent without a valid configuration", () => {
    expect(resolveFlushPlan({ config: null })).toBeNull();
  });

  it("persists through memory_store only, with disjoint lookup tools", () => {
    const parsed = parseConfig({});
    if (!parsed.ok) throw new Error(parsed.error);
    const plan = resolveFlushPlan({ config: parsed.config, nowMs: 0 });
    expect(plan).toMatchObject({
      persistenceToolNames: ["memory_store"],
      lookupToolNames: ["memory_search", "memory_get"],
      prompt: expect.stringContaining("NO_REPLY"),
      systemPrompt: expect.stringContaining("never memory files"),
    });
  });
});

function waiter() {
  const controller = new AbortController();
  const state = { revoked: false, checks: 0 };
  const guard = {
    signal: controller.signal,
    assertCurrent: () => {
      state.checks += 1;
      controller.signal.throwIfAborted();
      if (state.revoked) throw new Error("revoked");
    },
  };
  return { guard, state, abort: (reason?: unknown) => { controller.abort(reason); } };
}

/** An operation that reaches dispatch only when the test says the "connection" is up. */
function operation(value = "result") {
  const connection = Promise.withResolvers<void>();
  const response = Promise.withResolvers<string>();
  const runs: SharedFlushRun[] = [];
  let dispatches = 0;
  const start = async (shared: SharedFlushRun): Promise<string> => {
    runs.push(shared);
    await connection.promise;
    shared.beforeDispatch();
    dispatches += 1;
    return response.promise;
  };
  return {
    start, runs, get dispatches() { return dispatches; },
    connect: () => connection.resolve(),
    respond: () => response.resolve(value),
    fail: (error: unknown) => response.reject(error),
  };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("flush coordinator", () => {
  it("checks a caller's authority before it joins, without touching the ledger", async () => {
    const coordinator = new FlushCoordinator<string>();
    const stale = waiter();
    stale.state.revoked = true;
    const op = operation();
    await expect(coordinator.run("k", stale.guard, op.start)).rejects.toThrow("revoked");
    expect(op.runs).toEqual([]);
    expect(coordinator.size).toBe(0);
  });

  it("runs the shared write on a ledger-owned signal, never on a waiter's", async () => {
    const coordinator = new FlushCoordinator<string>();
    const first = waiter();
    const second = waiter();
    const op = operation();
    const one = coordinator.run("k", first.guard, op.start);
    const two = coordinator.run("k", second.guard, operation().start);
    expect(op.runs).toHaveLength(1);
    expect(op.runs[0]?.signal).not.toBe(first.guard.signal);
    first.abort(new Error("first left"));
    await expect(one).rejects.toThrow("first left");
    expect(op.runs[0]?.signal.aborted).toBe(false);
    op.connect();
    op.respond();
    await expect(two).resolves.toEqual({ value: "result", deduplicated: true });
    expect(op.dispatches).toBe(1);
  });

  it("vetoes dispatch when no waiter is current, frees the key and leaves the next caller a clean dispatch", async () => {
    const coordinator = new FlushCoordinator<string>();
    const only = waiter();
    const op = operation();
    const first = coordinator.run("k", only.guard, op.start);
    only.state.revoked = true;
    op.connect();
    await expect(first).rejects.toThrow("revoked");
    expect(op.dispatches).toBe(0);
    expect(coordinator.size).toBe(0);
    const retry = operation("retried");
    const second = coordinator.run("k", waiter().guard, retry.start);
    retry.connect();
    retry.respond();
    await expect(second).resolves.toEqual({ value: "retried", deduplicated: false });
  });

  it("dispatches while any one waiter is current and hands the revoked one its access error", async () => {
    const coordinator = new FlushCoordinator<string>();
    const revoked = waiter();
    const current = waiter();
    const op = operation();
    const one = coordinator.run("k", revoked.guard, op.start);
    const two = coordinator.run("k", current.guard, operation().start);
    revoked.state.revoked = true;
    op.connect();
    op.respond();
    await expect(one).rejects.toThrow("revoked");
    await expect(two).resolves.toMatchObject({ value: "result" });
    expect(op.dispatches).toBe(1);
  });

  it("gives a revoked waiter the access error, not the failure, when the write fails", async () => {
    const coordinator = new FlushCoordinator<string>();
    const revoked = waiter();
    const current = waiter();
    const op = operation();
    const one = coordinator.run("k", revoked.guard, op.start);
    const two = coordinator.run("k", current.guard, operation().start);
    op.connect();
    await tick();
    revoked.state.revoked = true;
    op.fail(new Error("socket reset"));
    await expect(one).rejects.toThrow("revoked");
    await expect(two).rejects.toMatchObject({ code: "outcome_unknown" });
  });

  it("finishes and caches the write after every waiter left post-dispatch", async () => {
    const coordinator = new FlushCoordinator<string>();
    const only = waiter();
    const op = operation();
    const first = coordinator.run("k", only.guard, op.start);
    op.connect();
    await tick();
    only.abort(new Error("left"));
    await expect(first).rejects.toThrow("left");
    op.respond();
    await tick();
    expect(coordinator.counts).toEqual({ pending: 0, success: 1, indeterminate: 0 });
    await expect(coordinator.run("k", waiter().guard, operation().start)).resolves.toEqual({ value: "result", deduplicated: true });
  });

  it("retains no caller guards in a cached outcome", async () => {
    const coordinator = new FlushCoordinator<string>();
    const first = waiter();
    const op = operation();
    const pending = coordinator.run("k", first.guard, op.start);
    op.connect();
    op.respond();
    await pending;
    const checksAfterRelease = first.state.checks;
    first.state.revoked = true;
    await expect(coordinator.run("k", waiter().guard, operation().start)).resolves.toMatchObject({ deduplicated: true });
    await expect(coordinator.run("k", waiter().guard, operation().start)).resolves.toMatchObject({ deduplicated: true });
    expect(first.state.checks).toBe(checksAfterRelease);
  });

  it("dispatches once when the same key re-enters synchronously from the start callback", async () => {
    const coordinator = new FlushCoordinator<string>();
    let inner: Promise<unknown> | undefined;
    let runs = 0;
    const outer = await coordinator.run("k", waiter().guard, async () => {
      runs += 1;
      inner = coordinator.run("k", waiter().guard, async () => {
        runs += 1;
        return "second";
      });
      return "first";
    });
    expect(await inner).toEqual({ value: "first", deduplicated: true });
    expect(outer.deduplicated).toBe(false);
    expect(runs).toBe(1);
  });

  it("rejects distinct work with the capacity error while one write is pending", async () => {
    const coordinator = new FlushCoordinator<string>({ maxEntries: 1 });
    const op = operation();
    const first = coordinator.run("a", waiter().guard, op.start);
    await expect(coordinator.run("b", waiter().guard, operation().start)).rejects.toBeInstanceOf(FlushCapacityError);
    op.connect();
    op.respond();
    await first;
  });

  it("settles waiters and rejects new work once disposed", async () => {
    const coordinator = new FlushCoordinator<string>();
    const op = operation();
    const first = coordinator.run("k", waiter().guard, op.start);
    op.connect();
    await tick();
    coordinator.dispose();
    await expect(first).rejects.toMatchObject({ code: "outcome_unknown" });
    await expect(coordinator.run("k", waiter().guard, operation().start)).rejects.toBeInstanceOf(FlushDisposedError);
    op.respond();
    await tick();
    expect(coordinator.size).toBe(0);
  });

  it("keeps two coordinators, one per backend, fully separate", async () => {
    const one = new FlushCoordinator<string>();
    const two = new FlushCoordinator<string>();
    const first = operation("one");
    const second = operation("two");
    const a = one.run("k", waiter().guard, first.start);
    const b = two.run("k", waiter().guard, second.start);
    first.connect(); first.respond(); second.connect(); second.respond();
    expect(await a).toEqual({ value: "one", deduplicated: false });
    expect(await b).toEqual({ value: "two", deduplicated: false });
  });
});
