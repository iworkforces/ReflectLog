import { describe, expect, it, vi } from "vitest";

import { FakeReflectLogClient } from "../support/fake-client.js";

const add = (client: FakeReflectLogClient, memories: readonly string[], options?: Parameters<FakeReflectLogClient["addMemories"]>[1]) =>
  client.addMemories({ workspaceId: "w", memories, dryRun: false, allowReplace: false }, options);

describe("fake client dispatch controls", () => {
  it("keeps failNext as a post-dispatch failure: the hook runs, the call is recorded, nothing is applied", async () => {
    const client = new FakeReflectLogClient();
    const hook = vi.fn();
    const error = new Error("lost");
    client.failNext("addMemories", error);
    await expect(add(client, ["a"], { beforeDispatch: hook })).rejects.toBe(error);
    expect(hook).toHaveBeenCalledTimes(1);
    expect(client.callsTo("addMemories")).toHaveLength(1);
    expect(client.records("w")).toEqual([]);
    await expect(add(client, ["a"])).resolves.toMatchObject({ storedCount: 1 });
  });

  it("failBeforeDispatch rejects before the hook and before recording, once", async () => {
    const client = new FakeReflectLogClient();
    const hook = vi.fn();
    const error = new Error("refused");
    client.failBeforeDispatch("addMemories", error);
    await expect(add(client, ["a"], { beforeDispatch: hook })).rejects.toBe(error);
    expect(hook).not.toHaveBeenCalled();
    expect(client.calls).toEqual([]);
    await expect(add(client, ["a"], { beforeDispatch: hook })).resolves.toMatchObject({ storedCount: 1 });
    expect(hook).toHaveBeenCalledTimes(1);
  });

  it("an aborted signal wins over every injected failure and records nothing", async () => {
    const client = new FakeReflectLogClient();
    const hook = vi.fn();
    client.failNext("addMemories", new Error("late"));
    await expect(add(client, ["a"], { signal: AbortSignal.abort(new Error("stop")), beforeDispatch: hook }))
      .rejects.toThrow("stop");
    expect(hook).not.toHaveBeenCalled();
    expect(client.calls).toEqual([]);
  });

  it("a throwing beforeDispatch vetoes the call without recording or applying it", async () => {
    const client = new FakeReflectLogClient();
    const veto = new Error("veto");
    await expect(add(client, ["a"], { beforeDispatch: () => { throw veto; } })).rejects.toBe(veto);
    expect(client.calls).toEqual([]);
    expect(client.records("w")).toEqual([]);
  });

  it("reports the call as dispatched only once the hook has returned", async () => {
    const client = new FakeReflectLogClient();
    const deferred = client.deferNext("addMemories");
    const order: string[] = [];
    void deferred.dispatched.then(() => order.push("dispatched"));
    const pending = add(client, ["a"], { beforeDispatch: () => { order.push("hook"); } });
    await deferred.dispatched;
    expect(order).toEqual(["hook", "dispatched"]);
    expect(client.callsTo("addMemories")).toHaveLength(1);
    expect(client.records("w")).toEqual([]);
    deferred.resolve();
    await expect(pending).resolves.toMatchObject({ storedCount: 1 });
    expect(client.records("w")).toHaveLength(1);
  });

  it("holds a deferred call until resolved and applies it exactly once", async () => {
    const client = new FakeReflectLogClient();
    const deferred = client.deferNext("addMemories");
    const pending = add(client, ["a"]);
    await deferred.dispatched;
    deferred.resolve();
    deferred.failUnapplied(new Error("ignored: first decision wins"));
    await expect(pending).resolves.toMatchObject({ storedCount: 1, skippedCount: 0 });
    expect(client.records("w")).toHaveLength(1);
  });

  it("failUnapplied rejects without mutating state", async () => {
    const client = new FakeReflectLogClient();
    const deferred = client.deferNext("addMemories");
    const pending = add(client, ["a"]);
    await deferred.dispatched;
    const error = new Error("server never acted");
    deferred.failUnapplied(error);
    await expect(pending).rejects.toBe(error);
    expect(client.records("w")).toEqual([]);
  });

  it("applyThenFail mutates state and still rejects (response loss)", async () => {
    const client = new FakeReflectLogClient();
    const deferred = client.deferNext("addMemories");
    const pending = add(client, ["a"]);
    await deferred.dispatched;
    const error = new Error("response lost");
    deferred.applyThenFail(error);
    await expect(pending).rejects.toBe(error);
    expect(client.records("w").map((record) => record.memory)).toEqual(["a"]);
  });

  it("queues deferrals first-in first-out and lets later calls through", async () => {
    const client = new FakeReflectLogClient();
    const first = client.deferNext("addMemories");
    const one = add(client, ["one"]);
    const two = add(client, ["two"]);
    await expect(two).resolves.toMatchObject({ storedCount: 1 });
    expect(client.records("w").map((record) => record.memory)).toEqual(["two"]);
    first.resolve();
    await one;
    expect(client.records("w").map((record) => record.memory)).toEqual(["two", "one"]);
  });

  it("rejects a deferred call with the abort reason, unapplied by default", async () => {
    const client = new FakeReflectLogClient();
    const deferred = client.deferNext("addMemories");
    const controller = new AbortController();
    const pending = add(client, ["a"], { signal: controller.signal });
    await deferred.dispatched;
    controller.abort(new Error("caller left"));
    await expect(pending).rejects.toThrow("caller left");
    expect(client.records("w")).toEqual([]);
  });

  it("can model an abort that arrives after the server already applied the write", async () => {
    const client = new FakeReflectLogClient();
    const deferred = client.deferNext("addMemories", { applyOnAbort: true });
    const controller = new AbortController();
    const pending = add(client, ["a"], { signal: controller.signal });
    await deferred.dispatched;
    controller.abort(new Error("caller left"));
    await expect(pending).rejects.toThrow("caller left");
    expect(client.records("w")).toHaveLength(1);
  });

  it("defers other methods too", async () => {
    const client = new FakeReflectLogClient();
    client.seed("w", ["a"]);
    const deferred = client.deferNext("getAllMemories");
    const pending = client.getAllMemories({ workspaceId: "w", limit: 10, offset: 0 });
    await deferred.dispatched;
    deferred.resolve();
    await expect(pending).resolves.toMatchObject({ total: 1 });
  });

  it("holdConnect keeps the call undispatched, unhooked and unrecorded until released", async () => {
    const client = new FakeReflectLogClient();
    const hold = client.holdConnect("addMemories");
    const hook = vi.fn();
    const pending = add(client, ["a"], { beforeDispatch: hook });
    await Promise.resolve();
    expect(hook).not.toHaveBeenCalled();
    expect(client.calls).toEqual([]);
    hold.release();
    await expect(pending).resolves.toMatchObject({ storedCount: 1 });
    expect(hook).toHaveBeenCalledTimes(1);
    expect(client.callsTo("addMemories")).toHaveLength(1);
  });

  it("holdConnect failure rejects before any dispatch", async () => {
    const client = new FakeReflectLogClient();
    const hold = client.holdConnect("addMemories");
    const hook = vi.fn();
    const pending = add(client, ["a"], { beforeDispatch: hook });
    const error = new Error("connect refused");
    hold.fail(error);
    await expect(pending).rejects.toBe(error);
    expect(hook).not.toHaveBeenCalled();
    expect(client.calls).toEqual([]);
  });

  it("holdConnect lets an abort reject the held call without dispatching it", async () => {
    const client = new FakeReflectLogClient();
    const hold = client.holdConnect("addMemories");
    const controller = new AbortController();
    const hook = vi.fn();
    const pending = add(client, ["a"], { signal: controller.signal, beforeDispatch: hook });
    controller.abort(new Error("caller left"));
    await expect(pending).rejects.toThrow("caller left");
    hold.release();
    expect(hook).not.toHaveBeenCalled();
    expect(client.calls).toEqual([]);
  });
});
