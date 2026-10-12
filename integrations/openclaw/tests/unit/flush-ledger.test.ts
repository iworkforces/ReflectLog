import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import {
  FlushCapacityError, FlushDisposedError, FlushLedger, type FlushOperationContext,
} from "../../src/flush-ledger.js";
import { ReflectLogError } from "../../src/reflectlog/types.js";

/** A ledger on a fake clock plus operations whose settlement the test controls. */
function harness(options: { maxEntries?: number; ttlMs?: number } = {}) {
  const clock = { now: 0 };
  const ledger = new FlushLedger<string>({ ...options, now: () => clock.now });
  const contexts: FlushOperationContext[] = [];
  /** An operation that marks itself dispatched, then waits for the test. */
  const deferred = (label = "ok") => {
    const gate = Promise.withResolvers<string>();
    const start = (context: FlushOperationContext) => {
      contexts.push(context);
      context.markDispatched();
      return gate.promise;
    };
    return { start, resolve: () => gate.resolve(label), reject: (error: unknown) => gate.reject(error) };
  };
  return { clock, ledger, contexts, deferred };
}

describe("flush ledger identity", () => {
  it("keys on flush id, workspace and the exact normalized batch", () => {
    const expected = createHash("sha256").update(JSON.stringify(["f", "w", ["a", "b"]]), "utf8").digest("hex");
    expect(FlushLedger.key("f", "w", ["a", "b"])).toBe(expected);
    expect(FlushLedger.key("f", "w", ["a", "b"])).toBe(FlushLedger.key("f", "w", ["a", "b"]));
    for (const other of [
      FlushLedger.key("g", "w", ["a", "b"]), FlushLedger.key("f", "x", ["a", "b"]),
      FlushLedger.key("f", "w", ["b", "a"]), FlushLedger.key("f", "w", ["a"]),
    ]) expect(other).not.toBe(expected);
  });

  it("rejects a capacity that cannot hold any entry", () => {
    expect(() => new FlushLedger({ maxEntries: 0 })).toThrow(RangeError);
    expect(() => new FlushLedger({ maxEntries: 1.5 })).toThrow(RangeError);
  });
});

describe("flush ledger", () => {
  it("shares one mutation across concurrent and repeated retries", async () => {
    const ledger = new FlushLedger<number>();
    let runs = 0;
    const operation = async () => {
      runs += 1;
      return runs;
    };
    const key = FlushLedger.key("flush-1", "ws", ["a"]);
    const [first, second] = await Promise.all([ledger.run(key, operation), ledger.run(key, operation)]);
    const third = await ledger.run(key, operation);
    expect(runs).toBe(1);
    expect([first.deduplicated, second.deduplicated, third.deduplicated]).toEqual([false, true, true]);
  });

  it("forgets a failure proven to precede dispatch so a retry can run", async () => {
    const ledger = new FlushLedger<string>();
    const key = FlushLedger.key("flush-1", "ws", ["a"]);
    await expect(ledger.run(key, async () => Promise.reject(new Error("boom")))).rejects.toThrow("boom");
    expect(ledger.size).toBe(0);
    expect(await ledger.run(key, async () => "ok")).toEqual({ value: "ok", deduplicated: false });
  });

  it("expires and bounds entries", async () => {
    const { clock, ledger } = harness({ maxEntries: 2, ttlMs: 10 });
    await ledger.run("a", async () => "a");
    await ledger.run("b", async () => "b");
    await ledger.run("c", async () => "c");
    expect(ledger.size).toBe(2);
    clock.now = 100;
    expect((await ledger.run("c", async () => "c2")).value).toBe("c2");
  });

  describe("reservation", () => {
    it("reserves the entry and registers the caller before the callback runs", async () => {
      const { ledger } = harness();
      const seen: string[] = [];
      const outcome = await ledger.run("k", async () => {
        seen.push(`size=${ledger.size} pending=${ledger.counts.pending}`);
        return "v";
      }, () => { seen.push(`registered size=${ledger.size}`); });
      expect(seen).toEqual(["registered size=1", "size=1 pending=1"]);
      expect(outcome.value).toBe("v");
    });

    it("dispatches once when the same key re-enters synchronously during the callback", async () => {
      const { ledger } = harness();
      let runs = 0;
      let reentrant: Promise<{ value: string; deduplicated: boolean }> | undefined;
      const start = async () => {
        runs += 1;
        reentrant = ledger.run("k", async () => {
          runs += 1;
          return "second";
        });
        return "first";
      };
      const outer = await ledger.run("k", start);
      expect(reentrant).toBeDefined();
      expect(await reentrant).toEqual({ value: "first", deduplicated: true });
      expect(outer).toEqual({ value: "first", deduplicated: false });
      expect(runs).toBe(1);
    });

    it("captures a synchronous throw as a rejection of the reserved entry and frees the key", async () => {
      const { ledger } = harness();
      let reentrant: Promise<unknown> | undefined;
      const start = (): Promise<string> => {
        reentrant = ledger.run("k", async () => "never");
        throw new Error("sync boom");
      };
      await expect(ledger.run("k", start)).rejects.toThrow("sync boom");
      await expect(reentrant).rejects.toThrow("sync boom");
      expect(ledger.size).toBe(0);
      expect((await ledger.run("k", async () => "retry")).value).toBe("retry");
    });

    it("pins the key when a synchronous throw happens after the effect was marked dispatched", async () => {
      const { ledger } = harness();
      const start = (context: FlushOperationContext): Promise<string> => {
        context.markDispatched();
        throw new Error("sync boom after dispatch");
      };
      await expect(ledger.run("k", start)).rejects.toMatchObject({ code: "outcome_unknown" });
      let runs = 0;
      await expect(ledger.run("k", async () => { runs += 1; return "x"; })).rejects.toMatchObject({ code: "outcome_unknown" });
      expect(runs).toBe(0);
    });
  });

  describe("pending entries", () => {
    it("share one dispatch when the flush stays pending past its TTL", async () => {
      const { clock, ledger, contexts, deferred } = harness({ ttlMs: 10 });
      const operation = deferred("late");
      const first = ledger.run("k", operation.start);
      clock.now = 1_000;
      const second = ledger.run("k", deferred("again").start);
      operation.resolve();
      expect(await first).toEqual({ value: "late", deduplicated: false });
      expect(await second).toEqual({ value: "late", deduplicated: true });
      expect(contexts).toHaveLength(1);
    });

    it("survive capacity pressure: a pending entry is never evicted for distinct work", async () => {
      const { ledger, contexts, deferred } = harness({ maxEntries: 2 });
      const a = deferred("a");
      const b = deferred("b");
      const first = ledger.run("a", a.start);
      const second = ledger.run("b", b.start);
      const third = deferred("c");
      await expect(ledger.run("c", third.start)).rejects.toBeInstanceOf(FlushCapacityError);
      expect(contexts).toHaveLength(2);
      expect(ledger.size).toBe(2);
      expect(ledger.counts).toEqual({ pending: 2, success: 0, indeterminate: 0 });
      const rejoin = ledger.run("a", deferred("again").start);
      a.resolve();
      b.resolve();
      expect((await Promise.all([first, second, rejoin])).map((outcome) => outcome.deduplicated)).toEqual([false, false, true]);
      expect(contexts).toHaveLength(2);
    });

    it("let completed successes make room once they settle", async () => {
      const { ledger, deferred } = harness({ maxEntries: 1 });
      const a = deferred("a");
      const pendingA = ledger.run("a", a.start);
      await expect(ledger.run("b", deferred("b").start)).rejects.toBeInstanceOf(FlushCapacityError);
      a.resolve();
      await pendingA;
      expect((await ledger.run("b", async () => "b")).value).toBe("b");
      expect(ledger.size).toBe(1);
      expect(ledger.counts.success).toBe(1);
    });

    it("reject distinct work without reserving or invoking it when pending and indeterminate entries fill the ledger", async () => {
      const { ledger, contexts, deferred } = harness({ maxEntries: 2 });
      const stuck = deferred();
      const pending = ledger.run("pending", stuck.start);
      const lost = deferred();
      const unconfirmed = ledger.run("lost", lost.start);
      lost.reject(new Error("socket reset"));
      await expect(unconfirmed).rejects.toMatchObject({ code: "outcome_unknown" });
      let invoked = false;
      await expect(ledger.run("new", async () => { invoked = true; return "x"; })).rejects.toBeInstanceOf(FlushCapacityError);
      expect(invoked).toBe(false);
      expect(ledger.size).toBe(2);
      await expect(ledger.run("lost", deferred().start)).rejects.toMatchObject({ code: "outcome_unknown" });
      expect(contexts).toHaveLength(2);
      stuck.resolve();
      await pending;
    });

    it("never let the ledger exceed its bound across many distinct keys", async () => {
      const { clock, ledger } = harness({ maxEntries: 3, ttlMs: 5 });
      for (let index = 0; index < 50; index++) {
        clock.now += index % 4;
        await ledger.run(`key-${index}`, async () => "v");
        expect(ledger.size).toBeLessThanOrEqual(3);
      }
    });
  });

  describe("successes", () => {
    it("start their TTL at settlement, not at dispatch", async () => {
      const { clock, ledger, deferred } = harness({ ttlMs: 10 });
      const operation = deferred("v");
      const pending = ledger.run("k", operation.start);
      clock.now = 100;
      operation.resolve();
      await pending;
      clock.now = 109;
      expect(await ledger.run("k", deferred("again").start)).toEqual({ value: "v", deduplicated: true });
      clock.now = 110;
      const again = deferred("fresh");
      const rerun = ledger.run("k", again.start);
      again.resolve();
      expect(await rerun).toEqual({ value: "fresh", deduplicated: false });
    });

    it("are evicted oldest-first, and expired ones are treated as absent", async () => {
      const { clock, ledger } = harness({ maxEntries: 2, ttlMs: 100 });
      await ledger.run("a", async () => "a");
      clock.now = 10;
      await ledger.run("b", async () => "b");
      clock.now = 20;
      await ledger.run("c", async () => "c");
      expect(ledger.size).toBe(2);
      expect((await ledger.run("b", async () => "b2")).deduplicated).toBe(true);
      expect((await ledger.run("a", async () => "a2")).value).toBe("a2");
      clock.now = 500;
      expect((await ledger.run("b", async () => "b3")).value).toBe("b3");
      expect(ledger.size).toBe(1);
    });
  });

  describe("indeterminate outcomes", () => {
    it("pin the key: joins reject with outcome_unknown and the operation never runs again", async () => {
      const { clock, ledger, contexts, deferred } = harness({ ttlMs: 10 });
      const operation = deferred();
      const first = ledger.run("k", operation.start);
      const joined = ledger.run("k", deferred().start);
      const socket = new Error("socket reset");
      operation.reject(socket);
      const outcomes = await Promise.allSettled([first, joined]);
      for (const outcome of outcomes) {
        expect(outcome).toMatchObject({ status: "rejected", reason: { code: "outcome_unknown", cause: socket } });
      }
      clock.now = 1_000_000;
      await expect(ledger.run("k", deferred().start)).rejects.toMatchObject({ code: "outcome_unknown" });
      expect(contexts).toHaveLength(1);
      expect(ledger.counts).toEqual({ pending: 0, success: 0, indeterminate: 1 });
    });

    it("keep an outcome_unknown the client already produced", async () => {
      const { ledger, deferred } = harness();
      const operation = deferred();
      const pending = ledger.run("k", operation.start);
      const original = new ReflectLogError("outcome_unknown", "ReflectLog call aborted");
      operation.reject(original);
      await expect(pending).rejects.toBe(original);
    });

    it("treat even a ReflectLogError of another code as unconfirmed once dispatched", async () => {
      const { ledger, deferred } = harness();
      const operation = deferred();
      const pending = ledger.run("k", operation.start);
      operation.reject(new ReflectLogError("tool_error", "server said no"));
      await expect(pending).rejects.toMatchObject({ code: "outcome_unknown", cause: { code: "tool_error" } });
    });
  });

  describe("abandon", () => {
    it("releases the key immediately as a proven pre-dispatch failure", async () => {
      const { ledger, contexts, deferred } = harness();
      const gate = Promise.withResolvers<string>();
      const first = ledger.run("k", (context) => { contexts.push(context); return gate.promise; });
      const veto = new Error("nobody left");
      expect(contexts[0]?.abandon(veto)).toBe(veto);
      expect(ledger.size).toBe(0);
      const retry = deferred("retry");
      const second = ledger.run("k", retry.start);
      retry.resolve();
      await expect(first).rejects.toBe(veto);
      expect(await second).toEqual({ value: "retry", deduplicated: false });
      gate.reject(new Error("late failure of the abandoned operation"));
      await Promise.resolve();
      expect(ledger.counts).toEqual({ pending: 0, success: 1, indeterminate: 0 });
    });

    it("does nothing once the effect was dispatched", async () => {
      const { ledger, contexts, deferred } = harness();
      const operation = deferred("v");
      const pending = ledger.run("k", operation.start);
      contexts[0]?.abandon(new Error("too late"));
      expect(ledger.counts.pending).toBe(1);
      operation.resolve();
      expect((await pending).value).toBe("v");
    });
  });

  describe("disposal", () => {
    it("settles a dispatched waiter as outcome_unknown, aborts the owned signal and drops the late result", async () => {
      const { ledger, contexts, deferred } = harness();
      const operation = deferred("late");
      const pending = ledger.run("k", operation.start);
      const joined = ledger.run("k", deferred().start);
      ledger.dispose();
      const outcomes = await Promise.allSettled([pending, joined]);
      for (const outcome of outcomes) expect(outcome).toMatchObject({ status: "rejected", reason: { code: "outcome_unknown" } });
      expect(contexts[0]?.signal.aborted).toBe(true);
      expect(ledger.size).toBe(0);
      operation.resolve();
      await Promise.resolve();
      expect(ledger.size).toBe(0);
    });

    it("settles a waiter that was never dispatched with the typed pre-dispatch error", async () => {
      const { ledger } = harness();
      const gate = Promise.withResolvers<string>();
      const pending = ledger.run("k", () => gate.promise);
      ledger.dispose();
      await expect(pending).rejects.toBeInstanceOf(FlushDisposedError);
    });

    it("rejects all later work, never becomes usable again and forgets pinned and cached state", async () => {
      const { ledger, deferred } = harness();
      await ledger.run("done", async () => "v");
      const lost = deferred();
      const unconfirmed = ledger.run("lost", lost.start);
      lost.reject(new Error("socket reset"));
      await expect(unconfirmed).rejects.toMatchObject({ code: "outcome_unknown" });
      ledger.dispose();
      ledger.dispose();
      expect(ledger.disposed).toBe(true);
      expect(ledger.size).toBe(0);
      for (const key of ["done", "lost", "fresh"]) {
        await expect(ledger.run(key, async () => "x")).rejects.toBeInstanceOf(FlushDisposedError);
      }
      expect(ledger.size).toBe(0);
    });
  });
});
