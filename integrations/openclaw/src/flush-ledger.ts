import { createHash } from "node:crypto";
import { ReflectLogError } from "./reflectlog/types.js";

/** Pre-dispatch rejection: pending and unconfirmed flushes fill the ledger. Nothing was reserved or sent. */
export class FlushCapacityError extends Error {
  constructor() {
    super("Too many ReflectLog flush writes are pending or unconfirmed; nothing was sent. Retry later.");
    this.name = "FlushCapacityError";
  }
}

/** Pre-dispatch rejection: the owning memory backend lifecycle has ended. Nothing was sent. */
export class FlushDisposedError extends Error {
  constructor() {
    super("The ReflectLog memory backend was shut down; nothing was sent. Retry once it is available.");
    this.name = "FlushDisposedError";
  }
}

/** Opaque identity of one pending entry; lets a coordinator attach per-entry state without the ledger knowing it. */
export type FlushEntryRef = object;

export type FlushOperationContext = Readonly<{
  entry: FlushEntryRef;
  /** Owned by the ledger, never by a caller; aborted only by {@link FlushLedger.dispose}. */
  signal: AbortSignal;
  /**
   * Call synchronously and immediately before the effect may leave the
   * process. From then on a failure is indeterminate and pins the key.
   */
  markDispatched(): void;
  /**
   * Settle this entry as a proven pre-dispatch failure right now (so the key
   * is free for the next caller) and return `error` for the caller to throw.
   */
  abandon(error: Error): Error;
}>;

export type FlushOutcome<T> = Readonly<{ value: T; deduplicated: boolean }>;

type Pending<T> = {
  readonly state: "pending";
  readonly ref: FlushEntryRef;
  readonly deferred: PromiseWithResolvers<T>;
  readonly controller: AbortController;
  dispatched: boolean;
  settled: boolean;
};
type Succeeded<T> = { readonly state: "success"; readonly value: T; readonly expiresAt: number };
type Indeterminate = { readonly state: "indeterminate"; readonly error: ReflectLogError };
type Entry<T> = Pending<T> | Succeeded<T> | Indeterminate;

function outcomeUnknown(cause: unknown): ReflectLogError {
  return cause instanceof ReflectLogError && cause.code === "outcome_unknown"
    ? cause
    : new ReflectLogError("outcome_unknown", "ReflectLog did not confirm the flush write", { cause });
}

/**
 * Process-local, bounded dedup of one backend's flush writes. It is NOT
 * restart-safe and makes no exactly-once claim: after a gateway restart, or
 * once a success expires, the same flush may be written again.
 *
 * The host gives every retry of one pre-compaction flush the same `flushId`;
 * with the target workspace and the exact batch it identifies one mutation.
 * Each key moves through one state machine:
 *
 * - `pending`: reserved BEFORE the operation runs, so a synchronous throw or a
 *   reentrant call of the same key sees it. Never evicted or expired.
 * - `success`: cached from settlement until `settlement + ttlMs`; the only
 *   evictable state.
 * - `indeterminate`: the operation failed after it may have been dispatched.
 *   Pinned until {@link dispose}; joins reject with `outcome_unknown` and the
 *   operation is never run again.
 * - absent: a failure proven to precede dispatch frees the key for a retry.
 *
 * When pending and indeterminate entries alone reach `maxEntries`, distinct
 * new work is rejected with {@link FlushCapacityError} instead of evicting.
 */
export class FlushLedger<T> {
  readonly #entries = new Map<string, Entry<T>>();
  readonly #maxEntries: number;
  readonly #ttlMs: number;
  readonly #now: () => number;
  #disposed = false;

  constructor(options: { maxEntries?: number; ttlMs?: number; now?: () => number } = {}) {
    this.#maxEntries = options.maxEntries ?? 512;
    this.#ttlMs = options.ttlMs ?? 6 * 60 * 60 * 1000;
    this.#now = options.now ?? Date.now;
    if (!Number.isInteger(this.#maxEntries) || this.#maxEntries < 1) {
      throw new RangeError("maxEntries must be a positive integer");
    }
  }

  static key(flushId: string, workspaceId: string, memories: readonly string[]): string {
    return createHash("sha256")
      .update(JSON.stringify([flushId, workspaceId, memories]), "utf8")
      .digest("hex");
  }

  /**
   * Run `start` once per key, or join what is already pending/cached.
   * `onPending` is invoked synchronously for the caller that reserves the
   * entry (before `start`) and for every caller that joins it while pending.
   */
  async run(
    key: string,
    start: (context: FlushOperationContext) => Promise<T>,
    onPending?: (entry: FlushEntryRef) => void,
  ): Promise<FlushOutcome<T>> {
    if (this.#disposed) throw new FlushDisposedError();
    const now = this.#now();
    const existing = this.#entries.get(key);
    if (existing !== undefined) {
      switch (existing.state) {
        case "pending":
          onPending?.(existing.ref);
          return { value: await existing.deferred.promise, deduplicated: true };
        case "indeterminate":
          throw existing.error;
        case "success":
          if (existing.expiresAt > now) return { value: existing.value, deduplicated: true };
          this.#entries.delete(key);
      }
    }
    this.#makeRoom(now);
    const entry: Pending<T> = {
      state: "pending", ref: {}, deferred: Promise.withResolvers<T>(),
      controller: new AbortController(), dispatched: false, settled: false,
    };
    this.#entries.set(key, entry);
    onPending?.(entry.ref);
    this.#start(key, entry, start);
    return { value: await entry.deferred.promise, deduplicated: false };
  }

  get size(): number {
    return this.#entries.size;
  }

  get counts(): Readonly<{ pending: number; success: number; indeterminate: number }> {
    const counts = { pending: 0, success: 0, indeterminate: 0 };
    for (const entry of this.#entries.values()) counts[entry.state] += 1;
    return counts;
  }

  get disposed(): boolean {
    return this.#disposed;
  }

  /**
   * End this ledger for good. Pending entries are aborted and their waiters
   * settle conservatively: dispatched => `outcome_unknown`, otherwise a
   * pre-dispatch {@link FlushDisposedError}. All state, including pinned
   * indeterminate keys, is dropped and nothing can run again.
   */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const entries = [...this.#entries.values()];
    this.#entries.clear();
    for (const entry of entries) {
      if (entry.state !== "pending" || entry.settled) continue;
      entry.settled = true;
      const disposed = new FlushDisposedError();
      entry.deferred.reject(entry.dispatched ? outcomeUnknown(disposed) : disposed);
      entry.controller.abort(disposed);
    }
  }

  #start(key: string, entry: Pending<T>, start: (context: FlushOperationContext) => Promise<T>): void {
    const context: FlushOperationContext = {
      entry: entry.ref,
      signal: entry.controller.signal,
      markDispatched: () => { entry.dispatched = true; },
      abandon: (error) => {
        if (!entry.dispatched) this.#settleFailure(key, entry, error);
        return error;
      },
    };
    let operation: Promise<T>;
    try {
      operation = start(context);
    } catch (error) {
      this.#settleFailure(key, entry, error);
      return;
    }
    void operation.then(
      (value) => this.#settleSuccess(key, entry, value),
      (error: unknown) => this.#settleFailure(key, entry, error),
    );
  }

  #settleSuccess(key: string, entry: Pending<T>, value: T): void {
    if (entry.settled) return;
    entry.settled = true;
    if (this.#entries.get(key) === entry) {
      this.#entries.delete(key);
      this.#entries.set(key, { state: "success", value, expiresAt: this.#now() + this.#ttlMs });
    }
    entry.deferred.resolve(value);
  }

  #settleFailure(key: string, entry: Pending<T>, error: unknown): void {
    if (entry.settled) return;
    entry.settled = true;
    if (!entry.dispatched) {
      if (this.#entries.get(key) === entry) this.#entries.delete(key);
      entry.deferred.reject(error);
      return;
    }
    const unconfirmed = outcomeUnknown(error);
    if (this.#entries.get(key) === entry) this.#entries.set(key, { state: "indeterminate", error: unconfirmed });
    entry.deferred.reject(unconfirmed);
  }

  /** Free one slot for a new key by dropping expired, then oldest completed, successes only. */
  #makeRoom(now: number): void {
    for (const [key, entry] of this.#entries) {
      if (entry.state === "success" && entry.expiresAt <= now) this.#entries.delete(key);
    }
    while (this.#entries.size >= this.#maxEntries) {
      let oldest: string | undefined;
      for (const [key, entry] of this.#entries) {
        if (entry.state === "success") {
          oldest = key;
          break;
        }
      }
      if (oldest === undefined) throw new FlushCapacityError();
      this.#entries.delete(oldest);
    }
  }
}
