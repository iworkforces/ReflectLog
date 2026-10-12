import { FlushLedger, type FlushEntryRef, type FlushOutcome } from "./flush-ledger.js";
import { ReflectLogError } from "./reflectlog/types.js";
import { waitFor } from "./reflectlog/wait-for.js";

/** Authority of one caller; structurally satisfied by the service's `EffectGuard`. */
export type FlushWaiterGuard = Readonly<{
  assertCurrent: () => void;
  signal?: AbortSignal | undefined;
}>;

export type SharedFlushRun = Readonly<{
  /** Owned by the ledger, never a caller's signal: no waiter can cancel another's write. */
  signal: AbortSignal;
  /** Pass to the client; vetoes the dispatch unless at least one waiter is still current. */
  beforeDispatch: () => void;
}>;

type Waiter = { readonly guard: FlushWaiterGuard };

/**
 * Shares one flush write between callers while keeping authority per caller.
 *
 * A caller's authority is checked before it joins, at dispatch (at least one
 * current waiter must remain) and again before it receives the settled result.
 * A waiter that is cancelled or revoked only detaches itself; the write keeps
 * going for the others, and for the ledger when nobody is left after dispatch.
 * Cached outcomes keep no guards: waiters live only in a per-entry set that is
 * emptied as each caller finishes. Not restart-safe; see {@link FlushLedger}.
 */
export class FlushCoordinator<T> {
  readonly #ledger: FlushLedger<T>;
  readonly #waiters = new WeakMap<FlushEntryRef, Set<Waiter>>();

  constructor(options: ConstructorParameters<typeof FlushLedger>[0] = {}) {
    this.#ledger = new FlushLedger<T>(options);
  }

  get size(): number {
    return this.#ledger.size;
  }

  get counts(): FlushLedger<T>["counts"] {
    return this.#ledger.counts;
  }

  async run(
    key: string,
    guard: FlushWaiterGuard,
    start: (shared: SharedFlushRun) => Promise<T>,
  ): Promise<FlushOutcome<T>> {
    guard.signal?.throwIfAborted();
    guard.assertCurrent();
    const waiter: Waiter = { guard };
    let joined: FlushEntryRef | undefined;
    try {
      const outcome = await waitFor(
        this.#ledger.run(
          key,
          (context) => start({
            signal: context.signal,
            beforeDispatch: () => {
              if (!this.#hasCurrentWaiter(context.entry)) {
                throw context.abandon(new ReflectLogError("aborted", "every flush caller left before dispatch"));
              }
              context.markDispatched();
            },
          }),
          (entry) => {
            joined = entry;
            let waiters = this.#waiters.get(entry);
            if (waiters === undefined) {
              waiters = new Set();
              this.#waiters.set(entry, waiters);
            }
            waiters.add(waiter);
          },
        ),
        guard.signal,
      );
      guard.signal?.throwIfAborted();
      guard.assertCurrent();
      return outcome;
    } catch (error) {
      guard.assertCurrent();
      throw error;
    } finally {
      if (joined !== undefined) this.#waiters.get(joined)?.delete(waiter);
    }
  }

  /** Reject new work, settle pending waiters conservatively and drop all state for good. */
  dispose(): void {
    this.#ledger.dispose();
  }

  /** Drops waiters that are aborted or revoked; each of them still re-checks itself on release. */
  #hasCurrentWaiter(entry: FlushEntryRef): boolean {
    const waiters = this.#waiters.get(entry);
    if (waiters === undefined) return false;
    let current = 0;
    for (const waiter of waiters) {
      if (waiter.guard.signal?.aborted === true) {
        waiters.delete(waiter);
        continue;
      }
      try {
        waiter.guard.assertCurrent();
        current += 1;
      } catch {
        waiters.delete(waiter);
      }
    }
    return current > 0;
  }
}
