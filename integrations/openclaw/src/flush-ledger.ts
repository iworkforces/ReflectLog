import { createHash } from "node:crypto";

type LedgerEntry<T> = { readonly promise: Promise<T>; readonly expiresAt: number };

/**
 * Best-effort, process-local dedup of flush persistence retries.
 *
 * The host gives every retry of one pre-compaction flush the same `flushId`;
 * combined with the target workspace and the exact batch it identifies one
 * mutation. In-flight and successful calls are shared, failures are forgotten
 * so a retry may try again. State does not survive a gateway restart.
 */
export class FlushLedger<T> {
  readonly #entries = new Map<string, LedgerEntry<T>>();
  readonly #maxEntries: number;
  readonly #ttlMs: number;
  readonly #now: () => number;

  constructor(options: { maxEntries?: number; ttlMs?: number; now?: () => number } = {}) {
    this.#maxEntries = options.maxEntries ?? 512;
    this.#ttlMs = options.ttlMs ?? 6 * 60 * 60 * 1000;
    this.#now = options.now ?? Date.now;
  }

  static key(flushId: string, workspaceId: string, memories: readonly string[]): string {
    return createHash("sha256")
      .update(JSON.stringify([flushId, workspaceId, memories]), "utf8")
      .digest("hex");
  }

  async run(key: string, operation: () => Promise<T>): Promise<{ value: T; deduplicated: boolean }> {
    const now = this.#now();
    const existing = this.#entries.get(key);
    if (existing !== undefined && existing.expiresAt > now) {
      return { value: await existing.promise, deduplicated: true };
    }
    this.#evict(now);
    const promise = operation();
    this.#entries.set(key, { promise, expiresAt: now + this.#ttlMs });
    try {
      return { value: await promise, deduplicated: false };
    } catch (error) {
      if (this.#entries.get(key)?.promise === promise) {
        this.#entries.delete(key);
      }
      throw error;
    }
  }

  get size(): number {
    return this.#entries.size;
  }

  #evict(now: number): void {
    for (const [key, entry] of this.#entries) {
      if (entry.expiresAt <= now) {
        this.#entries.delete(key);
      }
    }
    while (this.#entries.size >= this.#maxEntries) {
      const oldest = this.#entries.keys().next();
      if (oldest.done === true) {
        break;
      }
      this.#entries.delete(oldest.value);
    }
  }
}
