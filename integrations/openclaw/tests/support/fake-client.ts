import type {
  AddMemoriesResult,
  CallOptions,
  GetMemoryByIdResult,
  GetAllMemoriesResult,
  ReflectLogClient,
  ReflectLogRecord,
  DeleteMemoriesResult,
  ScoreKind,
  SearchMemoriesResult,
} from "../../src/reflectlog/types.js";
import { waitFor } from "../../src/reflectlog/wait-for.js";

type Method = Exclude<keyof ReflectLogClient, "close">;

export type FakeCall = { readonly method: Method; readonly params: unknown };

type Workspace = { nextId: number; records: ReflectLogRecord[] };

type Verdict =
  | { readonly kind: "apply" }
  | { readonly kind: "discard"; readonly error: unknown }
  | { readonly kind: "lose"; readonly error: unknown };

/**
 * Holds one call in its connection-acquisition phase, before `beforeDispatch`
 * and before anything is recorded (see {@link FakeReflectLogClient.holdConnect}).
 */
export class HeldConnect {
  readonly #settled = Promise.withResolvers<void>();

  /** Let the connection come up; the call proceeds to dispatch. */
  release(): void {
    this.#settled.resolve();
  }

  /** Fail the connection: the call rejects with `error` before any dispatch. */
  fail(error: unknown): void {
    this.#settled.reject(error);
  }

  /** @internal Used by the fake client only. */
  get promise(): Promise<void> {
    return this.#settled.promise;
  }
}

/**
 * Controls one deferred call (see {@link FakeReflectLogClient.deferNext}). The
 * first decision wins; later ones are ignored. If the caller's signal aborts
 * first, the call rejects with the abort reason and is not applied unless
 * `applyOnAbort` was requested (models an abort that arrives after the server
 * already acted).
 */
export class DeferredCall {
  /** Resolves when the call is handed to the fake transport (after `beforeDispatch`). */
  readonly dispatched: Promise<void>;
  readonly #markDispatched: () => void;
  readonly #verdict = Promise.withResolvers<Verdict>();
  readonly #applyOnAbort: boolean;

  constructor(applyOnAbort: boolean) {
    const dispatched = Promise.withResolvers<void>();
    this.dispatched = dispatched.promise;
    this.#markDispatched = dispatched.resolve;
    this.#applyOnAbort = applyOnAbort;
  }

  /** Apply the call to the fake store and resolve it normally. */
  resolve(): void {
    this.#verdict.resolve({ kind: "apply" });
  }

  /** Reject without applying anything: the server never acted. */
  failUnapplied(error: unknown): void {
    this.#verdict.resolve({ kind: "discard", error });
  }

  /** Apply the call, then reject: partial success or a lost response. */
  applyThenFail(error: unknown): void {
    this.#verdict.resolve({ kind: "lose", error });
  }

  /** @internal Used by the fake client only. */
  dispatch(signal: AbortSignal | undefined): Promise<Verdict> {
    this.#markDispatched();
    if (signal === undefined) return this.#verdict.promise;
    const onAbort = () => this.#verdict.resolve(
      this.#applyOnAbort ? { kind: "lose", error: signal.reason } : { kind: "discard", error: signal.reason });
    if (signal.aborted) onAbort();
    else signal.addEventListener("abort", onAbort, { once: true });
    return this.#verdict.promise.finally(() => signal.removeEventListener("abort", onAbort));
  }
}

function tokens(text: string): Set<string> {
  return new Set(text.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((token) => token !== ""));
}

/**
 * In-memory ReflectLog with per-workspace AUTOINCREMENT ids, exact-content
 * dedup and token-overlap scoring. Records every call for assertions.
 */
export class FakeReflectLogClient implements ReflectLogClient {
  readonly calls: FakeCall[] = [];
  scoreKind: ScoreKind = "reranker";
  closeCount = 0;
  readonly #workspaces = new Map<string, Workspace>();
  readonly #failures = new Map<Method, Error>();
  readonly #preDispatchFailures = new Map<Method, Error>();
  readonly #deferrals = new Map<Method, DeferredCall[]>();
  readonly #holds = new Map<Method, HeldConnect[]>();

  /**
   * Make the next call to `method` reject with `error` AFTER it was dispatched:
   * the call is recorded and `beforeDispatch` has run, but nothing is applied.
   */
  failNext(method: Method, error: Error): void {
    this.#failures.set(method, error);
  }

  /**
   * Make the next call to `method` reject with `error` BEFORE dispatch: neither
   * `beforeDispatch` nor the call record happens. Models a proven pre-send
   * failure such as a refused connection.
   */
  failBeforeDispatch(method: Method, error: Error): void {
    this.#preDispatchFailures.set(method, error);
  }

  /**
   * Hold the next call to `method` while it "connects": it is not dispatched,
   * recorded or hooked until the controller releases it, and a signal abort
   * during the hold rejects it with the abort reason. Holds queue FIFO.
   */
  holdConnect(method: Method): HeldConnect {
    const hold = new HeldConnect();
    const queue = this.#holds.get(method) ?? [];
    queue.push(hold);
    this.#holds.set(method, queue);
    return hold;
  }

  /**
   * Hold the next call to `method` after dispatch until the returned controller
   * decides its fate. Deferrals queue in FIFO order, one per call.
   */
  deferNext(method: Method, options: { readonly applyOnAbort?: boolean } = {}): DeferredCall {
    const deferred = new DeferredCall(options.applyOnAbort === true);
    const queue = this.#deferrals.get(method) ?? [];
    queue.push(deferred);
    this.#deferrals.set(method, queue);
    return deferred;
  }

  seed(workspaceId: string, memories: readonly string[]): ReflectLogRecord[] {
    const workspace = this.#workspace(workspaceId);
    return memories.map((memory) => {
      const record = { id: workspace.nextId++, memory, createdAt: "2026-10-10 00:00:00" };
      workspace.records.push(record);
      return record;
    });
  }

  records(workspaceId: string): readonly ReflectLogRecord[] {
    return this.#workspace(workspaceId).records;
  }

  callsTo(method: Method): FakeCall[] {
    return this.calls.filter((call) => call.method === method);
  }

  async searchMemories(
    params: { workspaceId: string; query: string; limit: number },
    options?: CallOptions,
  ): Promise<SearchMemoriesResult> {
    return this.#run("searchMemories", params, options, () => this.#search(params));
  }

  #search(params: { workspaceId: string; query: string; limit: number }): SearchMemoriesResult {
    const queryTokens = tokens(params.query);
    const scored = this.#workspace(params.workspaceId)
      .records.map((record) => {
        const recordTokens = tokens(record.memory);
        const overlap = [...queryTokens].filter((token) => recordTokens.has(token)).length;
        return { ...record, score: queryTokens.size === 0 ? 0 : overlap / queryTokens.size };
      })
      .filter((record) => record.score > 0)
      .sort((left, right) => right.score - left.score || left.id - right.id)
      .slice(0, params.limit);
    return { results: scored, scoreKind: scored.length > 0 ? this.scoreKind : "none", limit: params.limit };
  }

  async getMemoryById(
    params: { workspaceId: string; id: number },
    options?: CallOptions,
  ): Promise<GetMemoryByIdResult> {
    return this.#run("getMemoryById", params, options, () => {
      const records = this.#workspace(params.workspaceId).records;
      return { record: records.find((record) => record.id === params.id) ?? null };
    });
  }

  async getAllMemories(
    params: { workspaceId: string; limit: number; offset: number },
    options?: CallOptions,
  ): Promise<GetAllMemoriesResult> {
    return this.#run("getAllMemories", params, options, () => {
      const records = this.#workspace(params.workspaceId).records;
      const page = records.slice(params.offset, params.offset + params.limit);
      return {
        records: page,
        total: records.length,
        offset: params.offset,
        limit: params.limit,
        truncated: params.offset + page.length < records.length,
      };
    });
  }

  async deleteMemories(
    params: { workspaceId: string; ids: readonly number[] },
    options?: CallOptions,
  ): Promise<DeleteMemoriesResult> {
    return this.#run("deleteMemories", params, options, () => {
      const workspace = this.#workspace(params.workspaceId);
      const removed = params.ids.filter((id) => workspace.records.some((record) => record.id === id));
      workspace.records = workspace.records.filter((record) => !removed.includes(record.id));
      return { removed, missing: params.ids.filter((id) => !removed.includes(id)) };
    });
  }

  async addMemories(
    params: { workspaceId: string; memories: readonly string[]; dryRun: boolean; allowReplace: boolean },
    options?: CallOptions,
  ): Promise<AddMemoriesResult> {
    return this.#run("addMemories", params, options, () => {
      const workspace = this.#workspace(params.workspaceId);
      let storedCount = 0;
      let skippedCount = 0;
      for (const memory of params.memories) {
        if (workspace.records.some((record) => record.memory === memory)) {
          skippedCount += 1;
          continue;
        }
        storedCount += 1;
        if (!params.dryRun) {
          workspace.records.push({ id: workspace.nextId++, memory, createdAt: "2026-10-10 00:00:00" });
        }
      }
      return { storedCount, skippedCount, replacedCount: 0, replacements: [], dryRun: params.dryRun };
    });
  }

  async close(): Promise<void> {
    this.closeCount += 1;
  }

  /**
   * Order: abort -> connect hold -> abort -> pre-dispatch failure -> `beforeDispatch` -> record (the
   * call is now dispatched) -> post-dispatch failure or deferral -> apply.
   * Everything up to the deferral runs synchronously with the caller.
   */
  async #run<R>(method: Method, params: unknown, options: CallOptions | undefined, apply: () => R): Promise<R> {
    options?.signal?.throwIfAborted();
    const hold = this.#holds.get(method)?.shift();
    if (hold !== undefined) {
      await waitFor(hold.promise, options?.signal);
      options?.signal?.throwIfAborted();
    }
    const early = this.#preDispatchFailures.get(method);
    if (early !== undefined) {
      this.#preDispatchFailures.delete(method);
      throw early;
    }
    options?.beforeDispatch?.();
    this.calls.push({ method, params });
    const failure = this.#failures.get(method);
    if (failure !== undefined) {
      this.#failures.delete(method);
      throw failure;
    }
    const deferral = this.#deferrals.get(method)?.shift();
    if (deferral === undefined) return apply();
    const verdict = await deferral.dispatch(options?.signal);
    switch (verdict.kind) {
      case "apply": return apply();
      case "discard": throw verdict.error;
      case "lose": apply(); throw verdict.error;
    }
  }

  #workspace(workspaceId: string): Workspace {
    let workspace = this.#workspaces.get(workspaceId);
    if (workspace === undefined) {
      workspace = { nextId: 1, records: [] };
      this.#workspaces.set(workspaceId, workspace);
    }
    return workspace;
  }
}

export const allowAll = { assertCurrent: (): void => {} };
