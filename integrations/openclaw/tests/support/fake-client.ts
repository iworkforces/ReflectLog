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

type Method = Exclude<keyof ReflectLogClient, "close">;

export type FakeCall = { readonly method: Method; readonly params: unknown };

type Workspace = { nextId: number; records: ReflectLogRecord[] };

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

  /** Make the next call to `method` reject with `error`. */
  failNext(method: Method, error: Error): void {
    this.#failures.set(method, error);
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
    this.#enter("searchMemories", params, options);
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
    this.#enter("getMemoryById", params, options);
    const records = this.#workspace(params.workspaceId).records;
    return { record: records.find((record) => record.id === params.id) ?? null };
  }

  async getAllMemories(
    params: { workspaceId: string; limit: number; offset: number },
    options?: CallOptions,
  ): Promise<GetAllMemoriesResult> {
    this.#enter("getAllMemories", params, options);
    const records = this.#workspace(params.workspaceId).records;
    const page = records.slice(params.offset, params.offset + params.limit);
    return {
      records: page,
      total: records.length,
      offset: params.offset,
      limit: params.limit,
      truncated: params.offset + page.length < records.length,
    };
  }

  async deleteMemories(
    params: { workspaceId: string; ids: readonly number[] },
    options?: CallOptions,
  ): Promise<DeleteMemoriesResult> {
    this.#enter("deleteMemories", params, options);
    const workspace = this.#workspace(params.workspaceId);
    const removed = params.ids.filter((id) => workspace.records.some((record) => record.id === id));
    workspace.records = workspace.records.filter((record) => !removed.includes(record.id));
    return { removed, missing: params.ids.filter((id) => !removed.includes(id)) };
  }

  async addMemories(
    params: { workspaceId: string; memories: readonly string[]; dryRun: boolean; allowReplace: boolean },
    options?: CallOptions,
  ): Promise<AddMemoriesResult> {
    this.#enter("addMemories", params, options);
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
  }

  async close(): Promise<void> {
    this.closeCount += 1;
  }

  #enter(method: Method, params: unknown, options?: CallOptions): void {
    this.calls.push({ method, params });
    options?.signal?.throwIfAborted();
    const failure = this.#failures.get(method);
    if (failure !== undefined) {
      this.#failures.delete(method);
      throw failure;
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
