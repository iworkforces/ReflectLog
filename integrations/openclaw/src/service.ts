import type { MemoryScope } from "./access.js";
import { type ReflectLogPluginConfig, MAX_SEARCH_RESULTS } from "./config.js";
import { FlushCoordinator } from "./flush-coordinator.js";
import { FlushDisposedError, FlushLedger } from "./flush-ledger.js";
import { looksLikePromptInjection, normalizeMemoryText } from "./policy.js";
import { formatRef, parseRef, revisionOf } from "./references.js";
import { ReflectLogError } from "./reflectlog/types.js";
import type {
  AddMemoriesResult,
  ReflectLogClient,
  ReflectLogRecord,
  ScoreKind,
  SearchExplanation,
  StageOutcomes,
} from "./reflectlog/types.js";

/**
 * Authority guard for one caller. `assertCurrent` throws once the caller's
 * authority is revoked; the service calls it before every backend effect and
 * again before releasing any data obtained after an await.
 */
export type EffectGuard = Readonly<{
  assertCurrent: () => void;
  signal?: AbortSignal | undefined;
}>;

export type MemoryEntry = Readonly<{
  ref: string;
  id: number;
  text: string;
  revision: string;
  createdAt: string | null;
}>;

export type MemoryHit = MemoryEntry & Readonly<{
  score: number | null;
  explanation?: SearchExplanation & Readonly<{ rawFinalScore: number }>;
}>;

export type SearchOutcome = Readonly<{
  hits: readonly MemoryHit[];
  scoreKind: ScoreKind;
  /** Whether `score` values are on a comparable 0..1 scale and exposed on hits. */
  scoresComparable: boolean;
  warning: string | null;
  stageOutcomes?: StageOutcomes;
}>;

export type GetOutcome =
  | Readonly<{
      status: "ok";
      entry: MemoryEntry;
      text: string;
      from?: number;
      lines?: number;
      truncated?: boolean;
      nextFrom?: number;
    }>
  | Readonly<{ status: "not_found"; successor?: Readonly<{ ref: string }> | null }>;

export type ListOutcome = Readonly<{
  entries: readonly MemoryEntry[];
  total: number;
  offset: number;
  limit: number;
  truncated: boolean;
}>;

export type ForgetOutcome = Readonly<{
  removed: readonly string[];
  missing: readonly string[];
}>;

export class MemoryInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MemoryInputError";
  }
}

export class MemoryConsistencyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MemoryConsistencyError";
  }
}

const MAX_QUERY_CHARS = 1000;
export const MAX_LIST_LIMIT = 1000;
export const MAX_STORE_BATCH = 20;
export const MAX_FORGET_REFS = 100;
const EXPORT_PAGE_SIZE = 500;
/** Score kinds ReflectLog reports on a 0..1 relevance scale. RRF/BM25/concatenated ranks are not. */
const COMPARABLE_SCORE_KINDS: ReadonlySet<ScoreKind> = new Set(["reranker", "semantic"]);

function toEntry(scope: MemoryScope, record: ReflectLogRecord): MemoryEntry {
  return {
    ref: formatRef(scope, record.id),
    id: record.id,
    text: record.memory,
    revision: revisionOf(record.memory),
    createdAt: record.createdAt,
  };
}

function clampInt(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.trunc(value)));
}

function sliceLines(
  text: string,
  bounds: { from?: number | undefined; lines?: number | undefined },
): { text: string; from?: number; lines?: number; truncated?: boolean; nextFrom?: number } {
  if (bounds.from === undefined && bounds.lines === undefined) {
    return { text };
  }
  const all = text.split("\n");
  const from = Math.max(1, Math.trunc(bounds.from ?? 1));
  const start = Math.min(from - 1, all.length);
  const end =
    bounds.lines === undefined ? all.length : Math.min(all.length, start + Math.max(1, Math.trunc(bounds.lines)));
  const truncated = end < all.length;
  return {
    text: all.slice(start, end).join("\n"),
    from,
    lines: end - start,
    truncated,
    ...(truncated ? { nextFrom: end + 1 } : {}),
  };
}

export type FlushOutcome = Readonly<{ result: AddMemoriesResult; deduplicated: boolean }>;
export type FlushLedgerOptions = ConstructorParameters<typeof FlushLedger>[0];

/**
 * The single code path from every entry point (provider handle, agent tools,
 * flush, CLI) to ReflectLog. Callers resolve a {@link MemoryScope} first.
 *
 * One instance is one backend lifecycle: it owns its flush dedup state, which
 * ends with {@link MemoryService.dispose}. Dedup is process-local and bounded;
 * it is not restart-safe.
 */
export class MemoryService {
  readonly #client: ReflectLogClient;
  readonly #config: ReflectLogPluginConfig;
  readonly #flushes: FlushCoordinator<AddMemoriesResult>;
  #disposed = false;

  constructor(client: ReflectLogClient, config: ReflectLogPluginConfig, options: { flush?: FlushLedgerOptions } = {}) {
    this.#client = client;
    this.#config = config;
    this.#flushes = new FlushCoordinator(options.flush);
  }

  /**
   * End this backend lifecycle: pending flushes settle conservatively
   * (dispatched => `outcome_unknown`), dedup state is dropped, and later
   * mutations are rejected before any I/O. The caller closes the client.
   */
  dispose(): void {
    this.#disposed = true;
    this.#flushes.dispose();
  }

  #assertOpen(): void {
    if (this.#disposed) throw new FlushDisposedError();
  }

  async search(
    scope: MemoryScope,
    request: { query: string; maxResults?: number | undefined; minScore?: number | undefined; explain?: boolean | undefined },
    guard: EffectGuard,
  ): Promise<SearchOutcome> {
    const trimmed = request.query.trim();
    if (trimmed === "") {
      return { hits: [], scoreKind: "none", scoresComparable: false, warning: null };
    }
    const codePoints = Array.from(trimmed);
    const query = codePoints.slice(0, MAX_QUERY_CHARS).join("");
    const warnings: string[] = [];
    if (codePoints.length > MAX_QUERY_CHARS) {
      warnings.push(`query truncated to ${MAX_QUERY_CHARS} characters`);
    }
    const limit = clampInt(request.maxResults ?? this.#config.defaultMaxResults, 1, MAX_SEARCH_RESULTS);

    guard.assertCurrent();
    const result = await this.#client.searchMemories(
      { workspaceId: scope.workspaceId, query, limit, ...(request.explain === true ? { explain: true } : {}) },
      { signal: guard.signal, ...(request.explain === true ? { beforeDispatch: guard.assertCurrent } : {}) },
    );
    guard.assertCurrent();

    if (request.explain === true && (result.stageOutcomes === undefined ||
      result.results.some((record) => record.explanation === undefined))) {
      throw new ReflectLogError("malformed", "ReflectLog omitted requested search explanations");
    }
    const scoresComparable = COMPARABLE_SCORE_KINDS.has(result.scoreKind);
    let records = result.results;
    const minScore = request.minScore;
    if (minScore !== undefined && minScore > 0 && records.length > 0) {
      if (scoresComparable) {
        records = records.filter((record) => record.score >= minScore);
      } else {
        records = [];
        warnings.push(
          `ReflectLog returned '${result.scoreKind}' scores, which cannot be compared with minScore ${minScore}; no hits returned`,
        );
      }
    }
    return {
      hits: records.map((record) => ({
        ...toEntry(scope, record),
        score: scoresComparable ? record.score : null,
        ...(request.explain === true && record.explanation !== undefined
          ? { explanation: { ...record.explanation, rawFinalScore: record.score } } : {}),
      })),
      scoreKind: result.scoreKind,
      scoresComparable,
      warning: warnings.length > 0 ? warnings.join("; ") : null,
      ...(request.explain === true && result.stageOutcomes !== undefined ? { stageOutcomes: result.stageOutcomes } : {}),
    };
  }

  async get(
    scope: MemoryScope,
    reference: { ref: string; revision?: string | undefined; includeSuccessor?: boolean | undefined },
    bounds: { from?: number | undefined; lines?: number | undefined },
    guard: EffectGuard,
  ): Promise<GetOutcome> {
    const id = parseRef(scope, reference.ref);
    if (id === null) {
      return { status: "not_found" };
    }
    guard.assertCurrent();
    const result = await this.#client.getMemoryById(
      { workspaceId: scope.workspaceId, id, ...(reference.includeSuccessor === true ? { includeSuccessor: true } : {}) },
      { signal: guard.signal, ...(reference.includeSuccessor === true ? { beforeDispatch: guard.assertCurrent } : {}) },
    );
    guard.assertCurrent();
    const record = result.record;
    if (record === null) {
      if (reference.includeSuccessor === true) {
        if (result.successor === undefined) throw new ReflectLogError("malformed", "ReflectLog omitted requested successor hint");
        return { status: "not_found", successor: result.successor === null ? null : { ref: formatRef(scope, result.successor.id) } };
      }
      return { status: "not_found" };
    }
    const entry = toEntry(scope, record);
    if (reference.revision !== undefined && reference.revision !== entry.revision) {
      return { status: "not_found" };
    }
    return { status: "ok", entry, ...sliceLines(entry.text, bounds) };
  }

  async list(
    scope: MemoryScope,
    page: { limit: number; offset: number },
    guard: EffectGuard,
  ): Promise<ListOutcome> {
    const limit = clampInt(page.limit, 1, MAX_LIST_LIMIT);
    const offset = Math.max(0, Math.trunc(page.offset));
    guard.assertCurrent();
    const result = await this.#client.getAllMemories(
      { workspaceId: scope.workspaceId, limit, offset },
      { signal: guard.signal },
    );
    guard.assertCurrent();
    return {
      entries: result.records.map((record) => toEntry(scope, record)),
      total: result.total,
      offset: result.offset,
      limit: result.limit,
      truncated: result.truncated,
    };
  }

  /**
   * Every record in the scope, read page by page. This is a best-effort
   * paginated observation, not an atomic snapshot: it fails when the total
   * changes between pages or the pages do not add up, but a concurrent
   * replace that keeps the count can still go unnoticed.
   */
  async exportAll(scope: MemoryScope, guard: EffectGuard): Promise<readonly MemoryEntry[]> {
    const entries: MemoryEntry[] = [];
    let expectedTotal: number | null = null;
    for (let afterId = 0; ; ) {
      guard.assertCurrent();
      const page = await this.#client.getAllMemories(
        { workspaceId: scope.workspaceId, limit: EXPORT_PAGE_SIZE, afterId },
        { signal: guard.signal, beforeDispatch: guard.assertCurrent },
      );
      guard.assertCurrent();
      if (expectedTotal === null) {
        expectedTotal = page.total;
      } else if (page.total !== expectedTotal) {
        throw new MemoryConsistencyError(
          `workspace changed during export (total ${expectedTotal} -> ${page.total}); retry`,
        );
      }
      let lastId = afterId;
      for (const record of page.records) {
        if (record.id <= lastId) throw new MemoryConsistencyError("export records are not strictly increasing; retry");
        lastId = record.id;
      }
      if (page.nextAfterId === undefined || page.nextAfterId !== lastId ||
        page.records.length > page.limit || page.limit > EXPORT_PAGE_SIZE || page.offset !== 0 ||
        (page.truncated && page.records.length === 0)) {
        throw new MemoryConsistencyError("export returned an invalid cursor page; retry");
      }
      entries.push(...page.records.map((record) => toEntry(scope, record)));
      if (entries.length > expectedTotal || page.truncated !== (entries.length < expectedTotal)) {
        throw new MemoryConsistencyError(`export returned ${entries.length} records for a total of ${expectedTotal}; retry`);
      }
      if (!page.truncated) break;
      afterId = page.nextAfterId;
    }
    if (entries.length !== expectedTotal || new Set(entries.map((entry) => entry.id)).size !== entries.length) {
      throw new MemoryConsistencyError(
        `export returned ${entries.length} records for a total of ${expectedTotal}; retry`,
      );
    }
    return entries;
  }

  /** Validate, normalize and add memories. Throws {@link MemoryInputError} before any I/O. */
  async store(
    scope: MemoryScope,
    memories: readonly string[],
    options: { dryRun: boolean; allowReplace: boolean },
    guard: EffectGuard,
  ): Promise<AddMemoriesResult> {
    this.#assertOpen();
    const normalized = this.prepareMemories(memories);
    guard.assertCurrent();
    const result = await this.#client.addMemories(
      {
        workspaceId: scope.workspaceId,
        memories: normalized,
        dryRun: options.dryRun,
        allowReplace: options.allowReplace,
      },
      { signal: guard.signal },
    );
    if (options.dryRun) guard.assertCurrent();
    else this.#assertMutationCurrent(guard);
    return result;
  }

  /**
   * Add-only persistence of one pre-compaction flush, shared by every caller
   * presenting the same `flushId` and batch to this service. Validation and
   * authority run before the caller joins; the shared write is never replace
   * and is dispatched only while at least one waiter is current. Each waiter
   * re-checks its authority before it receives the result.
   */
  async flush(
    scope: MemoryScope,
    flushId: string,
    memories: readonly string[],
    guard: EffectGuard,
  ): Promise<FlushOutcome> {
    this.#assertOpen();
    const prepared = this.prepareMemories(memories);
    guard.assertCurrent();
    const outcome = await this.#flushes.run(
      FlushLedger.key(flushId, scope.workspaceId, prepared),
      guard,
      ({ signal, beforeDispatch }) => this.#client.addMemories(
        { workspaceId: scope.workspaceId, memories: prepared, dryRun: false, allowReplace: false },
        { signal, beforeDispatch },
      ),
    );
    return { result: outcome.value, deduplicated: outcome.deduplicated };
  }

  prepareMemories(memories: readonly string[]): readonly string[] {
    if (memories.length === 0) {
      throw new MemoryInputError("at least one memory is required");
    }
    if (memories.length > MAX_STORE_BATCH) {
      throw new MemoryInputError(`at most ${MAX_STORE_BATCH} memories can be stored at once`);
    }
    const normalized = memories.map(normalizeMemoryText);
    const empty = normalized.flatMap((text, index) => (text === "" ? [index + 1] : []));
    if (empty.length > 0) {
      throw new MemoryInputError(`memory ${empty.join(", ")} is empty`);
    }
    const suspicious = normalized.flatMap((text, index) =>
      looksLikePromptInjection(text) ? [index + 1] : [],
    );
    if (suspicious.length > 0) {
      throw new MemoryInputError(
        `memory ${suspicious.join(", ")} looks like an instruction to the model and was not stored`,
      );
    }
    return normalized;
  }

  /** Remove by reference. Malformed or foreign references are reported missing without I/O. */
  async forget(scope: MemoryScope, refs: readonly string[], guard: EffectGuard): Promise<ForgetOutcome> {
    this.#assertOpen();
    if (refs.length === 0) {
      throw new MemoryInputError("at least one reference is required");
    }
    if (refs.length > MAX_FORGET_REFS) {
      throw new MemoryInputError(`at most ${MAX_FORGET_REFS} references can be removed at once`);
    }
    const byId = new Map<number, string>();
    const missing: string[] = [];
    for (const ref of refs) {
      const id = parseRef(scope, ref);
      if (id === null) {
        missing.push(ref);
      } else if (!byId.has(id)) {
        byId.set(id, ref.trim());
      }
    }
    if (byId.size === 0) {
      return { removed: [], missing };
    }
    guard.assertCurrent();
    const result = await this.#client.deleteMemories(
      { workspaceId: scope.workspaceId, ids: [...byId.keys()] },
      { signal: guard.signal },
    );
    this.#assertMutationCurrent(guard);
    const removed = result.removed.flatMap((id) => {
      const ref = byId.get(id);
      return ref === undefined ? [] : [ref];
    });
    const removedIds = new Set(result.removed);
    for (const [id, ref] of byId) {
      if (!removedIds.has(id)) {
        missing.push(ref);
      }
    }
    return { removed, missing };
  }

  /** A revoked caller gets no result and must not mistake an acknowledged mutation for an unsent one. */
  #assertMutationCurrent(guard: EffectGuard): void {
    try {
      guard.assertCurrent();
    } catch (cause) {
      throw new ReflectLogError(
        "outcome_unknown",
        "the mutation may have been applied, but the caller is no longer current",
        { cause },
      );
    }
  }

  async health(scope: MemoryScope, guard: EffectGuard): Promise<void> {
    guard.assertCurrent();
    await this.#client.getAllMemories(
      { workspaceId: scope.workspaceId, limit: 0, offset: 0 },
      { signal: guard.signal },
    );
    guard.assertCurrent();
  }
}
