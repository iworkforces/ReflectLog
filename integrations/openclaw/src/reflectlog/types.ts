/** Provenance of ReflectLog's final search scores (`search_memories.score_kind`). */
export type ScoreKind = "reranker" | "rrf" | "semantic" | "fulltext" | "concatenated" | "none";

export type ReflectLogRecord = {
  readonly id: number;
  readonly memory: string;
  readonly createdAt: string | null;
};

export type ScoredRecord = ReflectLogRecord & { readonly score: number };

export type SearchMemoriesResult = {
  readonly results: readonly ScoredRecord[];
  readonly scoreKind: ScoreKind;
  readonly limit: number;
};

export type GetMemoryByIdResult = {
  readonly record: ReflectLogRecord | null;
};

export type GetAllMemoriesResult = {
  readonly records: readonly ReflectLogRecord[];
  readonly total: number;
  readonly offset: number;
  readonly limit: number;
  readonly truncated: boolean;
};

export type DeleteMemoriesResult = {
  readonly removed: readonly number[];
  readonly missing: readonly number[];
};

export type Replacement = {
  readonly oldMemory: string;
  readonly newMemory: string;
  readonly confidence: number;
  readonly reason: string | null;
};

export type AddMemoriesResult = {
  readonly storedCount: number;
  readonly skippedCount: number;
  readonly replacedCount: number;
  readonly replacements: readonly Replacement[];
  readonly dryRun: boolean;
};

export type CallOptions = { readonly signal?: AbortSignal | undefined };

/**
 * Typed view of the ReflectLog MCP tools this plugin depends on.
 *
 * Implementations own transport lifecycle. Every method rejects with
 * {@link ReflectLogError}; none resolves on failure.
 */
export interface ReflectLogClient {
  searchMemories(
    params: { workspaceId: string; query: string; limit: number },
    options?: CallOptions,
  ): Promise<SearchMemoriesResult>;
  getMemoryById(
    params: { workspaceId: string; id: number },
    options?: CallOptions,
  ): Promise<GetMemoryByIdResult>;
  getAllMemories(
    params: { workspaceId: string; limit: number; offset: number },
    options?: CallOptions,
  ): Promise<GetAllMemoriesResult>;
  /** Mutation: never retried after the request may have reached the server. */
  deleteMemories(
    params: { workspaceId: string; ids: readonly number[] },
    options?: CallOptions,
  ): Promise<DeleteMemoriesResult>;
  /** Mutation unless `dryRun`: never retried after the request may have reached the server. */
  addMemories(
    params: {
      workspaceId: string;
      memories: readonly string[];
      dryRun: boolean;
      allowReplace: boolean;
    },
    options?: CallOptions,
  ): Promise<AddMemoriesResult>;
  /** Closes the shared connection; a later call reconnects lazily. */
  close(): Promise<void>;
}

/**
 * - `unavailable`: no request reached the server (connect/transport failure before send).
 * - `timeout`: a read timed out.
 * - `outcome_unknown`: a mutation may have reached the server but no result arrived.
 * - `tool_error`: the server answered with `isError`.
 * - `unauthorized`: the bearer token was rejected.
 * - `incompatible`: required tools or parameters are missing on the server.
 * - `malformed`: the server answered with an unexpected shape.
 * - `aborted`: the caller's signal aborted the call.
 */
export type ReflectLogErrorCode =
  | "unavailable"
  | "timeout"
  | "outcome_unknown"
  | "tool_error"
  | "unauthorized"
  | "incompatible"
  | "malformed"
  | "aborted";

export class ReflectLogError extends Error {
  readonly code: ReflectLogErrorCode;

  constructor(code: ReflectLogErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "ReflectLogError";
    this.code = code;
  }
}

/** Tool names this plugin requires on the ReflectLog server. */
export const REQUIRED_REFLECTLOG_TOOLS = [
  "add_memories",
  "search_memories",
  "get_all_memories",
  "get_memory_by_id",
  "delete_memories",
] as const;
