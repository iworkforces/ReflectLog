import { z } from "zod";
import { ReflectLogError } from "./types.js";

const id = z.int().positive();
const count = z.int().nonnegative();
const record = z.object({ id, memory: z.string(), created_at: z.string().nullable() });
const mapRecord = (value: z.output<typeof record>) => ({
  id: value.id, memory: value.memory, createdAt: value.created_at,
});
export const searchRecordsSchema = z.object({
  results: z.array(record.extend({ score: z.number() })),
  score_kind: z.enum(["reranker", "rrf", "semantic", "fulltext", "concatenated", "none"]),
  limit: count,
}).transform((value) => ({
  results: value.results.map((item) => ({ ...mapRecord(item), score: item.score })),
  scoreKind: value.score_kind, limit: value.limit,
}));
export const getRecordsSchema = z.object({ records: z.array(record), missing: z.array(id) })
  .transform((value) => ({ records: value.records.map(mapRecord), missing: value.missing }));
export const listRecordsSchema = z.object({
  records: z.array(record), total: count, offset: count, limit: count, truncated: z.boolean(),
}).transform((value) => ({ ...value, records: value.records.map(mapRecord) }));
export const removeRecordsSchema = z.object({ removed: z.array(id), missing: z.array(id) });
export const addSchema = z.object({
  stored_count: count, skipped_count: count, replaced_count: count, dry_run: z.boolean(),
  replacements: z.array(z.object({
    old_memory: z.string(), new_memory: z.string(), confidence: z.number(),
    reason: z.string().nullable().optional(),
  })),
}).transform((value) => ({
  storedCount: value.stored_count, skippedCount: value.skipped_count,
  replacedCount: value.replaced_count, dryRun: value.dry_run,
  replacements: value.replacements.map((item) => ({
    oldMemory: item.old_memory, newMemory: item.new_memory,
    confidence: item.confidence, reason: item.reason ?? null,
  })),
}));
const optionalEngine = z.string().nullable().catch(null);
export const healthCheckSchema = z.object({
  status: z.enum(["healthy", "degraded", "unhealthy"]),
  semantic_engine: optionalEngine, tantivy_engine: optionalEngine, reranker_engine: optionalEngine,
  pending_intent_count: count.nullable().catch(null),
}).transform((value) => ({
  status: value.status, semanticEngine: value.semantic_engine,
  fulltextEngine: value.tantivy_engine, rerankerEngine: value.reranker_engine,
  pendingIntentCount: value.pending_intent_count,
}));

/** Narrow envelope shared by SDK results and injected connections. */
export type ToolResponse = {
  readonly structuredContent?: unknown;
  readonly content?: readonly { readonly type: string; readonly text?: string | undefined }[] | undefined;
  readonly isError?: boolean | undefined;
};

export function parseToolResponse<T>(response: ToolResponse, schema: z.ZodType<T>): T {
  if (response.isError) {
    const text = response.content?.find((block) => block.type === "text")?.text ?? "Tool failed";
    throw new ReflectLogError(/Unauthorized/i.test(text) ? "unauthorized" : "tool_error", text,
      { cause: response });
  }
  try {
    const text = response.content?.find((block) => block.type === "text")?.text;
    const payload: unknown = response.structuredContent !== undefined
      ? response.structuredContent : JSON.parse(text ?? "");
    return schema.parse(payload);
  } catch (cause) {
    throw new ReflectLogError("malformed", "ReflectLog returned an invalid tool result", { cause });
  }
}
