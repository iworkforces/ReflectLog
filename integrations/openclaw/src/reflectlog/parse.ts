import { z } from "zod";
import { ReflectLogError } from "./types.js";

const id = z.int().positive();
const count = z.int().nonnegative();
const scoreKind = z.enum(["reranker", "rrf", "semantic", "fulltext", "concatenated", "none"]);
const contribution = z.object({ rank: id, score: z.number().finite() }).nullable();
const explanation = z.object({
  semantic: contribution, fulltext: contribution,
  pre_rerank_score: z.number().finite(), pre_rerank_score_kind: scoreKind, final_score_kind: scoreKind,
}).transform((value) => ({
  semantic: value.semantic, fulltext: value.fulltext,
  preRerankScore: value.pre_rerank_score, preRerankScoreKind: value.pre_rerank_score_kind,
  finalScoreKind: value.final_score_kind,
}));
const stage = z.enum(["disabled", "failed", "unavailable", "empty", "skipped_single_hit", "applied", "degraded"]);
const record = z.object({ id, memory: z.string(), created_at: z.string().nullable() });
const mapRecord = (value: z.output<typeof record>) => ({
  id: value.id, memory: value.memory, createdAt: value.created_at,
});
export const searchMemoriesSchema = z.object({
  results: z.array(record.extend({ score: z.number() })),
  score_kind: scoreKind,
  limit: count,
}).transform((value) => ({
  results: value.results.map((item) => ({ ...mapRecord(item), score: item.score })),
  scoreKind: value.score_kind, limit: value.limit,
}));
export const explainedSearchMemoriesSchema = z.object({
  results: z.array(record.extend({ score: z.number().finite(), explanation })),
  score_kind: scoreKind, limit: count,
  stage_outcomes: z.object({ semantic: stage, fulltext: stage, reranker: stage }),
}).transform((value) => ({
  results: value.results.map((item) => ({ ...mapRecord(item), score: item.score, explanation: item.explanation })),
  scoreKind: value.score_kind, limit: value.limit, stageOutcomes: value.stage_outcomes,
}));
export const getMemoryByIdSchema = z.object({ record: record.nullable() })
  .transform((value) => ({ record: value.record === null ? null : mapRecord(value.record) }));
export const successorMemoryByIdSchema = z.union([
  z.object({ record, successor: z.never().optional() })
    .transform((value) => ({ record: mapRecord(value.record) })),
  z.object({ record: z.null(), successor: z.object({ id }).nullable() }),
]);
export const getAllMemoriesSchema = z.object({
  records: z.array(record), total: count, offset: count, limit: count, truncated: z.boolean(),
}).transform((value) => ({ ...value, records: value.records.map(mapRecord) }));
export const cursorMemoriesSchema = z.object({
  records: z.array(record), total: count, offset: z.literal(0), limit: count,
  truncated: z.boolean(), next_after_id: count,
}).transform((value) => ({
  records: value.records.map(mapRecord), total: value.total, offset: value.offset,
  limit: value.limit, truncated: value.truncated, nextAfterId: value.next_after_id,
}));
export const deleteMemoriesSchema = z.object({ removed: z.array(id), missing: z.array(id) });
export const addMemoriesSchema = z.object({
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
