import type { EntryRejection } from "./importer.js";

/**
 * Typed model for the import receipt and its progress journal. Nothing here
 * carries memory text: entries are `source:line` references only.
 */

export const IMPORT_RECEIPT_FORMAT = "reflectlog.openclaw-import";
export const IMPORT_RECEIPT_VERSION = 1;

/**
 * - `acknowledged`: the backend confirmed the batch; counts are aggregate.
 * - `outcome_unknown`: the batch may or may not have been applied.
 * - `failed_before_dispatch`: proven not sent; nothing was applied.
 * - `not_attempted`: never dispatched because an earlier step stopped the run.
 */
export type BatchState = "acknowledged" | "outcome_unknown" | "failed_before_dispatch" | "not_attempted";

export const BATCH_STATES: readonly BatchState[] = [
  "acknowledged", "outcome_unknown", "failed_before_dispatch", "not_attempted",
];

export type BatchRecord = Readonly<{
  /** 1-based position in the plan. */
  index: number;
  size: number;
  state: BatchState;
  /** Aggregate server count for the whole batch; never attributable to one entry. */
  storedCount?: number;
  /** Aggregate server count for the whole batch; never attributable to one entry. */
  skippedCount?: number;
  errorCode?: string;
}>;

export type EntryCounts = Readonly<{ total: number; accepted: number; rejected: number }>;

export type ReceiptSource = Readonly<{
  relativePath: string;
  /** Bytes captured (and archived), which may differ from the file's size afterwards. */
  size: number;
  sha256: string;
  /** Path of the archived copy relative to the archive directory. */
  archivedPath: string;
  parse: "parsed" | "unparsed";
  unparsedReason?: string;
  entries: EntryCounts;
}>;

export type AcceptedRef = Readonly<{ source: string; line: number; batch: number }>;
export type RejectedRef = Readonly<{ source: string; line: number; reason: EntryRejection }>;
export type SourceRef = Readonly<{ source: string; line: number }>;

export type VerificationStatus = "not_run" | "all_observed" | "missing" | "failed";

export type Verification = Readonly<{
  status: VerificationStatus;
  /** The check reads the workspace page by page. */
  paginated: true;
  /** It is never a point-in-time snapshot of the workspace. */
  globalSnapshot: false;
  missing?: readonly SourceRef[];
  errorCode?: string;
}>;

export type ImportFailureStage = "capture" | "archive" | "batch" | "journal" | "verification";

export type ImportFailure = Readonly<{
  stage: ImportFailureStage;
  code: string;
  /** 1-based batch the failure concerns, when it concerns one. */
  batch?: number;
}>;

export type ImportTotals = Readonly<{
  /** Sum of `storedCount` over acknowledged batches. */
  stored: number;
  /** Sum of `skippedCount` over acknowledged batches (server-side dedupe). */
  skipped: number;
  /** Entries rejected locally before any submission. */
  rejected: number;
}>;

export type ImportReceipt = Readonly<{
  format: typeof IMPORT_RECEIPT_FORMAT;
  formatVersion: typeof IMPORT_RECEIPT_VERSION;
  /**
   * Whether capture, every batch and verification succeeded when this receipt
   * was written. It says nothing about archive completeness: the archive is
   * complete only if `receipt.json` exists and the `INCOMPLETE` marker is absent.
   */
  outcome: "succeeded" | "failed";
  workspaceId: string;
  startedAt: string;
  finishedAt: string;
  sources: readonly ReceiptSource[];
  entries: Readonly<{ accepted: readonly AcceptedRef[]; rejected: readonly RejectedRef[] }>;
  batches: readonly BatchRecord[];
  totals: ImportTotals;
  verification: Verification;
  failure?: ImportFailure;
  notes: readonly string[];
}>;

export const RECEIPT_NOTES: readonly string[] = [
  "outcome describes the run when this receipt was written; the archive itself is complete only if receipt.json exists and the INCOMPLETE marker is absent.",
  "Accepted entries were accepted for submission, not confirmed as persisted one by one.",
  "Totals are aggregate server counts for acknowledged batches; they do not establish which entry was stored or skipped.",
  "A batch in state outcome_unknown may have been applied; check list/export before retrying.",
  "Batches and archive files commit independently; this import is neither atomic nor rolled back on failure.",
  "Verification reads the workspace page by page and is not a global snapshot.",
  "A batch_intent journal line without a batch_result means the process stopped while that batch was in flight.",
  "Live source files are never modified; sources/ holds the bytes captured at import time.",
];

export type JournalEvent =
  | Readonly<{ event: "run_started"; at: string; sources: number; batches: number }>
  | Readonly<{ event: "batch_intent"; at: string; batch: number; size: number }>
  | Readonly<{
      event: "batch_result";
      at: string;
      batch: number;
      state: Exclude<BatchState, "not_attempted">;
      storedCount?: number;
      skippedCount?: number;
      errorCode?: string;
    }>;

export function countBatchStates(batches: readonly BatchRecord[]): Record<BatchState, number> {
  const counts: Record<BatchState, number> = {
    acknowledged: 0, outcome_unknown: 0, failed_before_dispatch: 0, not_attempted: 0,
  };
  for (const batch of batches) counts[batch.state] += 1;
  return counts;
}

export function totalsOf(batches: readonly BatchRecord[], rejected: number): ImportTotals {
  let stored = 0;
  let skipped = 0;
  for (const batch of batches) {
    if (batch.state !== "acknowledged") continue;
    stored += batch.storedCount ?? 0;
    skipped += batch.skippedCount ?? 0;
  }
  return { stored, skipped, rejected };
}

export function buildReceipt(input: Readonly<{
  outcome: "succeeded" | "failed";
  workspaceId: string;
  startedAt: string;
  finishedAt: string;
  sources: readonly ReceiptSource[];
  accepted: readonly AcceptedRef[];
  rejected: readonly RejectedRef[];
  batches: readonly BatchRecord[];
  verification: Verification;
  failure?: ImportFailure | undefined;
}>): ImportReceipt {
  return {
    format: IMPORT_RECEIPT_FORMAT,
    formatVersion: IMPORT_RECEIPT_VERSION,
    outcome: input.outcome,
    workspaceId: input.workspaceId,
    startedAt: input.startedAt,
    finishedAt: input.finishedAt,
    sources: input.sources,
    entries: { accepted: input.accepted, rejected: input.rejected },
    batches: input.batches,
    totals: totalsOf(input.batches, input.rejected.length),
    verification: input.verification,
    ...(input.failure === undefined ? {} : { failure: input.failure }),
    notes: RECEIPT_NOTES,
  };
}
