import type { MemoryScope } from "./access.js";
import { FlushDisposedError } from "./flush-ledger.js";
import { ImportArchive, ImportArchiveError, type ArchiveFs } from "./import-archive.js";
import {
  buildReceipt, totalsOf,
  type AcceptedRef, type BatchRecord, type BatchState, type ImportFailure, type ImportTotals,
  type JournalEvent, type ReceiptSource, type RejectedRef, type SourceRef, type Verification,
} from "./import-receipt.js";
import {
  captureMemoryFiles, classifyEntries, parseCapturedSource,
  type CaptureFs, type CapturedSource, type ImportEntry, type RejectedEntry,
} from "./importer.js";
import { ReflectLogError } from "./reflectlog/types.js";
import {
  MAX_STORE_BATCH, MemoryConsistencyError, MemoryInputError, type EffectGuard, type MemoryService,
} from "./service.js";

const allowAll: EffectGuard = { assertCurrent: (): void => {} };

export type ImportArchiveOutcome = Readonly<{
  /**
   * `complete`: receipt written and marker removed. `incomplete`: an archive
   * directory exists but is not complete. `not_requested`: no `--archive`, or a
   * dry run. `not_created`: requested, but no directory was created (nothing
   * was captured, or the exclusive creation failed).
   */
  status: "complete" | "incomplete" | "not_requested" | "not_created";
  path: string | null;
  /** Set only when `receipt.json` was written. */
  receipt: string | null;
  errorCode?: string;
}>;

export type ImportRunResult = Readonly<{
  ok: boolean;
  dryRun: boolean;
  sources: readonly ReceiptSource[];
  accepted: readonly AcceptedRef[];
  rejected: readonly RejectedRef[];
  batches: readonly BatchRecord[];
  totals: ImportTotals;
  verification: Verification;
  archive: ImportArchiveOutcome;
  failure?: ImportFailure;
  /** The error behind `failure`, for hint mapping. Never print its message. */
  cause?: unknown;
}>;

export type ImportRunOptions = Readonly<{
  service: Pick<MemoryService, "store" | "exportAll">;
  scope: MemoryScope;
  workspaceDir: string;
  dryRun: boolean;
  archive: boolean;
  now: () => Date;
  guard?: EffectGuard;
  fs?: Readonly<{ capture?: CaptureFs; archive?: ArchiveFs }>;
}>;

type Failure = Readonly<{ failure: ImportFailure; cause: unknown }>;

function errno(error: unknown): string | undefined {
  return error instanceof Error && "code" in error && typeof error.code === "string" ? error.code : undefined;
}

/**
 * Classify a failed `store`. This is sound only because the client rethrows
 * EVERY failure after a mutation may have been sent as `outcome_unknown`
 * (`mcp-client.ts` `invoke`: `uncertain = mutation && sent`), keeping its own
 * codes only for failures proven to precede dispatch. So any other
 * `ReflectLogError`, a `MemoryInputError` or a `FlushDisposedError` means
 * nothing reached the server; every other error type is treated conservatively
 * as an unknown outcome.
 */
function classifyStoreFailure(error: unknown): { state: Exclude<BatchState, "acknowledged" | "not_attempted">; errorCode: string } {
  if (error instanceof ReflectLogError) {
    return { state: error.code === "outcome_unknown" ? "outcome_unknown" : "failed_before_dispatch", errorCode: error.code };
  }
  if (error instanceof MemoryInputError) return { state: "failed_before_dispatch", errorCode: "invalid_input" };
  if (error instanceof FlushDisposedError) return { state: "failed_before_dispatch", errorCode: "service_disposed" };
  return { state: "outcome_unknown", errorCode: "unexpected_error" };
}

function verificationFailureCode(error: unknown): string {
  if (error instanceof ReflectLogError) return error.code;
  if (error instanceof MemoryConsistencyError) return "verification_inconsistent";
  return "unexpected_error";
}

const NOT_RUN: Verification = { status: "not_run", paginated: true, globalSnapshot: false };

/**
 * Capture, classify, (optionally) archive and import the memory Markdown of
 * one workspace. Remote batches and archive files commit independently: nothing
 * here is atomic and nothing is rolled back. Failures after capture are
 * returned, never thrown, together with all progress made.
 */
export async function runImport(options: ImportRunOptions): Promise<ImportRunResult> {
  const { service, scope, dryRun } = options;
  const guard = options.guard ?? allowAll;
  const startedAt = options.now();
  const wantsArchive = options.archive && !dryRun;
  const notCreated = (errorCode?: string): ImportArchiveOutcome => ({
    status: wantsArchive ? "not_created" : "not_requested", path: null, receipt: null,
    ...(errorCode === undefined ? {} : { errorCode }),
  });
  const result = (parts: Omit<ImportRunResult, "ok" | "dryRun" | "totals">): ImportRunResult => ({
    ok: parts.failure === undefined,
    dryRun,
    totals: totalsOf(parts.batches, parts.rejected.length),
    ...parts,
  });

  let captured: CapturedSource[];
  try {
    captured = await captureMemoryFiles(options.workspaceDir, options.fs?.capture);
  } catch (cause) {
    return result({
      sources: [], accepted: [], rejected: [], batches: [], verification: NOT_RUN, archive: notCreated(),
      failure: { stage: "capture", code: errno(cause) ?? "capture_failed" }, cause,
    });
  }

  const sources: ReceiptSource[] = [];
  const acceptedEntries: ImportEntry[] = [];
  const rejectedEntries: RejectedEntry[] = [];
  for (const source of captured) {
    const entries = parseCapturedSource(source);
    const { accepted, rejected } = classifyEntries(entries);
    acceptedEntries.push(...accepted);
    rejectedEntries.push(...rejected);
    sources.push({
      relativePath: source.relativePath, size: source.size, sha256: source.sha256,
      archivedPath: `sources/${source.relativePath}`,
      parse: source.text === undefined ? "unparsed" : "parsed",
      ...(source.unparsedReason === undefined ? {} : { unparsedReason: source.unparsedReason }),
      entries: { total: entries.length, accepted: accepted.length, rejected: rejected.length },
    });
  }
  const accepted: AcceptedRef[] = acceptedEntries.map((entry, index) => ({
    source: entry.source, line: entry.line, batch: Math.floor(index / MAX_STORE_BATCH) + 1,
  }));
  const rejected: RejectedRef[] = rejectedEntries.map(({ entry, reason }) => ({ source: entry.source, line: entry.line, reason }));
  const batches: BatchRecord[] = [];
  for (let offset = 0; offset < acceptedEntries.length; offset += MAX_STORE_BATCH) {
    batches.push({
      index: batches.length + 1, size: Math.min(MAX_STORE_BATCH, acceptedEntries.length - offset), state: "not_attempted",
    });
  }

  let failure: Failure | undefined;
  let archiveErrorCode: string | undefined;
  let archivePath: string | null = null;
  let receiptPath: string | null = null;
  let archive: ImportArchive | undefined;
  let verification: Verification = NOT_RUN;
  const fail = (next: Failure): void => { failure ??= next; };
  // Closures assign `failure`, which control-flow narrowing cannot see; read it through here.
  const failed = (): Failure | undefined => failure;
  const archiveFailed = (error: unknown): ImportFailure => {
    const code = error instanceof ImportArchiveError ? error.code : "archive_failed";
    archiveErrorCode ??= code;
    return { stage: error instanceof ImportArchiveError && error.stage !== "journal" ? "archive" : "journal", code };
  };
  /** Append to the journal; false (and a recorded failure) when that fails. */
  const record = async (event: JournalEvent): Promise<boolean> => {
    if (archive === undefined) return true;
    try {
      await archive.append(event);
      return true;
    } catch (cause) {
      fail({ failure: archiveFailed(cause), cause });
      return false;
    }
  };

  if (wantsArchive && captured.length > 0) {
    try {
      archive = await ImportArchive.create({ workspaceDir: options.workspaceDir, now: startedAt, fs: options.fs?.archive });
      archivePath = archive.path;
      await archive.stageSources(captured);
      await archive.openJournal();
    } catch (cause) {
      if (cause instanceof ImportArchiveError && cause.archivePath !== undefined) archivePath = cause.archivePath;
      fail({ failure: archiveFailed(cause), cause });
    }
    if (failed() === undefined) {
      await record({ event: "run_started", at: startedAt.toISOString(), sources: captured.length, batches: batches.length });
    }
  }

  for (const batch of batches) {
    if (failed() !== undefined) break;
    const texts = acceptedEntries.slice((batch.index - 1) * MAX_STORE_BATCH, batch.index * MAX_STORE_BATCH).map((entry) => entry.text);
    if (!(await record({ event: "batch_intent", at: options.now().toISOString(), batch: batch.index, size: batch.size }))) break;
    try {
      const stored = await service.store(scope, texts, { dryRun, allowReplace: false }, guard);
      batches[batch.index - 1] = {
        index: batch.index, size: batch.size, state: "acknowledged",
        storedCount: stored.storedCount, skippedCount: stored.skippedCount,
      };
      await record({
        event: "batch_result", at: options.now().toISOString(), batch: batch.index, state: "acknowledged",
        storedCount: stored.storedCount, skippedCount: stored.skippedCount,
      });
    } catch (cause) {
      const outcome = classifyStoreFailure(cause);
      batches[batch.index - 1] = { index: batch.index, size: batch.size, state: outcome.state, errorCode: outcome.errorCode };
      fail({ failure: { stage: "batch", code: outcome.errorCode, batch: batch.index }, cause });
      await record({
        event: "batch_result", at: options.now().toISOString(), batch: batch.index, state: outcome.state, errorCode: outcome.errorCode,
      });
    }
  }

  if (failed() === undefined && !dryRun && acceptedEntries.length > 0) {
    try {
      const observed = new Set((await service.exportAll(scope, guard)).map((entry) => entry.text));
      const missing: SourceRef[] = acceptedEntries.flatMap((entry) => observed.has(entry.text) ? [] : [{ source: entry.source, line: entry.line }]);
      if (missing.length === 0) {
        verification = { status: "all_observed", paginated: true, globalSnapshot: false };
      } else {
        verification = { status: "missing", paginated: true, globalSnapshot: false, missing };
        fail({
          failure: { stage: "verification", code: "verification_missing" },
          cause: new MemoryConsistencyError("Import verification failed"),
        });
      }
    } catch (cause) {
      const code = verificationFailureCode(cause);
      verification = { status: "failed", paginated: true, globalSnapshot: false, errorCode: code };
      fail({ failure: { stage: "verification", code }, cause });
    }
  }

  if (archive !== undefined) {
    try {
      await archive.closeJournal();
    } catch (cause) {
      fail({ failure: archiveFailed(cause), cause });
    }
    const written = buildReceipt({
      outcome: failed() === undefined ? "succeeded" : "failed",
      workspaceId: scope.workspaceId, startedAt: startedAt.toISOString(), finishedAt: options.now().toISOString(),
      sources, accepted, rejected, batches, verification, failure: failed()?.failure,
    });
    try {
      await archive.writeReceipt(written);
      receiptPath = archive.receiptPath;
    } catch (cause) {
      fail({ failure: archiveFailed(cause), cause });
    }
    if (failed() === undefined) {
      try {
        await archive.publish();
      } catch (cause) {
        fail({ failure: archiveFailed(cause), cause });
      }
    }
  }

  const final = failed();
  let archiveOutcome: ImportArchiveOutcome;
  if (!wantsArchive) archiveOutcome = notCreated();
  else if (archivePath === null) archiveOutcome = notCreated(archiveErrorCode);
  else archiveOutcome = {
    status: final === undefined ? "complete" : "incomplete", path: archivePath, receipt: receiptPath,
    ...(archiveErrorCode === undefined ? {} : { errorCode: archiveErrorCode }),
  };
  return result({
    sources, accepted, rejected, batches, verification, archive: archiveOutcome,
    ...(final === undefined ? {} : { failure: final.failure, cause: final.cause }),
  });
}
