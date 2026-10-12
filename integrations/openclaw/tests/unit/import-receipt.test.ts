import { describe, expect, it } from "vitest";
import { buildReceipt, countBatchStates, totalsOf, type BatchRecord } from "../../src/import-receipt.js";

const batches: readonly BatchRecord[] = [
  { index: 1, size: 20, state: "acknowledged", storedCount: 18, skippedCount: 2 },
  { index: 2, size: 20, state: "outcome_unknown", errorCode: "outcome_unknown" },
  { index: 3, size: 1, state: "not_attempted" },
];

describe("import receipt model", () => {
  it("counts every batch state, including zero counts", () => {
    expect(countBatchStates(batches)).toEqual({ acknowledged: 1, outcome_unknown: 1, failed_before_dispatch: 0, not_attempted: 1 });
  });
  it("totals aggregate counts from acknowledged batches only", () => {
    expect(totalsOf(batches, 3)).toEqual({ stored: 18, skipped: 2, rejected: 3 });
    expect(totalsOf([{ index: 1, size: 5, state: "outcome_unknown", storedCount: 5 }], 0)).toEqual({ stored: 0, skipped: 0, rejected: 0 });
  });
  it("builds a labeled, reference-only receipt that never claims a snapshot or per-entry certainty", () => {
    const receipt = buildReceipt({
      outcome: "failed", workspaceId: "ws", startedAt: "a", finishedAt: "b",
      sources: [], accepted: [{ source: "MEMORY.md", line: 1, batch: 1 }], rejected: [{ source: "MEMORY.md", line: 3, reason: "too_long" }],
      batches, verification: { status: "not_run", paginated: true, globalSnapshot: false },
      failure: { stage: "batch", code: "outcome_unknown", batch: 2 },
    });
    expect(receipt).toMatchObject({
      format: "reflectlog.openclaw-import", formatVersion: 1, outcome: "failed",
      totals: { stored: 18, skipped: 2, rejected: 1 }, verification: { paginated: true, globalSnapshot: false },
      failure: { stage: "batch", batch: 2 },
    });
    expect(receipt.notes.join(" ")).toMatch(/not confirmed as persisted one by one/);
    expect(receipt.notes.join(" ")).toMatch(/not a global snapshot/);
    expect(Object.keys(buildReceipt({ ...receipt, failure: undefined, accepted: [], rejected: [] }))).not.toContain("failure");
  });
});
