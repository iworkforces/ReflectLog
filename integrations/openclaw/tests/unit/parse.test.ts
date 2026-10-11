import { describe, expect, it } from "vitest";

import {
  addSchema,
  getRecordsSchema,
  healthCheckSchema,
  listRecordsSchema,
  parseToolResponse,
  removeRecordsSchema,
  searchRecordsSchema,
} from "../../src/reflectlog/parse.js";
import { ReflectLogError } from "../../src/reflectlog/types.js";

const record = { id: 7, memory: "Prefers tea", created_at: "2026-10-10 00:00:00" };

function caught(run: () => unknown): ReflectLogError {
  try {
    run();
  } catch (error) {
    if (error instanceof ReflectLogError) {
      return error;
    }
    throw error;
  }
  throw new Error("expected a ReflectLogError");
}

describe("ReflectLog tool results", () => {
  it("maps search_records to camelCase scored records", () => {
    const payload = { results: [{ ...record, score: 0.82 }], score_kind: "reranker", limit: 5 };
    expect(parseToolResponse({ structuredContent: payload }, searchRecordsSchema)).toEqual({
      results: [{ id: 7, memory: "Prefers tea", createdAt: "2026-10-10 00:00:00", score: 0.82 }],
      scoreKind: "reranker",
      limit: 5,
    });
  });

  it("maps get, list and remove results", () => {
    expect(
      parseToolResponse({ structuredContent: { records: [record], missing: [9] } }, getRecordsSchema),
    ).toEqual({ records: [{ id: 7, memory: "Prefers tea", createdAt: "2026-10-10 00:00:00" }], missing: [9] });
    expect(
      parseToolResponse(
        { structuredContent: { records: [{ ...record, created_at: null }], total: 3, offset: 1, limit: 1, truncated: true } },
        listRecordsSchema,
      ),
    ).toEqual({
      records: [{ id: 7, memory: "Prefers tea", createdAt: null }],
      total: 3,
      offset: 1,
      limit: 1,
      truncated: true,
    });
    expect(
      parseToolResponse({ structuredContent: { removed: [7], missing: [] } }, removeRecordsSchema),
    ).toEqual({ removed: [7], missing: [] });
  });

  it("maps add results including replacement details", () => {
    const payload = {
      stored_count: 1,
      skipped_count: 0,
      replaced_count: 1,
      dry_run: false,
      replacements: [{ old_memory: "Likes coffee", new_memory: "Likes tea", confidence: 0.9 }],
    };
    expect(parseToolResponse({ structuredContent: payload }, addSchema)).toEqual({
      storedCount: 1,
      skippedCount: 0,
      replacedCount: 1,
      dryRun: false,
      replacements: [{ oldMemory: "Likes coffee", newMemory: "Likes tea", confidence: 0.9, reason: null }],
    });
  });

  it("keeps only content-free health fields and tolerates missing engine details", () => {
    const payload = { status: "degraded", workspace_id: "w", semantic_engine: "ready", search_failures: {} };
    expect(parseToolResponse({ structuredContent: payload }, healthCheckSchema)).toEqual({
      status: "degraded",
      semanticEngine: "ready",
      fulltextEngine: null,
      rerankerEngine: null,
      pendingIntentCount: null,
    });
  });

  it("falls back to JSON text when structured content is absent", () => {
    const response = { content: [{ type: "text", text: JSON.stringify({ removed: [], missing: [1] }) }] };
    expect(parseToolResponse(response, removeRecordsSchema)).toEqual({ removed: [], missing: [1] });
  });

  it("prefers structured content over text", () => {
    const response = {
      structuredContent: { removed: [2], missing: [] },
      content: [{ type: "text", text: "not json" }],
    };
    expect(parseToolResponse(response, removeRecordsSchema)).toEqual({ removed: [2], missing: [] });
  });

  it("classifies tool errors and bearer rejections", () => {
    expect(
      caught(() => parseToolResponse({ isError: true, content: [{ type: "text", text: "StorageError: busy" }] }, removeRecordsSchema)).code,
    ).toBe("tool_error");
    expect(
      caught(() => parseToolResponse({ isError: true, content: [{ type: "text", text: "Unauthorized" }] }, removeRecordsSchema)).code,
    ).toBe("unauthorized");
  });

  it("reports unexpected shapes and unparsable text as malformed", () => {
    expect(caught(() => parseToolResponse({ structuredContent: { removed: "7" } }, removeRecordsSchema)).code).toBe(
      "malformed",
    );
    expect(caught(() => parseToolResponse({ content: [{ type: "text", text: "{" }] }, removeRecordsSchema)).code).toBe(
      "malformed",
    );
    expect(caught(() => parseToolResponse({}, removeRecordsSchema)).code).toBe("malformed");
    expect(
      caught(() =>
        parseToolResponse(
          { structuredContent: { results: [{ ...record, score: 1 }], score_kind: "bm25", limit: 5 } },
          searchRecordsSchema,
        ),
      ).code,
    ).toBe("malformed");
  });
});
