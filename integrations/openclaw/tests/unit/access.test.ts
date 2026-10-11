import { describe, expect, it } from "vitest";

import {
  MemoryAccessDeniedError,
  conversationScope,
  ownerScope,
  resolveProviderScope,
  resolveToolScope,
} from "../../src/access.js";
import { parseConfig, type ReflectLogPluginConfig } from "../../src/config.js";
import { formatRef, parseRef } from "../../src/references.js";

function config(raw: unknown = {}): ReflectLogPluginConfig {
  const parsed = parseConfig(raw);
  if (!parsed.ok) {
    throw new Error(parsed.error);
  }
  return parsed.config;
}

const WORKSPACE_ID_RE = /^[a-z0-9_.-]{1,64}$/;

describe("workspace partitioning", () => {
  it("derives valid, distinct workspaces per agent, partition and conversation", () => {
    const cfg = config();
    const ids = [
      ownerScope(cfg, "main").workspaceId,
      ownerScope(cfg, "other").workspaceId,
      conversationScope(cfg, "main", "agent:main:discord:channel:1").workspaceId,
      conversationScope(cfg, "main", "agent:main:discord:channel:2").workspaceId,
    ];
    expect(new Set(ids).size).toBe(ids.length);
    for (const id of ids) {
      expect(id).toMatch(WORKSPACE_ID_RE);
    }
  });

  it("keeps workspace ids within 64 characters for the longest agent id", () => {
    const longAgent = `a${"b".repeat(63)}`;
    expect(ownerScope(config(), longAgent).workspaceId.length).toBeLessThanOrEqual(64);
  });

  it("separates deployments by namespace", () => {
    expect(ownerScope(config({ namespace: "prod" }), "main").workspaceId).not.toBe(
      ownerScope(config({ namespace: "dev" }), "main").workspaceId,
    );
  });

  it("applies owner overrides only to the owner partition", () => {
    const cfg = config({ workspaceOverrides: { main: "shared-notes" } });
    expect(ownerScope(cfg, "main").workspaceId).toBe("shared-notes");
    expect(conversationScope(cfg, "main", "k").workspaceId).not.toBe("shared-notes");
  });

  it("rejects overrides that collide or mimic derived ids", () => {
    expect(parseConfig({ workspaceOverrides: { a: "same", b: "same" } }).ok).toBe(false);
    expect(parseConfig({ workspaceOverrides: { a: `p-main-${"0".repeat(24)}` } }).ok).toBe(false);
    expect(parseConfig({ workspaceOverrides: { a: ".." } }).ok).toBe(false);
    expect(parseConfig({ workspaceOverrides: { a: "Upper" } }).ok).toBe(false);
  });
});

describe("caller authorization", () => {
  const cfg = config();
  const owner = { kind: "owner-private", agentId: "main" } as const;
  const conversation = {
    kind: "conversation",
    agentId: "main",
    sessionKey: "agent:main:telegram:group:9",
    sessionId: "s-1",
  } as const;

  it("routes owner and conversation audiences to separate partitions", () => {
    const ownerResolved = resolveToolScope(cfg, { agentId: "main", memoryAudience: owner });
    const conversationResolved = resolveToolScope(cfg, { agentId: "main", memoryAudience: conversation });
    expect(ownerResolved.partition).toBe("owner-private");
    expect(conversationResolved.partition).toBe("conversation");
    expect(ownerResolved.workspaceId).not.toBe(conversationResolved.workspaceId);
  });

  it("keeps a conversation partition across session resets", () => {
    const before = resolveToolScope(cfg, { agentId: "main", memoryAudience: conversation });
    const after = resolveToolScope(cfg, {
      agentId: "main",
      memoryAudience: { ...conversation, sessionId: "s-2" },
    });
    expect(after.workspaceId).toBe(before.workspaceId);
  });

  it("denies tools without an audience or with another agent's audience", () => {
    expect(() => resolveToolScope(cfg, { agentId: "main" })).toThrow(MemoryAccessDeniedError);
    expect(() => resolveToolScope(cfg, { memoryAudience: owner })).toThrow(MemoryAccessDeniedError);
    expect(() =>
      resolveToolScope(cfg, { agentId: "other", memoryAudience: owner }),
    ).toThrow(MemoryAccessDeniedError);
  });

  it("grants operators the owner partition and denies host operations", () => {
    expect(
      resolveProviderScope(cfg, "main", { kind: "operator", scopes: ["operator.read"] }).partition,
    ).toBe("owner-private");
    expect(() =>
      resolveProviderScope(cfg, "main", { kind: "host", operation: "memory-wiki.query" }),
    ).toThrow(MemoryAccessDeniedError);
  });

  it("denies session callers without an audience", () => {
    expect(() =>
      resolveProviderScope(cfg, "main", { kind: "session", sessionKey: "k", sandboxed: false }),
    ).toThrow(MemoryAccessDeniedError);
  });

  it("rejects malformed agent ids", () => {
    expect(() => ownerScope(cfg, "Main")).toThrow(MemoryAccessDeniedError);
    expect(() => ownerScope(cfg, "../x")).toThrow(MemoryAccessDeniedError);
  });
});

describe("references", () => {
  const cfg = config();

  it("round-trips inside the minting scope only", () => {
    const owner = ownerScope(cfg, "main");
    const conversation = conversationScope(cfg, "main", "k");
    const ref = formatRef(owner, 42);
    expect(parseRef(owner, ref)).toBe(42);
    expect(parseRef(conversation, ref)).toBeNull();
  });

  it("rejects malformed references", () => {
    const owner = ownerScope(cfg, "main");
    for (const ref of ["", "42", `${owner.tag}:0`, `${owner.tag}:-1`, `${owner.tag}:1.5`, "MEMORY.md"]) {
      expect(parseRef(owner, ref)).toBeNull();
    }
  });
});
