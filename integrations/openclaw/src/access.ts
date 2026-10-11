import { createHash } from "node:crypto";

import { type ReflectLogPluginConfig, isValidAgentId } from "./config.js";
import type { MemoryAudience, MemoryCallerAuthority } from "./host-types.js";

export type PartitionKind = "owner-private" | "conversation";

/**
 * One ReflectLog workspace an authorized caller may touch.
 *
 * ReflectLog workspace ids are routing, not authorization: this plugin is the
 * only isolation boundary, so every entry point resolves a scope here.
 */
export type MemoryScope = Readonly<{
  agentId: string;
  partition: PartitionKind;
  workspaceId: string;
  /** Short workspace fingerprint embedded in references to bind them to this scope. */
  tag: string;
  sessionKey: string | null;
}>;

export class MemoryAccessDeniedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MemoryAccessDeniedError";
  }
}

const WORKSPACE_DERIVATION_VERSION = "memory-reflectlog/v1";

function sha256Hex(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

/**
 * Derive a ReflectLog workspace id (1-64 chars of `[a-z0-9_.-]`) from the
 * canonical (namespace, agent, partition, conversation) tuple.
 */
export function deriveWorkspaceId(
  config: ReflectLogPluginConfig,
  agentId: string,
  partition: PartitionKind,
  sessionKey: string | null,
): string {
  if (partition === "owner-private") {
    const override = config.workspaceOverrides[agentId];
    if (override !== undefined) {
      return override;
    }
  }
  const canonical = JSON.stringify([
    WORKSPACE_DERIVATION_VERSION,
    config.namespace,
    agentId,
    partition,
    partition === "conversation" ? sessionKey : null,
  ]);
  const prefix = partition === "owner-private" ? "p" : "c";
  return `${prefix}-${agentId.slice(0, 32)}-${sha256Hex(canonical).slice(0, 24)}`;
}

function buildScope(
  config: ReflectLogPluginConfig,
  agentId: string,
  partition: PartitionKind,
  sessionKey: string | null,
): MemoryScope {
  if (!isValidAgentId(agentId)) {
    throw new MemoryAccessDeniedError(`invalid agent id '${agentId}'`);
  }
  if (partition === "conversation" && (sessionKey === null || sessionKey.trim() === "")) {
    throw new MemoryAccessDeniedError("conversation memory requires a session key");
  }
  const workspaceId = deriveWorkspaceId(config, agentId, partition, sessionKey);
  return Object.freeze({
    agentId,
    partition,
    workspaceId,
    tag: sha256Hex(workspaceId).slice(0, 10),
    sessionKey: partition === "conversation" ? sessionKey : null,
  });
}

export function ownerScope(config: ReflectLogPluginConfig, agentId: string): MemoryScope {
  return buildScope(config, agentId, "owner-private", null);
}

/** Operator-only: conversation partitions are otherwise reached through a host-minted audience. */
export function conversationScope(
  config: ReflectLogPluginConfig,
  agentId: string,
  sessionKey: string,
): MemoryScope {
  return buildScope(config, agentId, "conversation", sessionKey);
}

/**
 * Map a host-minted audience to its partition. Conversation partitions key on
 * the durable root session key, so they survive session resets the same way
 * owner memory does; `sessionId` only drives the host's currency checks.
 */
export function resolveAudienceScope(
  config: ReflectLogPluginConfig,
  agentId: string,
  audience: MemoryAudience | undefined,
): MemoryScope {
  if (audience === undefined) {
    throw new MemoryAccessDeniedError("no memory audience was granted for this caller");
  }
  if (audience.agentId !== agentId) {
    throw new MemoryAccessDeniedError("memory audience belongs to another agent");
  }
  if (audience.kind === "owner-private") {
    return ownerScope(config, agentId);
  }
  return conversationScope(config, agentId, audience.sessionKey);
}

/** Authorize a native provider caller for search/get. Health does not need a grant. */
export function resolveProviderScope(
  config: ReflectLogPluginConfig,
  agentId: string,
  authority: MemoryCallerAuthority,
): MemoryScope {
  switch (authority.kind) {
    case "operator":
      return ownerScope(config, agentId);
    case "session":
      return resolveAudienceScope(config, agentId, authority.audience);
    case "host":
      throw new MemoryAccessDeniedError(
        `host operation '${authority.operation}' has no memory read grant`,
      );
  }
}

/** Authorize an agent tool invocation from its host-supplied context. */
export function resolveToolScope(
  config: ReflectLogPluginConfig,
  ctx: { readonly agentId?: string | undefined; readonly memoryAudience?: MemoryAudience | undefined },
): MemoryScope {
  if (ctx.agentId === undefined || ctx.agentId === "") {
    throw new MemoryAccessDeniedError("memory tools require an agent");
  }
  return resolveAudienceScope(config, ctx.agentId, ctx.memoryAudience);
}
