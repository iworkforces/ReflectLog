import { createHash } from "node:crypto";

import type { MemoryScope } from "./access.js";

const REF_RE = /^([0-9a-f]{10}):([1-9][0-9]{0,15})$/;

/** Reference id for a ReflectLog record, bound to the scope that produced it. */
export function formatRef(scope: MemoryScope, recordId: number): string {
  return `${scope.tag}:${recordId}`;
}

/**
 * Resolve a reference id to a record id inside `scope`. Malformed references and
 * references minted for another workspace both return null, so a foreign
 * reference can never resolve the same numeric id in a different workspace.
 */
export function parseRef(scope: MemoryScope, ref: string): number | null {
  const match = REF_RE.exec(ref.trim());
  if (match === null || match[1] !== scope.tag) {
    return null;
  }
  const id = Number(match[2]);
  return Number.isSafeInteger(id) ? id : null;
}

/** Content revision; ReflectLog never edits a record in place, so this detects foreign or stale refs. */
export function revisionOf(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}
