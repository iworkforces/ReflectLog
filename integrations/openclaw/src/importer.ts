import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdir, open, readFile, readdir, rename } from "node:fs/promises";
import { dirname, join } from "node:path";
import { looksLikePromptInjection, normalizeMemoryText } from "./policy.js";

export type ImportEntry = { readonly text: string; readonly source: string; readonly line: number };
export type MemoryFile = { readonly relativePath: string; readonly content: string };

export type UnparsedReason = "invalid_utf8";

/**
 * One source file exactly as it was read: a single buffer from a single handle.
 * Digest, size, provenance and parsed text all derive from `bytes`; the live
 * path is never consulted again.
 */
export type CapturedSource = Readonly<{
  relativePath: string;
  bytes: Uint8Array;
  sha256: string;
  /** `bytes.length`: what was captured, not what the file may have grown to. */
  size: number;
  mtimeMs: number;
  dev: number;
  ino: number;
  /** Strict UTF-8 decoding of `bytes`; `undefined` when they are not valid UTF-8. */
  text: string | undefined;
  unparsedReason?: UnparsedReason;
}>;

export type CaptureStat = { isFile(): boolean; readonly mtimeMs: number; readonly dev: number; readonly ino: number };
export type CaptureHandle = {
  stat(): Promise<CaptureStat>;
  readFile(): Promise<Uint8Array>;
  close(): Promise<void>;
};
export type CaptureFs = {
  lstat(path: string): Promise<{ isDirectory(): boolean }>;
  readdir(path: string): Promise<readonly { readonly name: string; isFile(): boolean }[]>;
  open(path: string, flags: number): Promise<CaptureHandle>;
};

const defaultCaptureFs: CaptureFs = {
  lstat: (path) => lstat(path),
  readdir: (path) => readdir(path, { withFileTypes: true }),
  open: (path, flags) => open(path, flags),
};

/** Entries longer than this are rejected before any submission (matches the server's memory limit). */
export const MAX_IMPORT_ENTRY_CHARS = 30720;
export type EntryRejection = "too_long" | "prompt_injection";
export type RejectedEntry = Readonly<{ entry: ImportEntry; reason: EntryRejection }>;

function missing(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}

function symlinkRefused(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ELOOP";
}

/**
 * Candidate set and order: MEMORY.md, memory.md, then sorted direct
 * `memory/*.md` children. Each regular file is opened once and read through
 * that handle: `fstat` for identity and type, one full read, then close. The
 * final path component is opened with `O_NOFOLLOW`, so a symlink there is
 * skipped (ELOOP); intermediate components are resolved normally and `memory/`
 * itself is only used when `lstat` reports a real directory. A missing file is
 * skipped, as is anything that is not a regular file. `O_NONBLOCK` keeps a
 * FIFO from stalling the open before `fstat` can reject it.
 */
export async function captureMemoryFiles(workspaceDir: string, fs: CaptureFs = defaultCaptureFs): Promise<CapturedSource[]> {
  const paths = ["MEMORY.md", "memory.md"];
  try {
    if ((await fs.lstat(join(workspaceDir, "memory"))).isDirectory()) {
      const children = await fs.readdir(join(workspaceDir, "memory"));
      paths.push(...children.filter((child) => child.isFile() && child.name.endsWith(".md"))
        .map((child) => `memory/${child.name}`).sort());
    }
  } catch (error) { if (!missing(error)) throw error; }
  const flags = constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0) | (constants.O_NONBLOCK ?? 0);
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const identities = new Set<string>();
  const sources: CapturedSource[] = [];
  for (const relativePath of paths) {
    let handle: CaptureHandle;
    try {
      handle = await fs.open(join(workspaceDir, relativePath), flags);
    } catch (error) {
      if (missing(error) || symlinkRefused(error)) continue;
      throw error;
    }
    try {
      const stat = await handle.stat();
      const identity = `${stat.dev}:${stat.ino}`;
      if (!stat.isFile() || identities.has(identity)) continue;
      identities.add(identity);
      const bytes = await handle.readFile();
      let text: string | undefined;
      let unparsedReason: UnparsedReason | undefined;
      try {
        text = decoder.decode(bytes);
      } catch (error) {
        if (!(error instanceof TypeError)) throw error;
        unparsedReason = "invalid_utf8";
      }
      sources.push({
        relativePath, bytes, size: bytes.length, mtimeMs: stat.mtimeMs, dev: stat.dev, ino: stat.ino,
        sha256: createHash("sha256").update(bytes).digest("hex"), text,
        ...(unparsedReason === undefined ? {} : { unparsedReason }),
      });
    } finally {
      await handle.close();
    }
  }
  return sources;
}

/** Entries parsed from the captured text only; an unparsed source yields none. */
export function parseCapturedSource(source: CapturedSource): ImportEntry[] {
  return source.text === undefined ? [] : splitMarkdownMemories(source.relativePath, source.text);
}

/** Same predicates the CLI always applied; `too_long` wins when both hold. */
export function classifyEntries(entries: readonly ImportEntry[]): { accepted: ImportEntry[]; rejected: RejectedEntry[] } {
  const accepted: ImportEntry[] = [];
  const rejected: RejectedEntry[] = [];
  for (const entry of entries) {
    if (entry.text.length > MAX_IMPORT_ENTRY_CHARS) rejected.push({ entry, reason: "too_long" });
    else if (looksLikePromptInjection(entry.text)) rejected.push({ entry, reason: "prompt_injection" });
    else accepted.push(entry);
  }
  return { accepted, rejected };
}

export function splitMarkdownMemories(relativePath: string, content: string): ImportEntry[] {
  const lines = content.replace(/\r\n?/g, "\n").replace(/^\uFEFF/, "").split("\n");
  const entries: ImportEntry[] = [];
  const headings: string[] = [];
  let body: string[] = [];
  let start = 0;
  let listIndent: number | undefined;
  let fence: { marker: string; length: number } | undefined;
  const flush = (): void => {
    const text = normalizeMemoryText(body.join("\n"));
    if (text) entries.push({ text: normalizeMemoryText(`${headings.filter(Boolean).join(" > ")}${headings.some(Boolean) ? ": " : ""}${text} (source: ${relativePath})`), source: relativePath, line: start + 1 });
    body = [];
    listIndent = undefined;
  };
  let offset = 0;
  if (lines[0] === "---") {
    const end = lines.findIndex((line, index) => index > 0 && /^(---|\.\.\.)\s*$/.test(line));
    if (end < 0) return [];
    offset = end + 1;
  }
  for (let index = offset; index < lines.length; index++) {
    const line = lines[index] ?? "";
    const marker = /^\s*(`{3,}|~{3,})(.*)$/.exec(line);
    if (fence) {
      body.push(line);
      if (marker?.[1]?.[0] === fence.marker && marker[1].length >= fence.length && !marker[2]?.trim()) fence = undefined;
      continue;
    }
    const heading = /^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
    if (heading?.[1] && heading[2]) {
      flush();
      const level = heading[1].length;
      headings.length = level;
      headings[level - 1] = normalizeMemoryText(heading[2]);
      continue;
    }
    const item = /^( *)(?:[-+*]|\d+[.)])\s+/.exec(line);
    const indent = line.match(/^ */)?.[0].length ?? 0;
    if (!line.trim()) {
      if (listIndent !== undefined) {
        const next = lines.slice(index + 1).find((candidate) => candidate.trim());
        if (next && (next.match(/^ */)?.[0].length ?? 0) > listIndent) { body.push(line); continue; }
      }
      flush();
      continue;
    }
    if (item && (listIndent === undefined || (item[1]?.length ?? 0) <= listIndent)) {
      flush();
      listIndent = item[1]?.length ?? 0;
    } else if (listIndent !== undefined && indent <= listIndent) flush();
    if (!body.length) start = index;
    body.push(line);
    if (marker?.[1]) fence = { marker: marker[1][0] ?? "`", length: marker[1].length };
  }
  flush();
  return entries;
}

export async function collectMemoryFiles(workspaceDir: string): Promise<MemoryFile[]> {
  const paths = ["MEMORY.md", "memory.md"];
  try {
    if ((await lstat(join(workspaceDir, "memory"))).isDirectory()) {
      const children = await readdir(join(workspaceDir, "memory"), { withFileTypes: true });
      paths.push(...children.filter((child) => child.isFile() && child.name.endsWith(".md"))
        .map((child) => `memory/${child.name}`).sort());
    }
  } catch (error) { if (!missing(error)) throw error; }
  const identities = new Set<string>();
  const files: MemoryFile[] = [];
  for (const relativePath of paths) {
    try {
      const stat = await lstat(join(workspaceDir, relativePath));
      const identity = `${stat.dev}:${stat.ino}`;
      if (!stat.isFile() || identities.has(identity)) continue;
      identities.add(identity);
      files.push({ relativePath, content: await readFile(join(workspaceDir, relativePath), "utf8") });
    } catch (error) { if (!missing(error)) throw error; }
  }
  return files;
}

export async function archiveImportedFiles(workspaceDir: string, relativePaths: readonly string[], now: Date): Promise<string> {
  if (relativePaths.some((path) => !/^(?:MEMORY\.md|memory\.md|memory\/[^/\\]+\.md)$/.test(path) || path.includes("\0"))) {
    throw new Error("Invalid import path");
  }
  const stamp = now.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
  const parent = join(workspaceDir, ".reflectlog-imported");
  await mkdir(parent, { recursive: true });
  if (!(await lstat(parent)).isDirectory()) throw new Error("Invalid archive directory");
  const target = join(parent, stamp);
  await mkdir(target); // Exclusive creation: never overwrite an earlier archive.
  for (const relativePath of new Set(relativePaths)) {
    const destination = join(target, relativePath);
    await mkdir(dirname(destination), { recursive: true });
    await rename(join(workspaceDir, relativePath), destination);
  }
  return target;
}
