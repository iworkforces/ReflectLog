import { lstat, mkdir, readFile, readdir, rename } from "node:fs/promises";
import { dirname, join } from "node:path";
import { normalizeMemoryText } from "./policy.js";

export type ImportEntry = { readonly text: string; readonly source: string; readonly line: number };
export type MemoryFile = { readonly relativePath: string; readonly content: string };

function missing(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
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
