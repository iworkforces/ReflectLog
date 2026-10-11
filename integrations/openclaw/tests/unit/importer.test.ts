import { link, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { archiveImportedFiles, collectMemoryFiles, splitMarkdownMemories } from "../../src/importer.js";

describe("Markdown splitting", () => {
  it("tracks heading paths and drops heading-only sections", () => {
    const markdown = "# Owner\n## Preferences\nTea\n# Empty\n## Also empty";
    expect(splitMarkdownMemories("MEMORY.md", markdown)).toEqual([
      { text: "Owner > Preferences: Tea (source: MEMORY.md)", source: "MEMORY.md", line: 3 },
    ]);
  });
  it("keeps sub-items and indented continuations with their top-level item", () => {
    const markdown = "- Tea\n  continued\n  - Green\n\n  more details\n- Coffee\n1. Water\n   - Still";
    expect(splitMarkdownMemories("memory/drinks.md", markdown).map((entry) => entry.text)).toEqual([
      "- Tea\n  continued\n  - Green\n\n  more details (source: memory/drinks.md)",
      "- Coffee (source: memory/drinks.md)", "1. Water\n   - Still (source: memory/drinks.md)",
    ]);
  });
  it("splits paragraphs without splitting their wrapped lines", () => {
    expect(splitMarkdownMemories("memory.md", "First\nwrapped\n\nSecond").map((entry) => entry.text)).toEqual([
      "First\nwrapped (source: memory.md)", "Second (source: memory.md)",
    ]);
  });
  it("keeps fenced code and blank lines inside a paragraph entry", () => {
    const body = "Example\n```ts\n# not a heading\n\n- not a list\n```";
    expect(splitMarkdownMemories("MEMORY.md", `${body}\n\nNext`)).toEqual([
      { text: `${body} (source: MEMORY.md)`, source: "MEMORY.md", line: 1 },
      { text: "Next (source: MEMORY.md)", source: "MEMORY.md", line: 8 },
    ]);
  });
  it("keeps indented tilde fences with a list item", () => {
    const body = "- Example\n  ~~~\n  a\n\n  b\n  ~~~";
    expect(splitMarkdownMemories("MEMORY.md", `${body}\n- Next`).map((entry) => entry.text)).toEqual([
      `${body} (source: MEMORY.md)`, "- Next (source: MEMORY.md)",
    ]);
  });
  it("skips YAML front matter while retaining original line numbers", () => {
    expect(splitMarkdownMemories("MEMORY.md", "---\nowner: Jane\n---\n\n# Facts\nTea")).toEqual([
      { text: "Facts: Tea (source: MEMORY.md)", source: "MEMORY.md", line: 6 },
    ]);
  });
  it("normalizes text and produces deterministic source-bearing entries", () => {
    const markdown = "\r\nCafe\u0301  \r\n\r\nTea\r\n";
    const expected = [
      { text: "Café (source: memory/facts.md)", source: "memory/facts.md", line: 2 },
      { text: "Tea (source: memory/facts.md)", source: "memory/facts.md", line: 4 },
    ];
    expect(splitMarkdownMemories("memory/facts.md", markdown)).toEqual(expected);
    expect(splitMarkdownMemories("memory/facts.md", markdown)).toEqual(expected);
  });
});

describe("import files", () => {
  let dir: string;
  beforeEach(async () => { dir = await mkdtemp(join(tmpdir(), "reflectlog-importer-")); });
  afterEach(async () => { await rm(dir, { recursive: true, force: true }); });

  it("collects root files and sorted direct Markdown children only", async () => {
    await mkdir(join(dir, "memory/nested"), { recursive: true });
    await Promise.all(["MEMORY.md", "memory/z.md", "memory/a.md", "memory/ignore.txt", "memory/nested/hidden.md"].map((path) => writeFile(join(dir, path), path)));
    const files = await collectMemoryFiles(dir);
    expect(files.map((file) => file.relativePath)).toEqual(["MEMORY.md", "memory/a.md", "memory/z.md"]);
    expect(files.map((file) => file.content)).toEqual(["MEMORY.md", "memory/a.md", "memory/z.md"]);
  });
  it("collects distinct lowercase and uppercase root files", async () => {
    await writeFile(join(dir, "MEMORY.md"), "upper");
    await writeFile(join(dir, "memory.md"), "lower");
    const files = await collectMemoryFiles(dir);
    // Case-insensitive filesystems expose one inode for these two names.
    expect(files.length === 1 ? files.map((file) => file.relativePath) : files.map((file) => file.relativePath)).toEqual(files.length === 1 ? ["MEMORY.md"] : ["MEMORY.md", "memory.md"]);
  });
  it("does not collect the same inode under two names", async () => {
    await writeFile(join(dir, "MEMORY.md"), "shared");
    await mkdir(join(dir, "memory"));
    await link(join(dir, "MEMORY.md"), join(dir, "memory/shared.md"));
    expect((await collectMemoryFiles(dir)).map((file) => file.relativePath)).toEqual(["MEMORY.md"]);
  });
  it("returns no files when workspace and memory directories are missing", async () => {
    expect(await collectMemoryFiles(join(dir, "absent"))).toEqual([]);
    expect(await collectMemoryFiles(dir)).toEqual([]);
  });
  it("moves files into a UTC timestamp archive and preserves content", async () => {
    await mkdir(join(dir, "memory"));
    await writeFile(join(dir, "MEMORY.md"), "owner content");
    await writeFile(join(dir, "memory/a.md"), "daily content");
    const destination = await archiveImportedFiles(dir, ["MEMORY.md", "memory/a.md"], new Date("2026-10-10T13:18:23.999Z"));
    expect(destination).toBe(join(dir, ".reflectlog-imported/20261010T131823Z"));
    expect(await readFile(join(destination, "MEMORY.md"), "utf8")).toBe("owner content");
    expect(await readFile(join(destination, "memory/a.md"), "utf8")).toBe("daily content");
    await expect(readFile(join(dir, "MEMORY.md"))).rejects.toMatchObject({ code: "ENOENT" });
    await expect(readFile(join(dir, "memory/a.md"))).rejects.toMatchObject({ code: "ENOENT" });
  });
  it("refuses timestamp collisions rather than overwriting an archive", async () => {
    const now = new Date("2026-10-10T13:18:23Z");
    await mkdir(join(dir, ".reflectlog-imported/20261010T131823Z"), { recursive: true });
    await writeFile(join(dir, "MEMORY.md"), "keep me");
    await expect(archiveImportedFiles(dir, ["MEMORY.md"], now)).rejects.toMatchObject({ code: "EEXIST" });
    expect(await readFile(join(dir, "MEMORY.md"), "utf8")).toBe("keep me");
  });
});
