import { lstat, mkdir, open, unlink } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { ImportReceipt, JournalEvent } from "./import-receipt.js";
import type { CapturedSource } from "./importer.js";

/**
 * Exclusive, copy-based archive of the sources an import captured.
 *
 * Layout under `<workspaceDir>/.reflectlog-imported/<stamp>/`:
 *
 *   INCOMPLETE      marker, created first and removed last
 *   sources/<rel>   the captured bytes of every source, one file each
 *   progress.jsonl  one fsynced line per batch event
 *   receipt.json    the run's outcome, written for success and failure
 *
 * An archive is complete only if `receipt.json` exists AND `INCOMPLETE` is
 * absent. Nothing is renamed into place, nothing is overwritten, and a failed
 * archive is left behind as an inspectable incomplete bundle. Live source
 * files are never touched. The parent directory is checked with `lstat`, which
 * does not follow a final symlink; this guards the parent's last component
 * only, not every ancestor.
 */

export const ARCHIVE_DIR_NAME = ".reflectlog-imported";
export const INCOMPLETE_MARKER = "INCOMPLETE";
export const RECEIPT_FILE = "receipt.json";
export const JOURNAL_FILE = "progress.jsonl";
export const SOURCES_DIR = "sources";

const DIR_MODE = 0o700;
const FILE_MODE = 0o600;
const INCOMPLETE_TEXT = "This archive is incomplete. It is complete only when receipt.json exists and this file is absent.\n";
const SAFE_SOURCE_PATH = /^(?:MEMORY\.md|memory\.md|memory\/[^/\\\0]+\.md)$/;
/** Directory fsync is unsupported on some platforms; these codes mean "cannot", not "failed". */
const DIRECTORY_SYNC_UNSUPPORTED: ReadonlySet<string> = new Set(["EISDIR", "EPERM", "EINVAL"]);

export type ArchiveHandle = {
  writeFile(data: string | Uint8Array): Promise<void>;
  sync(): Promise<void>;
  close(): Promise<void>;
};

/** The narrow filesystem surface the archive writer needs; tests inject faults here. */
export type ArchiveFs = {
  lstat(path: string): Promise<{ isDirectory(): boolean; isSymbolicLink(): boolean }>;
  mkdir(path: string, options: { readonly recursive?: boolean; readonly mode: number }): Promise<unknown>;
  open(path: string, flags: string, mode?: number): Promise<ArchiveHandle>;
  unlink(path: string): Promise<void>;
};

const defaultArchiveFs: ArchiveFs = {
  lstat: (path) => lstat(path),
  mkdir: (path, options) => mkdir(path, options),
  open: (path, flags, mode) => open(path, flags, mode),
  unlink: (path) => unlink(path),
};

export type ArchiveStage = "parent" | "exists" | "stamp" | "marker" | "source" | "journal" | "receipt" | "publish";

export class ImportArchiveError extends Error {
  readonly stage: ArchiveStage;
  readonly code: string;
  /** The stamp directory, once it exists; it may then hold a partial, incomplete archive. */
  archivePath?: string;

  constructor(stage: ArchiveStage, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "ImportArchiveError";
    this.stage = stage;
    this.code = `archive_${stage}`;
  }
}

function errno(error: unknown): string | undefined {
  return error instanceof Error && "code" in error && typeof error.code === "string" ? error.code : undefined;
}

async function within<T>(stage: ArchiveStage, message: string, operation: () => Promise<T>): Promise<T> {
  try {
    return await operation();
  } catch (cause) {
    if (cause instanceof ImportArchiveError) throw cause;
    throw new ImportArchiveError(stage, message, { cause });
  }
}

/** `YYYYMMDDTHHMMSSZ` in UTC. */
export function archiveStamp(now: Date): string {
  return now.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

export class ImportArchive {
  readonly path: string;
  readonly #fs: ArchiveFs;
  #journal: ArchiveHandle | undefined;
  #journalOpened = false;

  private constructor(path: string, fs: ArchiveFs) {
    this.path = path;
    this.#fs = fs;
  }

  get receiptPath(): string {
    return join(this.path, RECEIPT_FILE);
  }

  /**
   * Create the stamp directory exclusively and its `INCOMPLETE` marker. An
   * existing stamp (EEXIST) is a typed `exists` error raised before anything
   * is written.
   */
  static async create(options: { workspaceDir: string; now: Date; fs?: ArchiveFs | undefined }): Promise<ImportArchive> {
    const fs = options.fs ?? defaultArchiveFs;
    const parent = join(options.workspaceDir, ARCHIVE_DIR_NAME);
    await within("parent", "cannot prepare the archive directory", async () => {
      await fs.mkdir(parent, { recursive: true, mode: DIR_MODE });
      const info = await fs.lstat(parent);
      if (info.isSymbolicLink() || !info.isDirectory()) {
        throw new ImportArchiveError("parent", "the archive directory is not a real directory");
      }
    });
    const path = join(parent, archiveStamp(options.now));
    try {
      await fs.mkdir(path, { mode: DIR_MODE });
    } catch (cause) {
      throw errno(cause) === "EEXIST"
        ? new ImportArchiveError("exists", "an archive with this timestamp already exists", { cause })
        : new ImportArchiveError("stamp", "cannot create the archive directory", { cause });
    }
    const archive = new ImportArchive(path, fs);
    try {
      await within("marker", "cannot create the INCOMPLETE marker", async () => {
        await archive.#writeExclusive(
          join(path, INCOMPLETE_MARKER),
          INCOMPLETE_TEXT,
        );
        await archive.#syncDirectory(path);
        await archive.#syncDirectory(parent);
      });
    } catch (error) {
      if (error instanceof ImportArchiveError) error.archivePath = path;
      throw error;
    }
    return archive;
  }

  /** Write the captured bytes of every source under `sources/`, each fsynced. */
  async stageSources(sources: readonly CapturedSource[]): Promise<void> {
    await within("source", "cannot write a captured source to the archive", async () => {
      const unsafe = sources.find((source) => !SAFE_SOURCE_PATH.test(source.relativePath));
      if (unsafe !== undefined) throw new ImportArchiveError("source", "unsafe source path");
      const root = join(this.path, SOURCES_DIR);
      await this.#fs.mkdir(root, { mode: DIR_MODE });
      const directories = new Set<string>([root]);
      for (const source of sources) {
        const destination = join(root, source.relativePath);
        await this.#fs.mkdir(dirname(destination), { recursive: true, mode: DIR_MODE });
        await this.#writeExclusive(destination, source.bytes);
        directories.add(dirname(destination));
      }
      for (const directory of directories) await this.#syncDirectory(directory);
      await this.#syncDirectory(this.path);
    });
  }

  /** Create `progress.jsonl` exclusively. Close it with {@link closeJournal}. */
  async openJournal(): Promise<void> {
    await within("journal", "cannot create the progress journal", async () => {
      if (this.#journalOpened) throw new ImportArchiveError("journal", "the progress journal was already opened");
      this.#journalOpened = true;
      this.#journal = await this.#fs.open(join(this.path, JOURNAL_FILE), "wx", FILE_MODE);
    });
  }

  /** Append one JSON line and fsync it before returning. */
  async append(event: JournalEvent): Promise<void> {
    await within("journal", "cannot append to the progress journal", async () => {
      if (this.#journal === undefined) throw new ImportArchiveError("journal", "the progress journal is not open");
      await this.#journal.writeFile(`${JSON.stringify(event)}\n`);
      await this.#journal.sync();
    });
  }

  /** Idempotent. */
  async closeJournal(): Promise<void> {
    const journal = this.#journal;
    this.#journal = undefined;
    if (journal === undefined) return;
    await within("journal", "cannot close the progress journal", () => journal.close());
  }

  async writeReceipt(receipt: ImportReceipt): Promise<void> {
    await within("receipt", "cannot write the receipt", async () => {
      await this.#writeExclusive(this.receiptPath, `${JSON.stringify(receipt, null, 2)}\n`);
      await this.#syncDirectory(this.path);
    });
  }

  /** Remove the marker. Call only after every file and the receipt are durable. */
  async publish(): Promise<void> {
    await within("publish", "cannot publish the archive", async () => {
      await this.#syncDirectory(this.path);
      await this.#fs.unlink(join(this.path, INCOMPLETE_MARKER));
      try {
        await this.#syncDirectory(this.path);
      } catch (cause) {
        try {
          await this.#writeExclusive(join(this.path, INCOMPLETE_MARKER), INCOMPLETE_TEXT);
          await this.#syncDirectory(this.path);
        } catch (restoreCause) {
          throw new ImportArchiveError("publish", "cannot publish the archive; the INCOMPLETE marker could not be restored", {
            cause: new AggregateError([cause, restoreCause], "archive publication and marker restoration failed"),
          });
        }
        throw new ImportArchiveError("publish", "cannot publish the archive", { cause });
      }
    });
  }

  async #writeExclusive(path: string, data: string | Uint8Array): Promise<void> {
    const handle = await this.#fs.open(path, "wx", FILE_MODE);
    try {
      await handle.writeFile(data);
      await handle.sync();
    } finally {
      await handle.close();
    }
  }

  async #syncDirectory(path: string): Promise<void> {
    let handle: ArchiveHandle;
    try {
      handle = await this.#fs.open(path, "r");
    } catch (error) {
      if (DIRECTORY_SYNC_UNSUPPORTED.has(errno(error) ?? "")) return;
      throw error;
    }
    try {
      await handle.sync();
    } catch (error) {
      if (!DIRECTORY_SYNC_UNSUPPORTED.has(errno(error) ?? "")) throw error;
    } finally {
      await handle.close();
    }
  }
}
