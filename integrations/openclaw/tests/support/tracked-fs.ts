import { lstat, mkdir, open, unlink } from "node:fs/promises";
import type { ArchiveFs, ArchiveHandle } from "../../src/import-archive.js";

export type FsOperation = "lstat" | "mkdir" | "open" | "unlink" | "writeFile" | "sync" | "close";

export type FsEvent = Readonly<{
  op: FsOperation;
  path: string;
  /** True for `writeFile`/`sync`/`close` on a handle opened read-only (how directories are fsynced). */
  directory: boolean;
  /** 1-based count of earlier-or-equal events with the same `op` and `path`. */
  count: number;
}>;

/**
 * Real filesystem behind {@link ArchiveFs} with a per-operation fault hook and
 * handle accounting. `fail` returns an error to throw INSTEAD of performing the
 * operation (a `close` fault still closes the real handle first).
 */
export class TrackedArchiveFs implements ArchiveFs {
  readonly events: FsEvent[] = [];
  readonly opened: string[] = [];
  readonly closed: string[] = [];
  fail: (event: FsEvent) => Error | undefined = () => undefined;
  readonly #counts = new Map<string, number>();

  /** Fail the first event matching `op` and a path suffix. */
  failOn(op: FsOperation, pathSuffix: string, error: Error = new Error(`injected ${op} fault`)): void {
    let fired = false;
    this.fail = (event) => {
      if (fired || event.op !== op || !event.path.endsWith(pathSuffix)) return undefined;
      fired = true;
      return error;
    };
  }

  #check(op: FsOperation, path: string, directory = false): void {
    const key = `${op}\0${path}`;
    const count = (this.#counts.get(key) ?? 0) + 1;
    this.#counts.set(key, count);
    const event = { op, path, directory, count };
    this.events.push(event);
    const error = this.fail(event);
    if (error !== undefined) throw error;
  }

  async lstat(path: string): Promise<{ isDirectory(): boolean; isSymbolicLink(): boolean }> {
    this.#check("lstat", path);
    return lstat(path);
  }

  async mkdir(path: string, options: { readonly recursive?: boolean; readonly mode: number }): Promise<unknown> {
    this.#check("mkdir", path);
    return mkdir(path, options);
  }

  async unlink(path: string): Promise<void> {
    this.#check("unlink", path);
    await unlink(path);
  }

  async open(path: string, flags: string, mode?: number): Promise<ArchiveHandle> {
    this.#check("open", path);
    const handle = await open(path, flags, mode);
    const directory = flags === "r";
    this.opened.push(path);
    return {
      writeFile: async (data) => {
        this.#check("writeFile", path, directory);
        await handle.writeFile(data);
      },
      sync: async () => {
        this.#check("sync", path, directory);
        await handle.sync();
      },
      close: async () => {
        try {
          this.#check("close", path, directory);
        } finally {
          await handle.close();
          this.closed.push(path);
        }
      },
    };
  }

  /** Paths opened but not closed. */
  leaked(): string[] {
    const remaining = [...this.closed];
    return this.opened.filter((path) => {
      const index = remaining.indexOf(path);
      if (index < 0) return true;
      remaining.splice(index, 1);
      return false;
    });
  }
}
