import { chmod, writeFile } from "node:fs/promises";
import type { Command } from "commander";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/plugin-entry";
import { resolveDefaultAgentId } from "openclaw/plugin-sdk/memory-host-core";
import { resolveAgentWorkspaceDir } from "openclaw/plugin-sdk/memory-core-host-engine-foundation";
import { z } from "zod";
import { conversationScope, ownerScope } from "./access.js";
import { PLUGIN_ID, type ReflectLogPluginConfig } from "./config.js";
import { ImportArchiveError, type ArchiveFs } from "./import-archive.js";
import { countBatchStates } from "./import-receipt.js";
import { runImport, type ImportRunResult } from "./import-run.js";
import type { CaptureFs } from "./importer.js";
import { ReflectLogError } from "./reflectlog/types.js";
import { MemoryConsistencyError, MemoryInputError, type MemoryService } from "./service.js";

const guard = { assertCurrent: (): void => {} };
const integer = z.coerce.number().int().safe();
const optionsSchema = z.object({
  agent: z.string(), conversation: z.string().min(1).optional(), json: z.boolean().default(false),
  limit: integer.positive().optional(), offset: integer.nonnegative().default(0),
  minScore: z.coerce.number().finite().min(0).max(1).optional(),
  from: integer.positive().optional(), lines: integer.positive().optional(),
  yes: z.boolean().default(false), out: z.string().min(1).optional(),
  workspaceDir: z.string().min(1).optional(), dryRun: z.boolean().default(false), archive: z.boolean().default(false),
});

function errorHint(error: unknown, config: ReflectLogPluginConfig): string {
  if (error instanceof ReflectLogError) {
    switch (error.code) {
      case "unauthorized": return `Authorization failed; set ${config.authTokenEnv}.`;
      case "unavailable": return "ReflectLog unavailable; start ReflectLog with --transport http.";
      case "incompatible": return "Incompatible server; upgrade ReflectLog.";
      case "outcome_unknown": return "Outcome unknown; re-run list/export to check before retrying.";
      case "timeout": return "ReflectLog request timed out.";
      case "tool_error": return "ReflectLog tool failed.";
      case "malformed": return "ReflectLog returned an invalid response.";
      case "aborted": return "ReflectLog operation aborted.";
    }
  }
  if (error instanceof MemoryConsistencyError) return "Import/export verification failed; live files were not modified. Check the receipt and re-run list/export before retrying.";
  if (error instanceof ImportArchiveError) return "Import archive could not be completed; live files were not modified and batches already sent are not rolled back. Inspect the incomplete archive and its receipt before retrying.";
  if (error instanceof MemoryInputError || error instanceof z.ZodError) return "Invalid command input; check arguments and numeric options.";
  return "ReflectLog command failed; check configuration, references, and file permissions.";
}

const IMPORT_NOTES = [
  "stored and skipped are aggregate server counts for acknowledged batches; they do not establish which entry was stored.",
  "accepted entries were accepted for submission; a batch in state outcome_unknown may have been applied.",
  "verification reads the workspace page by page and is not a global snapshot.",
] as const;

function importSummary(result: ImportRunResult, error: string | undefined) {
  return {
    ok: result.ok,
    files: result.sources.map((source) => ({
      source: source.relativePath, entries: source.entries.total, sha256: source.sha256, parse: source.parse,
      ...(source.unparsedReason === undefined ? {} : { unparsedReason: source.unparsedReason }),
    })),
    accepted: result.accepted.length,
    stored: result.totals.stored,
    skipped: result.totals.rejected + result.totals.skipped,
    totals: result.totals,
    rejected: result.rejected.map((entry) => `${entry.source}:${entry.line}`),
    dryRun: result.dryRun,
    archivedTo: result.archive.status === "complete" ? result.archive.path ?? undefined : undefined,
    archive: {
      path: result.archive.path, status: result.archive.status, receipt: result.archive.receipt,
      ...(result.archive.errorCode === undefined ? {} : { error: result.archive.errorCode }),
    },
    batches: { counts: countBatchStates(result.batches), items: result.batches },
    verification: result.verification,
    ...(result.failure === undefined ? {} : { failure: result.failure }),
    ...(error === undefined ? {} : { error }),
    notes: IMPORT_NOTES,
  };
}

export type ReflectLogCliDeps = {
  readonly config: ReflectLogPluginConfig;
  readonly getService: () => MemoryService;
  /** Import seams; defaults are the wall clock and the real filesystem. */
  readonly import?: { readonly now?: () => Date; readonly fs?: { readonly capture?: CaptureFs; readonly archive?: ArchiveFs } };
};

export function registerReflectLogCli(api: Pick<OpenClawPluginApi, "registerCli">, deps: ReflectLogCliDeps): void {
  api.registerCli(({ program, config: hostConfig }) => {
    const root = program.command("reflectlog").description("Operate ReflectLog memory");
    const command = (name: string): Command => root.command(name)
      .option("--agent <id>", "Agent partition", resolveDefaultAgentId(hostConfig))
      .option("--conversation <sessionKey>", "Conversation partition instead of owner")
      .option("--json", "Machine-readable output");
    const run = (cmd: Command, action: (args: readonly string[], options: z.output<typeof optionsSchema>) => Promise<void>): void => {
      cmd.action(async (...args: unknown[]) => {
        try {
          const options = optionsSchema.parse(cmd.opts());
          const positional = args.flatMap((arg) => typeof arg === "string" ? [arg] : Array.isArray(arg) ? z.array(z.string()).parse(arg) : []);
          await action(positional, options);
        } catch (error) { console.error(errorHint(error, deps.config)); process.exitCode = 1; }
      });
    };
    const scope = (options: z.output<typeof optionsSchema>) => options.conversation
      ? conversationScope(deps.config, options.agent, options.conversation) : ownerScope(deps.config, options.agent);
    const output = (value: unknown, json: boolean): void => { console.log(JSON.stringify(value, null, json ? undefined : 2)); };

    run(command("status"), async (_args, options) => {
      const selected = scope(options);
      const checks = [
        { name: "memory-slot", pass: hostConfig.plugins?.slots?.memory === PLUGIN_ID },
        { name: "memory-core-disabled", pass: hostConfig.plugins?.entries?.["memory-core"]?.enabled === false },
        { name: "transport/token", pass: Boolean(process.env[deps.config.authTokenEnv]?.trim()), transport: deps.config.transport, tokenEnv: deps.config.authTokenEnv },
        { name: "workspace", pass: true, workspaceId: selected.workspaceId },
      ];
      let health: { readonly name: string; readonly pass: boolean; readonly hint?: string };
      try {
        await deps.getService().health(selected, guard);
        health = { name: "connectivity", pass: true };
      } catch (error) { health = { name: "connectivity", pass: false, hint: errorHint(error, deps.config) }; }
      const all = [...checks, health];
      output({ workspaceId: selected.workspaceId, checks: all }, options.json);
      if (all.some((check) => !check.pass)) process.exitCode = 1;
    });
    run(command("search <query...>").option("--limit <n>").option("--min-score <x>"), async (args, options) => {
      output(await deps.getService().search(scope(options), { query: args.join(" "), maxResults: options.limit, minScore: options.minScore }, guard), options.json);
    });
    run(command("list").option("--limit <n>").option("--offset <n>"), async (_args, options) => {
      output(await deps.getService().list(scope(options), { limit: options.limit ?? 100, offset: options.offset }, guard), options.json);
    });
    run(command("get <ref>").option("--from <n>").option("--lines <n>"), async (args, options) => {
      output(await deps.getService().get(scope(options), { ref: args[0] ?? "" }, { from: options.from, lines: options.lines }, guard), options.json);
    });
    run(command("forget <ref...>").option("--yes", "Confirm deletion"), async (args, options) => {
      if (!options.yes) { console.error("Refusing forget without --yes."); process.exitCode = 1; return; }
      const result = await deps.getService().forget(scope(options), args, guard);
      output({ removed: result.removed, notFound: result.missing }, options.json);
    });
    run(command("export").option("--out <file>"), async (_args, options) => {
      const entries = await deps.getService().exportAll(scope(options), guard);
      const jsonl = entries.map((entry) => JSON.stringify(entry)).join("\n") + (entries.length ? "\n" : "");
      if (options.out) { await writeFile(options.out, jsonl, { mode: 0o600 }); await chmod(options.out, 0o600); }
      else process.stdout.write(jsonl);
    });
    run(command("import").option("--workspace-dir <dir>").option("--dry-run").option("--archive"), async (_args, options) => {
      if (options.archive && options.dryRun) { console.error("--archive cannot be combined with --dry-run."); process.exitCode = 1; return; }
      const workspaceDir = options.workspaceDir ?? resolveAgentWorkspaceDir(hostConfig, options.agent);
      const result = await runImport({
        service: deps.getService(), scope: scope(options), workspaceDir, dryRun: options.dryRun, archive: options.archive,
        now: deps.import?.now ?? (() => new Date()), guard, ...(deps.import?.fs ? { fs: deps.import.fs } : {}),
      });
      output(importSummary(result, result.failure === undefined ? undefined : errorHint(result.cause, deps.config)), options.json);
      if (!result.ok) process.exitCode = 1;
    });
  }, { descriptors: [{ name: "reflectlog", description: "Operate ReflectLog memory", hasSubcommands: true }] });
}
