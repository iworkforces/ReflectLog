import { chmod, writeFile } from "node:fs/promises";
import type { Command } from "commander";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/plugin-entry";
import { resolveDefaultAgentId } from "openclaw/plugin-sdk/memory-host-core";
import { resolveAgentWorkspaceDir } from "openclaw/plugin-sdk/memory-core-host-engine-foundation";
import { z } from "zod";
import { conversationScope, ownerScope } from "./access.js";
import { PLUGIN_ID, type ReflectLogPluginConfig } from "./config.js";
import { archiveImportedFiles, collectMemoryFiles, splitMarkdownMemories } from "./importer.js";
import { looksLikePromptInjection } from "./policy.js";
import { ReflectLogError } from "./reflectlog/types.js";
import { MAX_STORE_BATCH, MemoryConsistencyError, MemoryInputError, type MemoryService } from "./service.js";

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
  if (error instanceof MemoryConsistencyError) return "Import/export verification failed; originals were not archived. Re-run list/export to check.";
  if (error instanceof MemoryInputError || error instanceof z.ZodError) return "Invalid command input; check arguments and numeric options.";
  return "ReflectLog command failed; check configuration, references, and file permissions.";
}

export function registerReflectLogCli(api: Pick<OpenClawPluginApi, "registerCli">, deps: { readonly config: ReflectLogPluginConfig; readonly getService: () => MemoryService }): void {
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
        { name: "transport/token", pass: deps.config.transport !== "http" || Boolean(process.env[deps.config.authTokenEnv]?.trim()), transport: deps.config.transport, tokenEnv: deps.config.authTokenEnv },
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
      const files = await collectMemoryFiles(workspaceDir);
      const perFile = files.map((file) => ({ source: file.relativePath, entries: splitMarkdownMemories(file.relativePath, file.content) }));
      const entries = perFile.flatMap((file) => file.entries);
      const rejected = entries.filter((entry) => entry.text.length > 30720 || looksLikePromptInjection(entry.text));
      const accepted = entries.filter((entry) => !rejected.includes(entry));
      let stored = 0;
      let skipped = rejected.length;
      const service = deps.getService();
      const selected = scope(options);
      for (let offset = 0; offset < accepted.length; offset += MAX_STORE_BATCH) {
        const result = await service.store(selected, accepted.slice(offset, offset + MAX_STORE_BATCH).map((entry) => entry.text), { dryRun: options.dryRun, allowReplace: false }, guard);
        stored += result.storedCount; skipped += result.skippedCount;
      }
      let archivedTo: string | undefined;
      if (!options.dryRun) {
        const texts = new Set((await service.exportAll(selected, guard)).map((entry) => entry.text));
        if (accepted.some((entry) => !texts.has(entry.text))) throw new MemoryConsistencyError("Import verification failed");
        if (options.archive && files.length) archivedTo = await archiveImportedFiles(workspaceDir, files.map((file) => file.relativePath), new Date());
      }
      output({ files: perFile.map((file) => ({ source: file.source, entries: file.entries.length })), stored, skipped, rejected: rejected.map((entry) => `${entry.source}:${entry.line}`), dryRun: options.dryRun, archivedTo }, options.json);
    });
  }, { descriptors: [{ name: "reflectlog", description: "Operate ReflectLog memory", hasSubcommands: true }] });
}
