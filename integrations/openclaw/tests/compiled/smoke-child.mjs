// Compiled-package smoke, child side. Plain Node ESM: no TypeScript loader, no vitest, no module mocks, no source alias.
// Run through smoke.test.mjs (cwd = package dir). Prints exactly one JSON report line on stdout and exits on its own.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { installHygiene } from "./support/hygiene.mjs";

const PACKAGE_NAME = "@iworkforces/openclaw-memory-reflectlog";
const PLUGIN_ID = "memory-reflectlog";
const TOKEN_ENV = "REFLECTLOG_MCP_AUTH_TOKEN";
const TOOLS = ["memory_search", "memory_get", "memory_store", "memory_preview", "memory_forget", "memory_list"];
const packageDir = fileURLToPath(new URL("../../", import.meta.url));
const checks = [];
const check = (name) => { checks.push(name); };
/** The host wraps registrations in ownership Proxies; plain data is what the assertions compare. */
const plain = (value) => JSON.parse(JSON.stringify(value));

// Guards first: nothing below (package, host chunk, Commander) may touch the network or the token.
const hygiene = installHygiene({ tokenEnv: TOKEN_ENV });

async function main() {
  const dist = join(packageDir, "dist", "index.js");
  if (!existsSync(dist)) throw new Error(`${dist} is missing: run pnpm build first`);
  const manifest = JSON.parse(readFileSync(join(packageDir, "openclaw.plugin.json"), "utf8"));
  const packageJson = JSON.parse(readFileSync(join(packageDir, "package.json"), "utf8"));
  assert.ok(!Object.hasOwn(process.env, TOKEN_ENV), "token env must be absent"); // never echo the value

  // 1. The package, by name, through the exports map ("." -> ./dist/index.js).
  const { default: entry, ...otherExports } = await import(PACKAGE_NAME);
  assert.deepEqual(Object.keys(otherExports), [], "the entry module only has a default export");
  assert.deepEqual(Object.keys(entry).sort(), ["configSchema", "description", "id", "name", "register"]);
  assert.equal(entry.id, PLUGIN_ID);
  assert.equal(entry.name, "ReflectLog Memory");
  assert.equal(typeof entry.register, "function");
  assert.equal(manifest.id, PLUGIN_ID);
  assert.equal(manifest.kind, "memory");
  assert.deepEqual(entry.configSchema.jsonSchema, manifest.configSchema, "entry config schema == manifest config schema");
  assert.deepEqual(packageJson.openclaw.extensions, ["./dist/index.js"]);
  const { MEMORY_TOOL_NAMES } = await import(pathToFileURL(join(packageDir, "dist", "tools.js")).href);
  assert.deepEqual([...MEMORY_TOOL_NAMES], TOOLS);
  assert.deepEqual([...manifest.contracts.tools].sort(), [...TOOLS].sort());
  check("entry-by-package-name");

  // Host seam: the real registry through the adapter (fail-closed; see support/openclaw-host.mjs).
  const { createLoaderRecord, openHostRegistry, registerLikeLoader } = await import("./support/openclaw-host.mjs");
  const newHost = async () => {
    const logs = [];
    const log = (level) => (...args) => { logs.push({ level, message: args.map(String).join(" ") }); };
    const logger = { info: log("info"), warn: log("warn"), error: log("error"), debug: log("debug") };
    // The plugin never touches api.runtime; make any access fail the gate rather than hand it a stub.
    const runtime = new Proxy({}, { get(_target, name) { throw new Error(`plugin touched host runtime.${String(name)}`); } });
    return { logs, host: await openHostRegistry({ logger, runtime }) };
  };
  const hostConfig = { plugins: { slots: { memory: PLUGIN_ID }, entries: { "memory-core": { enabled: false } } } };
  const pluginConfig = { url: "http://reflectlog.invalid:9103/mcp", namespace: "smoke" };
  const record = (memorySlotSelected = true) => createLoaderRecord({
    manifest, packageJson, rootDir: packageDir, source: dist, memorySlotSelected });
  // Host verdict from BOTH the entry schema and the manifest copy (what the real loader validates); they must agree.
  const { buildJsonPluginConfigSchema } = await import("openclaw/plugin-sdk/plugin-entry");
  const manifestSchema = buildJsonPluginConfigSchema(manifest.configSchema);
  const hostVerdict = (config) => {
    const entryVerdict = entry.configSchema.safeParse(config);
    const manifestVerdict = manifestSchema.safeParse(config);
    assert.equal(entryVerdict.success, manifestVerdict.success, "entry and manifest schema verdicts agree");
    return manifestVerdict;
  };
  assert.equal(hostVerdict(pluginConfig).success, true);

  // 2. Valid registration against a real registry, read back from the registry.
  const { host, logs } = await newHost();
  hygiene.forbidSpawning();
  const pluginRecord = record();
  registerLikeLoader({ host, entry, record: pluginRecord, config: hostConfig, pluginConfig });
  const { registry } = host;
  assert.deepEqual(registry.diagnostics, [], "registry reported no registration diagnostics");
  assert.deepEqual(logs.filter(({ level }) => level === "warn" || level === "error"), [], "no warnings with the slot configured");
  assert.equal(registry.memoryCapabilities.length, 1);
  const [{ pluginId, capability, memorySlotSelected }] = registry.memoryCapabilities;
  assert.equal(pluginId, PLUGIN_ID);
  assert.equal(memorySlotSelected, true);
  assert.deepEqual(plain(capability.recallToolNames), ["memory_search", "memory_get"]);
  assert.equal(capability.deterministicRecallToolName, "memory_search");
  assert.equal(capability.supportsPrivateTranscriptRecall, false);
  for (const member of ["promptBuilder", "providerFlushPlanResolver"]) assert.equal(typeof capability[member], "function", member);
  for (const member of ["open", "closeAllMemorySearchManagers"]) assert.equal(typeof capability.providerRuntime[member], "function", member);
  assert.equal(registry.tools.length, TOOLS.length);
  assert.deepEqual(plain(registry.tools.flatMap((tool) => tool.names)), TOOLS);
  assert.deepEqual(pluginRecord.toolNames, TOOLS);
  assert.equal(registry.services.length, 1);
  assert.equal(registry.services[0].id, PLUGIN_ID);
  assert.deepEqual(pluginRecord.services, [PLUGIN_ID]);
  assert.equal(typeof registry.services[0].service.stop, "function");
  assert.equal(registry.cliRegistrars.length, 1);
  assert.deepEqual(plain(registry.cliRegistrars[0].commands), ["reflectlog"]);
  assert.deepEqual(plain(registry.cliRegistrars[0].descriptors),
    [{ name: "reflectlog", description: "Operate ReflectLog memory", hasSubcommands: true }]);
  assert.deepEqual(pluginRecord.cliCommands, ["reflectlog"]);
  assert.equal(hygiene.tokenReads.count, 0, "registration is lazy: nothing read the token, so no backend was allocated");
  check("registration-valid-config");

  // Tool factories and the access policy that gates them (no allocation on a denied call).
  const toolFactory = (name) => registry.tools.find((tool) => tool.names.includes(name)).factory;
  const owner = { agentId: "main", config: hostConfig, memoryAudience: { kind: "owner-private", agentId: "main" } };
  const tools = Object.fromEntries(TOOLS.map((name) => [name, toolFactory(name)(owner)]));
  for (const name of TOOLS) assert.equal(tools[name]?.name, name);
  const searchParameters = plain(tools.memory_search.parameters);
  assert.deepEqual(searchParameters.properties.attempt, { type: "integer", minimum: 1, maximum: 5 });
  assert.ok(!searchParameters.required.includes("attempt"), "search attempt is optional");
  const tokenReadsBeforeInvalidSearch = hygiene.tokenReads.count;
  await assert.rejects(tools.memory_search.execute("call-invalid-attempt", { query: "x", attempt: 6 }),
    { name: "MemoryInputError" });
  assert.equal(hygiene.tokenReads.count, tokenReadsBeforeInvalidSearch, "invalid search attempt never allocates a backend");
  check("search-attempt-before-allocation");
  assert.equal(toolFactory("memory_search")({ config: hostConfig }), null, "no agent, no tool");
  assert.equal(toolFactory("memory_search")({ ...owner, config: { plugins: { slots: { memory: "memory-core" } } } }), null,
    "another memory owner, no tool");
  const noAudience = toolFactory("memory_search")({ agentId: "main", config: hostConfig });
  await assert.rejects(noAudience.execute("call-denied", { query: "x" }), { name: "MemoryAccessDeniedError" });
  const otherAgent = toolFactory("memory_search")({ ...owner, memoryAudience: { kind: "owner-private", agentId: "intruder" } });
  await assert.rejects(otherAgent.execute("call-denied", { query: "x" }), { name: "MemoryAccessDeniedError" });
  const { providerRuntime } = capability;
  const hostOpen = await providerRuntime.open({ agentId: "main", cfg: hostConfig,
    context: { authority: { kind: "host", operation: "smoke" }, assertCurrent() {} } });
  await assert.rejects(hostOpen.provider.search({ query: "x" }), { name: "MemoryAccessDeniedError" });
  await hostOpen.provider.close();
  assert.equal(hygiene.tokenReads.count, 0, "denied calls never allocate a backend");
  check("access-policy-before-allocation");

  // 3. The registered CLI registrar on a REAL Commander program.
  const { Command } = await import("commander");
  const program = new Command().exitOverride().configureOutput({ writeOut() {}, writeErr() {} });
  await registry.cliRegistrars[0].register({ program, config: hostConfig, logger: { info() {}, warn() {}, error() {}, debug() {} } });
  const root = program.commands.find((command) => command.name() === "reflectlog");
  assert.ok(root, "reflectlog command registered");
  const shared = ["--agent", "--conversation", "--json"];
  const expectedCommands = {
    status: [], search: ["--limit", "--min-score"], list: ["--limit", "--offset"], get: ["--from", "--lines"],
    forget: ["--yes"], export: ["--out"], import: ["--workspace-dir", "--dry-run", "--archive"],
  };
  assert.deepEqual(root.commands.map((command) => command.name()).sort(), Object.keys(expectedCommands).sort());
  for (const [name, extra] of Object.entries(expectedCommands)) {
    const longs = root.commands.find((command) => command.name() === name).options.map((option) => option.long).sort();
    assert.deepEqual(longs, [...shared, ...extra].sort(), `${name} options`);
  }
  const argumentsOf = (name) => root.commands.find((command) => command.name() === name).registeredArguments
    .map((argument) => `${argument.name()}${argument.variadic ? "..." : ""}${argument.required ? "!" : ""}`);
  assert.deepEqual([argumentsOf("search"), argumentsOf("get"), argumentsOf("forget")],
    [["query...!"], ["ref!"], ["ref...!"]]);
  check("cli-commander-tree");

  // 5. Lifecycle after lazy allocation, token absent.
  // Allocation proof: only a created client reads the token variable (at connect time), and only it can raise
  // `unauthorized`; a denied/never-allocated call raises neither. Both are observed per operation.
  const allocated = async (operation, label) => {
    const before = hygiene.tokenReads.count;
    await assert.rejects(operation(), (error) => error?.name === "ReflectLogError" && error.code === "unauthorized"
      && error.message.includes(TOKEN_ENV), `${label}: expected the plugin's unauthorized error`);
    assert.ok(hygiene.tokenReads.count > before, `${label}: the backend client read the token variable`);
  };
  const flushTool = toolFactory("memory_store")({ ...owner, memoryFlush: { flushId: "smoke-flush" } });
  const run = (name, params = { query: "x" }) => tools[name].execute("call", params);
  const store = () => run("memory_store", { memories: ["smoke fact"] });
  const service = registry.services[0].service;
  const retirements = {
    "service.stop": () => service.stop(),
    "provider.closeAllMemorySearchManagers": () => providerRuntime.closeAllMemorySearchManagers(),
  };

  // Retirement before any allocation has nothing to dispose.
  await service.start({});
  await retirements["service.stop"]();
  assert.equal(hygiene.tokenReads.count, 0);

  const open = await providerRuntime.open({ agentId: "main", cfg: hostConfig,
    context: { authority: { kind: "operator" }, assertCurrent() {} } });
  assert.equal(hygiene.tokenReads.count, 0, "opening the provider allocates nothing");
  for (const [retirement, retire] of Object.entries(retirements)) {
    await allocated(() => run("memory_search"), `${retirement}: first lazy allocation (tool)`);
    await allocated(() => open.provider.search({ query: "x" }), `${retirement}: provider search shares the backend`);
    const health = await open.provider.health();
    assert.equal(health.status, "unavailable");
    assert.match(health.message, new RegExp(TOKEN_ENV));
    // The allocated service now holds a PENDING flush entry (reserved before the effect runs); retiring in the same
    // tick disposes that ledger, and the dispatch-less waiter settles with the pre-dispatch FlushDisposedError.
    const inflight = flushTool.execute("flush-call", { memories: ["smoke fact"] });
    const retired = retire();
    const [flushed, retiredResult] = await Promise.allSettled([inflight, retired]);
    assert.equal(retiredResult.status, "fulfilled", `${retirement}: retirement completes`);
    assert.equal(flushed.status, "rejected");
    assert.equal(flushed.reason?.name, "FlushDisposedError", `${retirement}: the allocated service was disposed with its pending flush`);
    assert.match(flushed.reason.message, /nothing was sent/);
    check(`lifecycle-retire:${retirement}`);

    // Fresh reuse: a disposed service rejects mutations with FlushDisposedError, so `unauthorized` here can only come
    // from a freshly allocated service and client. The same flush id is free again (the old ledger is gone).
    await allocated(store, `${retirement}: fresh service accepts a mutation`);
    await allocated(() => flushTool.execute("flush-call", { memories: ["smoke fact"] }), `${retirement}: flush id reusable`);
    await allocated(() => run("memory_list", {}), `${retirement}: reads work`);
    check(`lifecycle-fresh-reuse:${retirement}`);
  }
  await open.provider.close();
  await retirements["service.stop"]();

  // CLI action through the real registrar: connectivity fails cleanly with the authorization hint.
  const lines = [];
  const log = console.log;
  console.log = (line) => { lines.push(String(line)); };
  try {
    await program.parseAsync(["reflectlog", "status", "--json"], { from: "user" });
  } finally {
    console.log = log;
  }
  assert.equal(process.exitCode, 1, "status with no token reports failure through the exit code");
  process.exitCode = 0;
  const status = JSON.parse(lines.join("\n"));
  assert.deepEqual(status.checks.filter((c) => !c.pass).map((c) => c.name), ["transport/token", "connectivity"]);
  assert.equal(status.checks.find((c) => c.name === "connectivity").hint, `Authorization failed; set ${TOKEN_ENV}.`);
  await service.stop();
  check("cli-status-unauthorized");

  // 4. Invalid configuration: a fresh real registry per case. hostAccepts=true cases are runtime-only
  // rules (duplicate override values, full URL parse); direct registration must still fail closed.
  const invalidCases = [
    { name: "bad-url-scheme", config: { url: "ftp://reflectlog.invalid/mcp" }, hostAccepts: false, issuePath: ["url"] },
    { name: "reserved-dot", config: { workspaceOverrides: { main: "." } }, hostAccepts: false, issuePath: ["workspaceOverrides", "main"] },
    { name: "reserved-dotdot", config: { workspaceOverrides: { main: ".." } }, hostAccepts: false, issuePath: ["workspaceOverrides", "main"] },
    { name: "derived-owner", config: { workspaceOverrides: { main: "p-main-0123456789abcdef01234567" } }, hostAccepts: false, issuePath: ["workspaceOverrides", "main"] },
    { name: "derived-conversation", config: { workspaceOverrides: { main: "c-main-0123456789abcdef01234567" } }, hostAccepts: false, issuePath: ["workspaceOverrides", "main"] },
    { name: "duplicate-overrides", config: { workspaceOverrides: { main: "shared", ops: "shared" } }, hostAccepts: true },
    { name: "malformed-url", config: { url: "http://" }, hostAccepts: true },
    { name: "unknown-key", config: { bogus: true }, hostAccepts: false },
  ];
  for (const invalid of invalidCases) {
    const verdict = hostVerdict(invalid.config);
    assert.equal(verdict.success, invalid.hostAccepts, `${invalid.name}: host schema verdict`);
    if (invalid.issuePath !== undefined) {
      const paths = (verdict.error?.issues ?? []).map((issue) => issue.path.join("/"));
      assert.ok(paths.includes(invalid.issuePath.join("/")),
        `${invalid.name}: issue path ${invalid.issuePath.join("/")} among [${paths}]`);
    }
    const { host: bad, logs: badLogs } = await newHost();
    const badRecord = record();
    registerLikeLoader({ host: bad, entry, record: badRecord, config: hostConfig, pluginConfig: invalid.config });
    assert.equal(bad.registry.tools.length, 0, `${invalid.name}: no tools`);
    assert.equal(bad.registry.services.length, 0, `${invalid.name}: no service`);
    assert.equal(bad.registry.cliRegistrars.length, 0, `${invalid.name}: no CLI`);
    assert.deepEqual(badRecord.toolNames.concat(badRecord.services, badRecord.cliCommands), []);
    assert.deepEqual(bad.registry.diagnostics, []);
    assert.equal(bad.registry.memoryCapabilities.length, 1, `${invalid.name}: only the disabled memory capability`);
    const disabled = bad.registry.memoryCapabilities[0].capability;
    assert.deepEqual(Object.keys(disabled).sort(), ["promptBuilder", "providerRuntime"]);
    assert.deepEqual(disabled.promptBuilder({}), []);
    const errors = badLogs.filter(({ level }) => level === "error");
    assert.equal(errors.length, 1);
    assert.match(errors[0].message, /^ReflectLog configuration rejected: /);
    assert.deepEqual(badLogs.filter(({ level }) => level === "warn"), []);
    const opened = await disabled.providerRuntime.open({ agentId: "main", cfg: hostConfig,
      context: { authority: { kind: "operator" }, assertCurrent() {} } });
    assert.equal(opened.provider, null);
    assert.equal(typeof opened.error, "string");
    await disabled.providerRuntime.closeAllMemorySearchManagers();
    check(`registration-invalid-config:${invalid.name}`);
  }
  assert.equal(hygiene.tokenReads.count > 0, true);

  // Valid custom overrides plus an uppercase-scheme URL: host-accepted and fully registered.
  const customConfig = {
    url: "HTTP://reflectlog.invalid:9103/mcp", namespace: "smoke",
    workspaceOverrides: { main: "team.notes", ops: "ops_ws" },
  };
  assert.equal(hostVerdict(customConfig).success, true, "custom overrides and uppercase scheme are host-accepted");
  const { host: custom, logs: customLogs } = await newHost();
  const customRecord = record();
  registerLikeLoader({ host: custom, entry, record: customRecord, config: hostConfig, pluginConfig: customConfig });
  assert.deepEqual(custom.registry.diagnostics, [], "custom config: no registration diagnostics");
  assert.deepEqual(customLogs.filter(({ level }) => level === "warn" || level === "error"), []);
  assert.equal(custom.registry.tools.length, TOOLS.length);
  assert.deepEqual(customRecord.toolNames, TOOLS);
  assert.equal(custom.registry.services.length, 1);
  assert.equal(custom.registry.cliRegistrars.length, 1);
  await custom.registry.services[0].service.stop();
  check("registration-valid-custom-overrides");

  // Slot warnings: the plugin registers fully but tells the operator how to take the slot.
  const { host: loose, logs: looseLogs } = await newHost();
  registerLikeLoader({ host: loose, entry, record: record(false), config: {}, pluginConfig });
  assert.deepEqual(looseLogs.filter(({ level }) => level === "warn").map(({ message }) => message), [
    "ReflectLog: set plugins.entries.memory-core.enabled=false to avoid conflicting memory guidance.",
    "ReflectLog: set plugins.slots.memory=memory-reflectlog to enable memory tools.",
  ]);
  assert.equal(loose.registry.memoryCapabilities[0].memorySlotSelected, false);
  assert.equal(loose.registry.tools.length, TOOLS.length);
  await loose.registry.services[0].service.stop();
  check("slot-warnings");

  return { host: { version: host.version, chunk: host.chunk } };
}

let report;
try {
  const result = await main();
  // Let process 'warning' events and rejected-promise reports surface before judging hygiene.
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(hygiene.violations, [], "hygiene violations");
  assert.deepEqual(hygiene.lingeringResources(), [], "lingering handles");
  report = { ok: true, checks, ...result };
} catch (error) {
  process.exitCode = 1;
  report = { ok: false, checks, violations: hygiene.violations, error: String(error?.stack ?? error) };
}
process.stdout.write(`${JSON.stringify(report)}\n`);
