// Compiled-package smoke gate (`pnpm test:dist`). Plain `node --test`: no TypeScript, no vitest, no mocks.
// Needs `pnpm build` first. Spawns smoke-child.mjs in a native Node process and judges its exit and report.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import {
  assertBuilderShape, assertHostSeam, assertHostVersion, assertRegistryChunk, assertRegistryFactory,
  describeInstalledHost, HostSeamError, REGISTRY_FACTORY_ARITY, REGISTRY_FACTORY_NAME, SUPPORTED_HOST_VERSION,
} from "./support/openclaw-host.mjs";

const packageDir = fileURLToPath(new URL("../../", import.meta.url));
const child = fileURLToPath(new URL("./smoke-child.mjs", import.meta.url));
const CHILD_TIMEOUT_MS = 120_000;
/** Everything the child may inherit; notably no NODE_OPTIONS (loaders), no token, no OPENCLAW_*, no proxies. */
const ENV_ALLOWLIST = ["PATH", "Path", "SystemRoot", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "LANG"];
const cleanEnv = () => Object.fromEntries(ENV_ALLOWLIST.flatMap((name) => process.env[name] === undefined ? [] : [[name, process.env[name]]]));

const EXPECTED_CHECKS = [
  "entry-by-package-name", "registration-valid-config", "access-policy-before-allocation", "cli-commander-tree",
  "lifecycle-retire:service.stop", "lifecycle-fresh-reuse:service.stop",
  "lifecycle-retire:provider.closeAllMemorySearchManagers", "lifecycle-fresh-reuse:provider.closeAllMemorySearchManagers",
  "cli-status-unauthorized",
  "registration-invalid-config:bad-url-scheme", "registration-invalid-config:reserved-dot",
  "registration-invalid-config:reserved-dotdot", "registration-invalid-config:derived-owner",
  "registration-invalid-config:derived-conversation", "registration-invalid-config:duplicate-overrides",
  "registration-invalid-config:malformed-url", "registration-invalid-config:unknown-key",
  "registration-valid-custom-overrides", "slot-warnings",
];

const assertBuilt = () => assert.ok(existsSync(join(packageDir, "dist", "index.js")), "dist/index.js is missing: run pnpm build first");

function pidExists(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === "EPERM";
  }
}

/** Run the child natively (no shell, no loader flags); kill it on timeout and say so. */
function runChild(env = cleanEnv()) {
  return new Promise((resolve, reject) => {
    const proc = spawn(process.execPath, [child], { cwd: packageDir, env, stdio: ["ignore", "pipe", "pipe"], shell: false, windowsHide: true });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    proc.stdout.setEncoding("utf8").on("data", (chunk) => { stdout += chunk; });
    proc.stderr.setEncoding("utf8").on("data", (chunk) => { stderr += chunk; });
    const timer = setTimeout(() => { timedOut = true; proc.kill("SIGKILL"); }, CHILD_TIMEOUT_MS);
    proc.on("error", reject);
    proc.on("close", (code, signal) => {
      clearTimeout(timer);
      resolve({ pid: proc.pid, code, signal, stdout, stderr, timedOut });
    });
  });
}

function parseReport(result) {
  const lines = result.stdout.split("\n").filter((line) => line !== "");
  assert.equal(lines.length, 1, `child stdout must be exactly one JSON line, got:\n${result.stdout}\nstderr:\n${result.stderr}`);
  return JSON.parse(lines[0]);
}

describe("host seam guards (synthetic descriptors)", () => {
  const chunk = (file, hasFactory = true, hasMemoryGate = true) => ({ file, hasFactory, hasMemoryGate });
  const arrays = { tools: "array", services: "array", cliRegistrars: "array", memoryCapabilities: "array", diagnostics: "array", plugins: "array" };
  const good = () => ({
    version: SUPPORTED_HOST_VERSION,
    chunks: [chunk("registry-a.mjs"), chunk("registry-b.mjs", false, false)],
    factories: [{ name: REGISTRY_FACTORY_NAME, arity: REGISTRY_FACTORY_ARITY }, { name: "other", arity: 0 }],
    builder: { createApi: "function", registry: { ...arrays } },
  });
  const rejected = (descriptor, pattern) => assert.throws(() => assertHostSeam(descriptor), (error) => {
    assert.ok(error instanceof HostSeamError, `expected HostSeamError, got ${error}`);
    assert.match(error.message, pattern);
    return true;
  });

  it("accepts a matching host", () => {
    assert.doesNotThrow(() => assertHostSeam(good()));
    assert.equal(assertRegistryChunk(good().chunks), "registry-a.mjs");
  });
  it("rejects any other host version, even a newer one", () => {
    for (const version of ["2026.10.1", "2026.10.3", "2027.1.0", "", undefined]) rejected({ ...good(), version }, /supports exactly 2026\.10\.2/);
    assert.throws(() => assertHostVersion("2026.10.2-beta.1"), HostSeamError);
  });
  it("rejects zero registry chunks and chunks with only one marker", () => {
    rejected({ ...good(), chunks: [] }, /found 0 among 0/);
    rejected({ ...good(), chunks: [chunk("registry-x.mjs", true, false), chunk("registry-y.mjs", false, true)] }, /found 0 among 2/);
  });
  it("rejects several candidate chunks", () => {
    rejected({ ...good(), chunks: [chunk("registry-a.mjs"), chunk("registry-c.mjs")] }, /found 2 \(registry-a\.mjs, registry-c\.mjs\)/);
  });
  it("rejects a missing, duplicated, or re-shaped factory", () => {
    rejected({ ...good(), factories: [{ name: "other", arity: 1 }] }, /found 0/);
    rejected({ ...good(), factories: [{ name: REGISTRY_FACTORY_NAME, arity: 1 }, { name: REGISTRY_FACTORY_NAME, arity: 1 }] }, /found 2/);
    rejected({ ...good(), factories: [{ name: REGISTRY_FACTORY_NAME, arity: 2 }] }, /takes 2 parameters, expected 1/);
    assert.throws(() => assertRegistryFactory([]), HostSeamError);
  });
  it("rejects a registry builder without createApi or the registry arrays", () => {
    rejected({ ...good(), builder: { createApi: "undefined", registry: { ...arrays } } }, /no createApi/);
    rejected({ ...good(), builder: { createApi: "function", registry: null } }, /no registry object/);
    rejected({ ...good(), builder: undefined }, /no createApi/);
    for (const member of Object.keys(arrays)) {
      rejected({ ...good(), builder: { createApi: "function", registry: { ...arrays, [member]: "undefined" } } }, new RegExp(`missing array member\\(s\\): ${member}`));
    }
    rejected({ ...good(), builder: { createApi: "function", registry: { ...arrays, tools: "object" } } }, /missing array member\(s\): tools/);
    assert.throws(() => assertBuilderShape({}), HostSeamError);
  });
  it("accepts the real installed host", () => {
    const host = describeInstalledHost();
    assertHostVersion(host.version);
    assert.match(assertRegistryChunk(host.chunks), /^registry-.*\.mjs$/);
  });
});

describe("compiled package smoke", { timeout: CHILD_TIMEOUT_MS + 30_000 }, () => {
  it("registers the built package against the real OpenClaw registry", async () => {
    assertBuilt();
    const result = await runChild();
    assert.equal(result.timedOut, false, `child timed out after ${CHILD_TIMEOUT_MS} ms (a lingering handle keeps it alive?)\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
    const report = parseReport(result);
    assert.equal(report.ok, true, `smoke failed:\n${report.error}\nviolations: ${JSON.stringify(report.violations)}\nchecks passed: ${report.checks}`);
    assert.equal(result.code, 0, "child exits 0 on its own");
    assert.equal(result.signal, null);
    assert.equal(result.stderr, "", "no warnings or errors on stderr");
    assert.deepEqual(report.checks, EXPECTED_CHECKS);
    assert.equal(report.host.version, SUPPORTED_HOST_VERSION);
    assert.equal(pidExists(result.pid), false, "child process is gone");
  });

  it("refuses to run with a credential in the environment", async () => {
    assertBuilt();
    const result = await runChild({ ...cleanEnv(), REFLECTLOG_MCP_AUTH_TOKEN: "smoke-must-not-use-this" });
    const report = parseReport(result);
    assert.equal(report.ok, false);
    assert.notEqual(result.code, 0);
    assert.match(report.error, /forbidden environment inherited: REFLECTLOG_MCP_AUTH_TOKEN|token env must be absent/);
    assert.equal(result.stdout.includes("smoke-must-not-use-this"), false, "the token value is never echoed");
  });
});
