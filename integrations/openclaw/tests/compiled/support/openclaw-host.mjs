// Host adapter for the compiled-package smoke: the ONLY file that knows how to reach OpenClaw's real plugin registry.
//
// OpenClaw publishes no registry factory. The real `createPluginRegistry` ships inside a hashed internal chunk
// (`dist/registry-<hash>.mjs`, exported under a mangled alias), so this adapter finds the chunk by content, imports it
// by file URL, and picks the function by its NAME. Every step is fail-closed with a `HostSeamError`: when the host pin
// moves and the seam shifts, the smoke stops with a clear message. It never falls back to a handwritten API, and it
// never uses the SDK's `createTestPluginApi` (a stub whose registerTool/registerService/... do nothing).
//
// WHEN THE HOST PIN MOVES (devDependencies.openclaw, openclaw.build, minHostVersion ...):
//  1. Check upstream `src/plugins/registry.ts` (`createPluginRegistry`, `PluginRegistryParams`), `registry-api.ts`,
//     `loader-records.ts` (`createPluginRecord`) and `loader-runtime-candidate.ts` (how the loader builds the record and
//     calls `registryBuilder.createApi`) at the new tag.
//  2. Update SUPPORTED_HOST_VERSION, the chunk markers, and `createLoaderRecord` below if those changed.
//  3. Only then rerun `pnpm test:dist`. Do not loosen a guard to make a failure go away.
//
// The guard logic is pure (`assertHostSeam` and friends take plain DESCRIPTORS) so smoke.test.mjs can exercise every
// negative path without touching node_modules.
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const SUPPORTED_HOST_VERSION = "2026.10.5";
export const REGISTRY_FACTORY_NAME = "createPluginRegistry";
export const REGISTRY_FACTORY_ARITY = 1;
export const REGISTRY_CHUNK_PATTERN = /^registry-.*\.mjs$/;
/** Both strings must occur in the one chunk that defines the real registry. */
export const REGISTRY_CHUNK_MARKERS = Object.freeze({
  factory: "function createPluginRegistry(",
  memoryGate: "only memory plugins can register a memory capability",
});
/** Members of `registry` the smoke reads back (all arrays). */
export const REQUIRED_REGISTRY_ARRAYS = Object.freeze([
  "tools", "services", "cliRegistrars", "memoryCapabilities", "diagnostics", "plugins",
]);

export class HostSeamError extends Error {
  constructor(message) {
    super(`OpenClaw host seam changed or is unusable: ${message}`);
    this.name = "HostSeamError";
  }
}

// ---- pure guards (descriptors in, HostSeamError out) -------------------------------------------------------------

/** Guard 1: the installed host is exactly the pinned release. */
export function assertHostVersion(version) {
  if (version !== SUPPORTED_HOST_VERSION) {
    throw new HostSeamError(`installed openclaw is ${JSON.stringify(version)}, this adapter supports exactly ${SUPPORTED_HOST_VERSION}`);
  }
}

/** Guard 2: exactly one registry chunk carries the markers. Returns its file name. */
export function assertRegistryChunk(chunks) {
  const matches = chunks.filter((chunk) => chunk.hasFactory === true && chunk.hasMemoryGate === true);
  if (matches.length !== 1) {
    throw new HostSeamError(`expected exactly one dist/registry-*.mjs chunk defining ${REGISTRY_FACTORY_NAME}, found ${matches.length}`
      + (matches.length ? ` (${matches.map((chunk) => chunk.file).join(", ")})` : ` among ${chunks.length} registry chunks`));
  }
  return matches[0].file;
}

/** Guard 3: the chunk exposes exactly one function named createPluginRegistry, with the expected arity. */
export function assertRegistryFactory(factories) {
  const matches = factories.filter((entry) => entry.name === REGISTRY_FACTORY_NAME);
  if (matches.length !== 1) {
    throw new HostSeamError(`expected exactly one exported function named ${REGISTRY_FACTORY_NAME}, found ${matches.length}`);
  }
  if (matches[0].arity !== REGISTRY_FACTORY_ARITY) {
    throw new HostSeamError(`${REGISTRY_FACTORY_NAME} takes ${matches[0].arity} parameters, expected ${REGISTRY_FACTORY_ARITY}`);
  }
}

/** Guard 4: the built registry exposes what the smoke reads and calls. */
export function assertBuilderShape(builder) {
  if (builder?.createApi !== "function") throw new HostSeamError("registry builder has no createApi function");
  if (builder.registry === null || typeof builder.registry !== "object") throw new HostSeamError("registry builder has no registry object");
  const missing = REQUIRED_REGISTRY_ARRAYS.filter((name) => builder.registry[name] !== "array");
  if (missing.length) throw new HostSeamError(`registry is missing array member(s): ${missing.join(", ")}`);
}

/** All four guards over one descriptor: { version, chunks, factories, builder }. */
export function assertHostSeam(descriptor) {
  assertHostVersion(descriptor.version);
  assertRegistryChunk(descriptor.chunks);
  assertRegistryFactory(descriptor.factories);
  assertBuilderShape(descriptor.builder);
}

// ---- discovery and import ----------------------------------------------------------------------------------------

/** Locate the installed host through a PUBLIC export (the exports map hides everything else). */
export function locateHostRoot() {
  let dir = dirname(fileURLToPath(import.meta.resolve("openclaw/plugin-sdk/plugin-entry")));
  for (;;) {
    try {
      if (JSON.parse(readFileSync(join(dir, "package.json"), "utf8")).name === "openclaw") return dir;
    } catch {
      // no package.json here; keep walking
    }
    const parent = dirname(dir);
    if (parent === dir) throw new HostSeamError("could not find the openclaw package root above its plugin-sdk entry");
    dir = parent;
  }
}

export function describeInstalledHost(root = locateHostRoot()) {
  const version = JSON.parse(readFileSync(join(root, "package.json"), "utf8")).version;
  const dist = join(root, "dist");
  const chunks = readdirSync(dist).filter((file) => REGISTRY_CHUNK_PATTERN.test(file)).map((file) => {
    const text = readFileSync(join(dist, file), "utf8");
    return { file, hasFactory: text.includes(REGISTRY_CHUNK_MARKERS.factory), hasMemoryGate: text.includes(REGISTRY_CHUNK_MARKERS.memoryGate) };
  });
  return { root, version, chunks };
}

const describeMember = (value) => Array.isArray(value) ? "array" : typeof value;

/**
 * Build a REAL registry. `logger` is the host logger the registry hands to plugins (the smoke records it);
 * `runtime` is `PluginRuntime`, which this plugin never touches, so the smoke passes a runtime that throws on access
 * (a plugin that starts using it fails the gate instead of silently getting a stub).
 */
export async function openHostRegistry({ logger, runtime }) {
  const host = describeInstalledHost();
  assertHostVersion(host.version);
  const file = assertRegistryChunk(host.chunks);
  const namespace = await import(pathToFileURL(join(host.root, "dist", file)).href);
  const factories = Object.values(namespace).filter((value) => typeof value === "function")
    .map((fn) => ({ name: fn.name, arity: fn.length }));
  assertRegistryFactory(factories);
  const factory = Object.values(namespace).find((value) => typeof value === "function" && value.name === REGISTRY_FACTORY_NAME);
  // Same shape the upstream loader passes, minus gateway handlers/host services (no gateway exists here).
  // activateGlobalSideEffects:false keeps the registry from touching process-global singletons (upstream test helpers do the same).
  const builder = factory({ logger, runtime, coreGatewayMethodNames: [], activateGlobalSideEffects: false });
  assertBuilderShape({
    createApi: describeMember(builder?.createApi),
    registry: builder?.registry && typeof builder.registry === "object"
      ? Object.fromEntries(REQUIRED_REGISTRY_ARRAYS.map((name) => [name, describeMember(builder.registry[name])])) : null,
  });
  return { version: host.version, chunk: file, registry: builder.registry, createApi: builder.createApi, builder };
}

// ---- record construction -----------------------------------------------------------------------------------------

/**
 * Record as upstream `createPluginRecord` (loader-records.ts) builds it for an installed npm plugin, then the loader
 * (loader-runtime-candidate.ts) sets `kind` from the manifest and `memorySlotSelected` when the memory slot picks it.
 * Fields the registrars push into (toolNames, services, cliCommands ...) start empty like upstream.
 */
export function createLoaderRecord({ manifest, packageJson, rootDir, source, memorySlotSelected }) {
  return {
    id: manifest.id,
    name: manifest.name ?? manifest.id,
    description: manifest.description,
    packageName: packageJson.name,
    packageVersion: packageJson.version,
    version: packageJson.version,
    builtWithOpenClawVersion: packageJson.openclaw?.build?.openclawVersion,
    format: "openclaw",
    source,
    rootDir,
    origin: "global",
    enabled: true,
    status: "loaded",
    syntheticAuthRefs: [],
    toolNames: [],
    hookNames: [],
    channelIds: [],
    cliBackendIds: [],
    providerIds: [],
    contextEngineIds: [],
    agentHarnessIds: [],
    cliCommands: [],
    services: [],
    gatewayDiscoveryServiceIds: [],
    commands: [],
    httpRoutes: 0,
    hookCount: 0,
    configSchema: true,
    contracts: manifest.contracts,
    kind: manifest.kind,
    ...(memorySlotSelected ? { memorySlotSelected: true } : {}),
  };
}

/** What the loader does for a plugin that registered successfully: createApi, then register(api). */
export function registerLikeLoader({ host, entry, record, config, pluginConfig }) {
  const api = host.createApi(record, { config, pluginConfig, registrationMode: "full" });
  const result = entry.register(api);
  // The real loader runs register synchronously and rejects a promise result.
  if (result !== undefined && typeof result?.then === "function") throw new HostSeamError("plugin register() returned a promise; the host requires a synchronous register");
  host.registry.plugins.push(record);
  return api;
}
