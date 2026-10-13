// Differential matrix: the pinned host's JSON Schema verdict (the manifest copy, through the public
// SDK validator the real loader uses) versus the runtime Zod `parseConfig`. The JSON Schema must never
// narrow Zod acceptance; the runtime parse stays authoritative for full URL parsing and uniqueness.
import { readFileSync } from "node:fs";

import { buildJsonPluginConfigSchema } from "openclaw/plugin-sdk/plugin-entry";
import { describe, expect, it } from "vitest";
import { z } from "zod";

import { configJsonSchema, parseConfig } from "../../src/config.js";

const manifest = z
  .object({ configSchema: z.unknown() })
  .parse(JSON.parse(readFileSync(new URL("../../openclaw.plugin.json", import.meta.url), "utf8")));

const hostConfigSchema = buildJsonPluginConfigSchema(
  manifest.configSchema as Parameters<typeof buildJsonPluginConfigSchema>[0],
);
const hostSafeParse = hostConfigSchema.safeParse;
if (hostSafeParse === undefined) throw new Error("pinned host schema offers no safeParse");

interface MatrixCase {
  readonly name: string;
  readonly config: Readonly<Record<string, unknown>>;
  readonly host: boolean;
  readonly runtime: boolean;
  readonly issuePath?: readonly (string | number)[];
}

const urlCase = (value: string, host: boolean, runtime: boolean): MatrixCase => ({
  name: `url ${JSON.stringify(value)}`,
  config: { url: value },
  host,
  runtime,
  ...(host ? {} : { issuePath: ["url"] as const }),
});

const overrideCase = (value: string, accepted: boolean): MatrixCase => ({
  name: `override ${JSON.stringify(value)}`,
  config: { workspaceOverrides: { main: value } },
  host: accepted,
  runtime: accepted,
  ...(accepted ? {} : { issuePath: ["workspaceOverrides", "main"] as const }),
});

const ACCEPTED_URLS = [
  "http://127.0.0.1:9103/mcp",
  "https://reflectlog.example/mcp",
  "HTTP://127.0.0.1:9103/mcp",
  "HtTpS://h/mcp",
  " http://h/mcp ",
  "\thttp://h/mcp\n",
  "\u00a0http://h/mcp",
  "\ufeffhttp://h/mcp",
  "http://exa\tmple.com/mcp",
  "http://h/a b",
  "http://[::1]:9103/mcp",
  "http://user:pw@h:1/mcp?x=1#f",
  "http:///h/mcp",
  "http://h:65535/mcp",
];

const REJECTED_URLS = [
  "ftp://reflectlog.invalid/mcp",
  "file:///tmp/x",
  "ws://h/mcp",
  "javascript:alert(1)",
  "mailto:a@b",
  "not a url",
  "",
  "http:example.com",
  "https:/path",
  "//h/mcp",
  "\u0001http://h/mcp",
  "ht\ttp://h/mcp",
];

/** Malformed http(s)-prefixed URLs only the runtime URL parse can reject. */
const RUNTIME_ONLY_URLS = ["http://", "https://", "http://exa mple.com", "https://[::1", "http://h:99999/mcp"];

const ACCEPTED_OVERRIDES = [
  "...",
  "a..b",
  "my.workspace",
  "team_a",
  "p-main",
  "p-main-0123456789abcdef0123456",
  "x-main-0123456789abcdef01234567",
];

const REJECTED_OVERRIDES = [
  ".",
  "..",
  "p-main-0123456789abcdef01234567",
  "c-main-0123456789abcdef01234567",
  "p-main-0123456789ABCDEF01234567",
];

const DUPLICATE_OVERRIDES: MatrixCase = {
  name: "override duplicate values",
  config: { workspaceOverrides: { main: "shared", ops: "shared" } },
  host: true,
  runtime: false,
};

const VALID_CUSTOM: MatrixCase = {
  name: "custom url + overrides",
  config: {
    url: "HTTP://reflectlog.invalid:9103/mcp",
    workspaceOverrides: { main: "team.notes", ops: "ops_ws" },
  },
  host: true,
  runtime: true,
};

const cases: readonly MatrixCase[] = [
  ...ACCEPTED_URLS.map((value) => urlCase(value, true, true)),
  ...REJECTED_URLS.map((value) => urlCase(value, false, false)),
  ...RUNTIME_ONLY_URLS.map((value) => urlCase(value, true, false)),
  ...ACCEPTED_OVERRIDES.map((value) => overrideCase(value, true)),
  ...REJECTED_OVERRIDES.map((value) => overrideCase(value, false)),
  DUPLICATE_OVERRIDES,
  VALID_CUSTOM,
];

describe("config schema differential matrix", () => {
  it("keeps the manifest configSchema identical to configJsonSchema", () => {
    expect(manifest.configSchema).toEqual(configJsonSchema);
  });

  it.each([...cases])("$name", (matrixCase) => {
    const hostVerdict = hostSafeParse(matrixCase.config);
    expect(hostVerdict.success).toBe(matrixCase.host);
    expect(parseConfig(matrixCase.config).ok).toBe(matrixCase.runtime);
    if (matrixCase.issuePath !== undefined) {
      const paths = (hostVerdict.error?.issues ?? []).map((issue) => issue.path);
      expect(paths).toContainEqual([...matrixCase.issuePath]);
    }
  });

  it("never narrows: every config the runtime accepts is host-accepted", () => {
    for (const matrixCase of cases) {
      if (parseConfig(matrixCase.config).ok) {
        expect(hostSafeParse(matrixCase.config).success, matrixCase.name).toBe(true);
      }
    }
  });

  it("host-accepted but runtime-rejected residue is exactly the documented set", () => {
    const residue = cases
      .filter((matrixCase) => hostSafeParse(matrixCase.config).success && !parseConfig(matrixCase.config).ok)
      .map((matrixCase) => matrixCase.name)
      .sort();
    const expected = [...RUNTIME_ONLY_URLS.map((value) => `url ${JSON.stringify(value)}`), DUPLICATE_OVERRIDES.name].sort();
    expect(residue).toEqual(expected);
  });
});
