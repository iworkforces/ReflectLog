// Hygiene guards for the compiled-package smoke child. Install BEFORE importing the package or the host.
//
// The gate needs no network, no credentials and no lingering handles. Every violation is recorded (and thrown at the
// call site where possible); the child fails at the end if any were recorded, even when the code under test swallowed
// the throw.
import childProcess from "node:child_process";
import dns from "node:dns";
import http from "node:http";
import https from "node:https";
import module from "node:module";
import net from "node:net";
import tls from "node:tls";

/** Env names the child must not inherit: the token itself, host config, and anything proxy-like. */
export const FORBIDDEN_ENV = /^(REFLECTLOG_|OPENCLAW_|(HTTPS?|ALL|NO)_PROXY$)/i;
/** Handles that may legitimately remain: the child's own stdio. */
export const STDIO_RESOURCES = new Set(["PipeWrap", "TTYWrap"]);

export function installHygiene({ tokenEnv }) {
  const violations = [];
  const tokenReads = { count: 0 };
  const deny = (what) => {
    const error = new Error(`hygiene: forbidden ${what}`);
    violations.push(error.message);
    throw error;
  };

  const inherited = Object.keys(process.env).filter((name) => FORBIDDEN_ENV.test(name));
  if (inherited.length) violations.push(`hygiene: forbidden environment inherited: ${inherited.join(", ")}`);

  // Count reads of the token variable so the smoke can tell "never read" (lazy) from "read by a client" (allocated).
  const realEnv = process.env;
  process.env = new Proxy(realEnv, {
    get(target, name, receiver) {
      if (name === tokenEnv) tokenReads.count += 1;
      return Reflect.get(target, name, receiver);
    },
  });

  // Network, at every layer a client can reach (undici and fetch bottom out in net/tls sockets).
  net.Socket.prototype.connect = function connect() { return deny("net.Socket#connect"); };
  net.connect = net.createConnection = () => deny("net.connect");
  tls.connect = () => deny("tls.connect");
  http.request = http.get = () => deny("http.request");
  https.request = https.get = () => deny("https.request");
  globalThis.fetch = () => deny("fetch");
  for (const name of ["lookup", "resolve", "resolve4", "resolve6", "resolveAny", "resolveSrv", "reverse"]) {
    dns[name] = () => deny(`dns.${name}`);
    dns.promises[name] = () => deny(`dns.promises.${name}`);
  }
  module.syncBuiltinESMExports(); // named imports of builtins in later-loaded modules see the guards

  process.on("unhandledRejection", (reason) => { violations.push(`hygiene: unhandledRejection: ${reason}`); });
  process.on("uncaughtException", (error) => { violations.push(`hygiene: uncaughtException: ${error?.stack ?? error}`); });
  process.on("warning", (warning) => { violations.push(`hygiene: process warning: ${warning.name}: ${warning.message}`); });

  return {
    tokenReads,
    violations,
    /**
     * Call once the host chunk is loaded: its own import graph runs `ps` once to read this process's start time, which is
     * host-internal and not plugin behaviour. From here on the plugin must not spawn processes.
     */
    forbidSpawning() {
      for (const name of ["spawn", "spawnSync", "exec", "execSync", "execFile", "execFileSync", "fork"]) {
        childProcess[name] = () => deny(`child_process.${name}`);
      }
      module.syncBuiltinESMExports();
    },
    /** Call last, after every awaited operation settled. Returns the lingering non-stdio handles. */
    lingeringResources() {
      return process.getActiveResourcesInfo().filter((name) => !STDIO_RESOURCES.has(name));
    },
  };
}
