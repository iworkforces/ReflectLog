import type { MemoryPluginCapability } from "openclaw/plugin-sdk/memory-host-core";

// The SDK exports the capability, not the native handle types; derive them as the host docs prescribe.
export type MemoryProviderRuntime = NonNullable<MemoryPluginCapability["providerRuntime"]>;
export type MemoryProviderOpenParams = Parameters<MemoryProviderRuntime["open"]>[0];
export type MemoryProviderOpenResult = Awaited<ReturnType<MemoryProviderRuntime["open"]>>;
export type MemoryProviderHandle = NonNullable<MemoryProviderOpenResult["provider"]>;
export type MemoryCallerContext = MemoryProviderOpenParams["context"];
export type MemoryCallerAuthority = MemoryCallerContext["authority"];
export type MemoryAudience = NonNullable<
  Extract<MemoryCallerAuthority, { kind: "session" }>["audience"]
>;
export type MemorySearchRequest = Parameters<MemoryProviderHandle["search"]>[0];
export type MemorySearchPage = Awaited<ReturnType<MemoryProviderHandle["search"]>>;
export type MemorySearchHit = MemorySearchPage["hits"][number];
export type MemoryReference = MemorySearchHit["reference"];
export type MemoryGetRequest = Parameters<MemoryProviderHandle["get"]>[0];
export type MemoryGetResult = Awaited<ReturnType<MemoryProviderHandle["get"]>>;
export type MemoryHealth = Awaited<ReturnType<MemoryProviderHandle["health"]>>;
export type MemoryProviderCapabilities = MemoryProviderHandle["capabilities"];
export type MemoryFlushPlanResolver = NonNullable<MemoryPluginCapability["providerFlushPlanResolver"]>;
export type MemoryPromptSectionBuilder = NonNullable<MemoryPluginCapability["promptBuilder"]>;
