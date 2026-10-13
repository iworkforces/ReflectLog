export const MAX_SEARCH_ATTEMPTS = 5;

export function searchGuidance(toolName: string, attempt: number, hitCount: number): string {
  const prefix = `Search attempt ${attempt} of ${MAX_SEARCH_ATTEMPTS}.`;
  if (attempt === MAX_SEARCH_ATTEMPTS) {
    return hitCount === 0
      ? `${prefix} Search limit reached. Stop searching and say no relevant memory was found.`
      : `${prefix} Search limit reached. Verify each memory is relevant to the request before using it. Stop searching; if none is relevant, say no relevant memory was found.`;
  }
  const outcome = hitCount === 0
    ? "No memories were returned. Rewrite the query"
    : "Verify each memory is relevant to the request before using it. If none is relevant, rewrite the query";
  return `${prefix} ${outcome} using different keywords, synonyms, broader or narrower terms, or key entities, then call ${toolName} with attempt=${attempt + 1}.`;
}
