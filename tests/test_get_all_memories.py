"""Live pagination demo and typed response boundary shared by MCP demos."""

import os

import anyio
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from pydantic import BaseModel


class Record(BaseModel):
    id: int
    memory: str
    created_at: str | None


class SearchRecord(Record):
    score: float


class Page(BaseModel):
    records: list[Record]
    total: int
    offset: int
    limit: int
    truncated: bool


class SearchResponse(BaseModel):
    results: list[SearchRecord]
    score_kind: str
    limit: int


class Replacement(BaseModel):
    old_memory: str
    new_memory: str
    confidence: float
    reason: str


class AddResponse(BaseModel):
    stored_count: int
    skipped_count: int
    replaced_count: int
    replacements: list[Replacement]
    dry_run: bool


class GetResponse(BaseModel):
    record: Record | None


class DeleteResponse(BaseModel):
    removed: list[int]
    missing: list[int]


def parse_response[T: BaseModel](result: CallToolResult, model: type[T]) -> T:
    """Reject MCP errors and parse structured or text JSON at the boundary."""
    if result.is_error:
        raise RuntimeError("MCP tool returned an error")
    if result.structured_content is not None:
        return model.model_validate(result.structured_content)
    for content in result.content:
        if isinstance(content, TextContent):
            return model.model_validate_json(content.text)
    raise ValueError("MCP response has no JSON content")


async def all_records(client: Client, workspace_id: str) -> list[Record]:
    records: list[Record] = []
    while True:
        page = parse_response(
            await client.call_tool(
                "get_all_memories",
                {"workspace_id": workspace_id, "limit": 100, "offset": len(records)},
            ),
            Page,
        )
        records.extend(page.records)
        if not page.truncated:
            return records
        if not page.records:
            raise ValueError("Truncated page made no progress")


async def main() -> None:
    workspace_id = os.environ.get("WORKSPACE_ID", "ReflectLog")
    async with Client("http://127.0.0.1:9103/mcp") as client:
        print("Tools:", [tool.name for tool in await client.list_tools()])
        for record in await all_records(client, workspace_id):
            print(record.model_dump())


if __name__ == "__main__":
    anyio.run(main)
