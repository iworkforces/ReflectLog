"""End-to-end FastMCP demonstration of all five tools against a live server."""

import os

import anyio
from fastmcp import Client
import pytest

from tests.test_get_all_memories import (
    AddResponse,
    DeleteResponse,
    GetResponse,
    SearchResponse,
    all_records,
    parse_response,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="Requires live server and API credentials",
)

MEMORIES = [
    "Python tutorial for beginners covers the essential syntax and core concepts. Learn about variables, data types, loops, and functions through practical examples. This comprehensive guide helps newcomers build a solid foundation in Python programming.",
    "Advanced Python programming explores decorators, metaclasses, and context managers for professional development. Master asynchronous programming with asyncio and concurrent.futures for high-performance applications. Deep dive into design patterns and best practices used in production systems.",
    "JavaScript fundamentals teach you the core concepts of web development and DOM manipulation. Understand event handling, callbacks, promises, and async/await for modern web applications. Learn ES6+ features including arrow functions, destructuring, and modules.",
    "Python data science guide introduces pandas, numpy, and matplotlib for data analysis and visualization. Process and clean large datasets efficiently using vectorized operations. Create insightful visualizations and statistical analyses to extract meaningful patterns from your data.",
    "Java enterprise patterns demonstrate architectural best practices for scalable business applications. Learn dependency injection, factory patterns, and service-oriented architecture principles. Implement robust solutions using Spring Framework and enterprise design patterns.",
    "Python machine learning covers scikit-learn, TensorFlow, and PyTorch for building intelligent systems. Train classification and regression models using supervised learning algorithms. Deploy neural networks and deep learning models for real-world prediction tasks.",
    "React frontend development focuses on building interactive user interfaces with components and hooks. Manage application state using useState, useEffect, and Context API. Create responsive, performant web applications following modern React best practices.",
    "Python web scraping demonstrates how to extract data from websites using BeautifulSoup and Scrapy. Handle dynamic content with Selenium and respect robots.txt guidelines. Parse HTML, navigate DOM trees, and store extracted data in structured formats.",
    "Docker containerization enables consistent deployment across different environments and platforms. Create lightweight, isolated containers using Dockerfile and docker-compose. Orchestrate multi-container applications and manage microservices efficiently.",
    "Python automation scripts streamline repetitive tasks and improve workflow efficiency. Automate file operations, data processing, and system administration using Python libraries. Schedule tasks with cron jobs and create robust error handling for production automation.",
]


async def test_real_server_with_fastmcp_client() -> None:
    workspace = os.getenv("WORKSPACE_ID", "ReflectLog")
    async with Client("http://127.0.0.1:9103/mcp") as client:
        print("Tools:", [tool.name for tool in await client.list_tools()])
        print(
            parse_response(
                await client.call_tool(
                    "add_memories",
                    {
                        "workspace_id": workspace,
                        "memories": MEMORIES,
                        "allow_replace": False,
                    },
                ),
                AddResponse,
            ).model_dump()
        )
        records = await all_records(client, workspace)
        print("Stored:", [record.model_dump() for record in records])
        for query in ("Python", "JavaScript"):
            response = parse_response(
                await client.call_tool(
                    "search_memories", {"workspace_id": workspace, "query": query}
                ),
                SearchResponse,
            )
            print(query, response.model_dump())
        java = next(record for record in records if record.memory == MEMORIES[4])
        fetched = parse_response(
            await client.call_tool(
                "get_memory_by_id", {"workspace_id": workspace, "id": java.id}
            ),
            GetResponse,
        )
        assert fetched.record is not None and fetched.record.memory == MEMORIES[4]
        deleted = parse_response(
            await client.call_tool(
                "delete_memories", {"workspace_id": workspace, "ids": [java.id]}
            ),
            DeleteResponse,
        )
        assert deleted.removed == [java.id] and not deleted.missing
        print("Deleted Java record:", deleted.model_dump())
        print("All five MCP tools exercised with real embeddings and USearch")


if __name__ == "__main__":
    anyio.run(test_real_server_with_fastmcp_client)
