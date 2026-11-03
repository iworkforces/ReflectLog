#!/usr/bin/env python3
"""Test case: cats -> dogs replacement."""

import asyncio
import os
import shutil
import sys

from dotenv import load_dotenv

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load .env file if it exists
load_dotenv(os.path.join(project_root, ".env"))


async def main() -> int:
    """Test the cats -> dogs replacement case."""
    workspace_id = "test-cats-dogs"
    index_path = f"indexes/{workspace_id}"

    # Clean up any existing index
    if os.path.exists(index_path):
        shutil.rmtree(index_path)

    # Set environment variables
    os.environ["WORKSPACE_ID"] = workspace_id
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["ENABLE_SMART_REPLACE"] = "true"
    os.environ["SMART_REPLACE_THRESHOLD"] = "0.7"
    os.environ["SMART_REPLACE_MIN_SIMILARITY"] = "0.5"  # Default threshold
    os.environ["LOG_LEVEL"] = "DEBUG"  # More verbose logging

    print("=" * 70)
    print("Testing: cats -> dogs replacement")
    print("=" * 70)

    from reflectlog.application.config.settings import Config
    from reflectlog.application.memory.manager import MemoryManager
    from reflectlog.application.utils.logging import create_logger

    config = Config.from_environment()
    logger = create_logger(config.workspace_id, config.log_level)

    print(f"llm_provider: {config.llm_provider}")
    print(f"smart_replace_threshold: {config.smart_replace_threshold}")
    print(f"smart_replace_min_similarity: {config.smart_replace_min_similarity}")
    print()

    manager = MemoryManager(config=config, logger=logger)

    first_message = "I like cats"
    second_message = "I don't like cat anymore, I like to have a golden retriever dog."

    try:
        # Step 1: Add first message
        print("-" * 70)
        print(f"Adding: {first_message}")
        print("-" * 70)
        result1 = await manager.add_messages_async([first_message])
        print(f"Result: {result1}")
        all_msgs = manager.get_all()
        print(f"Messages now: {all_msgs}")
        print()

        # Step 2: Add second message
        print("-" * 70)
        print(f"Adding: {second_message}")
        print("-" * 70)
        result2 = await manager.add_messages_async([second_message])
        print(f"Result: {result2}")
        all_msgs = manager.get_all()
        print(f"Messages now: {all_msgs}")
        print()

        # Verification
        print("-" * 70)
        print("Verification")
        print("-" * 70)
        if len(all_msgs) == 1 and "dog" in all_msgs[0].lower():
            print("✅ SUCCESS: Second message replaced the first!")
        elif len(all_msgs) == 2:
            print("❌ FAILED: Both messages exist (no replacement)")
            print("   Let's check the similarity score...")

            # Check similarity manually
            from reflectlog.core.config_adapters import ConfigAdapter
            from reflectlog.infrastructure.smart_replacer import (
                SmartReplacer,
                SmartReplacerConfig,
            )

            sr_config = SmartReplacerConfig.from_config(ConfigAdapter(config))
            replacer = SmartReplacer(config=sr_config, logger=None)

            print()
            print("   Testing replacement check directly...")
            should_replace, confidence, reason = await replacer.check_replacement(
                new_memory=second_message,
                existing_memory=first_message,
            )
            print(f"   should_replace: {should_replace}")
            print(f"   confidence: {confidence}")
            print(f"   reason: {reason}")
        else:
            print(f"❓ UNEXPECTED: {len(all_msgs)} messages")

    finally:
        manager.close()
        if os.path.exists(index_path):
            shutil.rmtree(index_path)

    print()
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
