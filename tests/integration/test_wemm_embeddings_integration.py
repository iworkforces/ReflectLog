"""Opt-in real-model checks for Tencent WeMM SentenceTransformer checkpoints."""

import math
import os

import pytest

from reflectlog.core.enums import WeMMDevice, WeMMModel
from reflectlog.infrastructure.embeddings.wemm_embedding import (
    WeMMEmbeddingConfig,
    WeMMEmbeddings,
)

RUN_LOCAL_MODEL_TESTS = os.getenv("RUN_LOCAL_MODEL_TESTS") == "1"
SELECTED_MODEL = os.getenv("WEMM_TEST_MODEL", WeMMModel.EMBEDDING_2B)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not RUN_LOCAL_MODEL_TESTS,
        reason="Set RUN_LOCAL_MODEL_TESTS=1 to download and run a WeMM checkpoint",
    ),
]


@pytest.mark.parametrize(
    ("model", "dimensions"),
    [
        (WeMMModel.EMBEDDING_2B, 2048),
        (WeMMModel.EMBEDDING_4B, 2560),
        (WeMMModel.EMBEDDING_9B, 4096),
    ],
)
def test_real_wemm_text_embeddings_are_finite_normalized_and_shaped(
    model: WeMMModel, dimensions: int
) -> None:
    if model.value != SELECTED_MODEL:
        pytest.skip(f"WEMM_TEST_MODEL selects {SELECTED_MODEL}")
    embeddings = WeMMEmbeddings(
        WeMMEmbeddingConfig(
            model=model,
            dimensions=dimensions,
            device=WeMMDevice.AUTO,
            batch_size=1,
        )
    )
    try:
        query = embeddings.embed_query("How are local embeddings configured?")
        documents = embeddings.embed_documents(["Local embeddings run on this host."])
    finally:
        embeddings.close()

    assert len(query) == dimensions
    assert len(documents) == 1
    assert len(documents[0]) == dimensions
    assert all(math.isfinite(value) for value in query)
    assert all(math.isfinite(value) for value in documents[0])
    assert math.isclose(
        math.sqrt(sum(value * value for value in query)), 1.0, abs_tol=1e-5
    )
    assert math.isclose(
        math.sqrt(sum(value * value for value in documents[0])), 1.0, abs_tol=1e-5
    )
