"""Deterministic tests for the local Tencent WeMM embedding adapter."""

import math
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reflectlog.core.enums import WeMMDevice, WeMMModel
from reflectlog.infrastructure.embeddings.wemm_embedding import (
    WeMMEmbeddingConfig,
    WeMMEmbeddings,
)


def _config(*, dimensions: int = 64) -> WeMMEmbeddingConfig:
    return WeMMEmbeddingConfig(
        model=WeMMModel.EMBEDDING_2B,
        dimensions=dimensions,
        device=WeMMDevice.CPU,
        batch_size=2,
    )


def _vector_rows(count: int, dimensions: int) -> np.ndarray:
    return np.full((count, dimensions), 1.0 / math.sqrt(dimensions), dtype=np.float32)


def test_model_load_is_lazy_and_uses_official_sentence_transformer_contract() -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        embeddings = WeMMEmbeddings(_config())
        model_class.assert_not_called()
        model_class.return_value.encode_query.return_value = _vector_rows(1, 64)

        result = embeddings.embed_query("query text")

    model_class.assert_called_once_with(
        WeMMModel.EMBEDDING_2B.value,
        trust_remote_code=True,
        device="cpu",
    )
    model_class.return_value.encode_query.assert_called_once_with(
        ["query text"],
        batch_size=2,
        normalize_embeddings=True,
        convert_to_numpy=True,
        truncate_dim=64,
    )
    assert result == pytest.approx([0.125] * 64)
    assert all(type(value) is float for value in result)


def test_documents_use_encode_document_and_empty_batch_does_not_load() -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        embeddings = WeMMEmbeddings(_config())
        assert embeddings.embed_documents([]) == []
        model_class.assert_not_called()
        model_class.return_value.encode_document.return_value = _vector_rows(2, 64)

        result = embeddings.embed_documents(["first", "second"])

    model_class.return_value.encode_document.assert_called_once_with(
        ["first", "second"],
        batch_size=2,
        normalize_embeddings=True,
        convert_to_numpy=True,
        truncate_dim=64,
    )
    assert len(result) == 2
    assert all(len(vector) == 64 for vector in result)


def test_query_normalizes_low_precision_model_output() -> None:
    rounded = np.full((1, 2048), 1.0 / math.sqrt(2048), dtype=np.float16)
    assert abs(math.sqrt(sum(float(value) ** 2 for value in rounded[0])) - 1.0) > 1e-5
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        model_class.return_value.encode_query.return_value = rounded
        result = WeMMEmbeddings(_config(dimensions=2048)).embed_query("query")

    assert len(result) == 2048
    assert all(type(value) is float and value > 0 for value in result)
    assert math.sqrt(sum(value * value for value in result)) == pytest.approx(1.0)


def test_documents_normalize_low_precision_model_output() -> None:
    rounded = np.full((2, 2048), 1.0 / math.sqrt(2048), dtype=np.float16)
    rounded[1] *= -1
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        model_class.return_value.encode_document.return_value = rounded
        result = WeMMEmbeddings(_config(dimensions=2048)).embed_documents(
            ["first", "second"]
        )

    assert len(result) == 2
    assert all(len(vector) == 2048 for vector in result)
    assert all(type(value) is float and value > 0 for value in result[0])
    assert all(type(value) is float and value < 0 for value in result[1])
    assert all(
        math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)
        for vector in result
    )


def test_query_normalizes_finite_nonunit_float32_output() -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        model_class.return_value.encode_query.return_value = np.ones(
            (1, 64), dtype=np.float32
        )
        result = WeMMEmbeddings(_config()).embed_query("query")

    assert result == pytest.approx([0.125] * 64)


def test_documents_reject_zero_vector() -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        model_class.return_value.encode_document.return_value = np.zeros(
            (1, 64), dtype=np.float16
        )
        embeddings = WeMMEmbeddings(_config())

        with pytest.raises(RuntimeError, match="nonzero"):
            embeddings.embed_documents(["document"])


@pytest.mark.parametrize("payload", [123, {"text": "query"}, ["query"]])
def test_query_rejects_non_text_before_model_load(payload: object) -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        embeddings = WeMMEmbeddings(_config())

        with pytest.raises(TypeError, match="query must be text"):
            MagicMock(wraps=embeddings.embed_query)(payload)

    model_class.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    ["single document", ("first", "second"), ["first", 2]],
)
def test_documents_reject_non_text_batches_before_model_load(payload: object) -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        embeddings = WeMMEmbeddings(_config())

        with pytest.raises(TypeError, match="documents must be a list of text"):
            MagicMock(wraps=embeddings.embed_documents)(payload)

    model_class.assert_not_called()


@pytest.mark.parametrize(
    ("device", "constructor_device"),
    [
        (WeMMDevice.AUTO, None),
        (WeMMDevice.CPU, "cpu"),
        (WeMMDevice.CUDA, "cuda"),
        (WeMMDevice.MPS, "mps"),
    ],
)
def test_device_selection_is_forwarded(
    device: WeMMDevice, constructor_device: str | None
) -> None:
    config = WeMMEmbeddingConfig(
        model=WeMMModel.EMBEDDING_2B,
        dimensions=64,
        device=device,
        batch_size=1,
    )
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        model_class.return_value.encode_query.return_value = _vector_rows(1, 64)
        WeMMEmbeddings(config).embed_query("query")

    model_class.assert_called_once_with(
        WeMMModel.EMBEDDING_2B.value,
        trust_remote_code=True,
        device=constructor_device,
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (np.ones((0, 64), dtype=np.float32), "batch count"),
        (np.ones((1, 63), dtype=np.float32), "width"),
        (np.empty((1, 0), dtype=np.float32), "empty"),
        (np.full((1, 64), np.nan, dtype=np.float32), "finite"),
        (np.full((1, 64), np.inf, dtype=np.float32), "finite"),
        (np.zeros((1, 64), dtype=np.float32), "nonzero"),
        (np.ones(64, dtype=np.float32), "matrix"),
    ],
)
def test_query_output_validation_fails_closed(
    payload: np.ndarray, message: str
) -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        model_class.return_value.encode_query.return_value = payload
        embeddings = WeMMEmbeddings(_config())

        with pytest.raises(RuntimeError, match=message):
            embeddings.embed_query("query")


def test_document_count_mismatch_fails_closed() -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        model_class.return_value.encode_document.return_value = _vector_rows(1, 64)
        embeddings = WeMMEmbeddings(_config())

        with pytest.raises(RuntimeError, match="batch count"):
            embeddings.embed_documents(["first", "second"])


def test_downstream_runtime_failure_is_wrapped_with_cause() -> None:
    cause = RuntimeError("device failure")
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        model_class.return_value.encode_query.side_effect = cause
        embeddings = WeMMEmbeddings(_config())

        with pytest.raises(RuntimeError, match="WeMM query inference failed") as exc:
            embeddings.embed_query("query")

    assert exc.value.__cause__ is cause


def test_programming_error_is_not_wrapped() -> None:
    cause = TypeError("bad invocation")
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        model_class.return_value.encode_query.side_effect = cause
        embeddings = WeMMEmbeddings(_config())

        with pytest.raises(TypeError, match="bad invocation"):
            embeddings.embed_query("query")


async def test_async_methods_offload_the_sync_contract() -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        model_class.return_value.encode_query.return_value = _vector_rows(1, 64)
        model_class.return_value.encode_document.return_value = _vector_rows(2, 64)
        embeddings = WeMMEmbeddings(_config())

        query = await embeddings.aembed_query("query")
        documents = await embeddings.aembed_documents(["first", "second"])

    assert len(query) == 64
    assert len(documents) == 2


def test_concurrent_first_use_loads_model_once() -> None:
    entered = threading.Event()
    release = threading.Event()
    model = MagicMock()
    model.encode_query.return_value = _vector_rows(1, 64)

    def construct_model(*_args: str, **_kwargs: str) -> MagicMock:
        entered.set()
        assert release.wait(timeout=2)
        return model

    embeddings = WeMMEmbeddings(_config())
    results: list[list[float]] = []
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer",
        side_effect=construct_model,
    ) as model_class:
        threads = [
            threading.Thread(target=lambda: results.append(embeddings.embed_query("q")))
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        assert entered.wait(timeout=2)
        release.set()
        for thread in threads:
            thread.join(timeout=3)

    assert len(results) == 4
    model_class.assert_called_once()


def test_inference_is_serialized_with_model_lifecycle() -> None:
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    call_lock = threading.Lock()
    call_count = 0
    model = MagicMock()

    def encode_query(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal call_count
        with call_lock:
            call_count += 1
            current = call_count
        if current == 1:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        return _vector_rows(1, 64)

    model.encode_query.side_effect = encode_query
    embeddings = WeMMEmbeddings(_config())
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer",
        return_value=model,
    ):
        first = threading.Thread(target=embeddings.embed_query, args=("first",))
        second = threading.Thread(target=embeddings.embed_query, args=("second",))
        first.start()
        assert first_entered.wait(timeout=2)
        second.start()
        assert not second_entered.wait(timeout=0.1)
        release_first.set()
        first.join(timeout=3)
        second.join(timeout=3)

    assert second_entered.is_set()


def test_close_clears_loaded_model_and_is_idempotent() -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        model_class.return_value.encode_query.return_value = _vector_rows(1, 64)
        embeddings = WeMMEmbeddings(_config())
        embeddings.embed_query("query")

        embeddings.close()
        embeddings.close()

    assert embeddings.is_loaded is False


def test_calls_after_close_are_rejected_without_reloading() -> None:
    with patch(
        "reflectlog.infrastructure.embeddings.wemm_embedding.SentenceTransformer"
    ) as model_class:
        embeddings = WeMMEmbeddings(_config())
        embeddings.close()

        with pytest.raises(RuntimeError, match="closed"):
            embeddings.embed_query("query")
        with pytest.raises(RuntimeError, match="closed"):
            embeddings.embed_documents([])

    model_class.assert_not_called()
