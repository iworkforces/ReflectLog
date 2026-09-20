"""Type stubs for sentence_transformers library.

This module provides type annotations for the sentence-transformers library,
specifically the CrossEncoder class used for reranking.

Reference: https://www.sbert.net/docs/cross_encoder/
"""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

class SentenceTransformer:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None = None,
        trust_remote_code: bool = False,
    ) -> None: ...
    def encode_query(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        truncate_dim: int | None = None,
    ) -> NDArray[np.float32]: ...
    def encode_document(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        truncate_dim: int | None = None,
    ) -> NDArray[np.float32]: ...

class CrossEncoder:
    """Cross-encoder model for computing similarity scores between sentence pairs.

    Cross-encoders jointly encode query and document together, enabling
    deeper semantic comparison than bi-encoder similarity.

    Example:
        ```python
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("BAAI/bge-reranker-v2-m3")

        # Score query-document pairs
        pairs = [
            ["What is Python?", "Python is a programming language."],
            ["What is Python?", "The weather is nice today."],
        ]
        scores = model.predict(pairs)
        # scores: [0.95, 0.12]
        ```
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int | None = None,
        max_length: int | None = None,
        device: str | None = None,
        tokenizer_args: dict[str, object] | None = None,
        automodel_args: dict[str, object] | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        default_activation_function: Callable[[object], object] | None = None,
        classifier_dropout: float | None = None,
    ) -> None:
        """Initialize a CrossEncoder model.

        Args:
            model_name: Hugging Face model identifier or path to local model.
            num_labels: Number of labels for classification. Default is 1
            for regression.
            max_length: Maximum sequence length. Default uses model's max length.
            device: Device to use ('cpu', 'cuda', 'mps').
            Default auto-detects.
            tokenizer_args: Additional arguments for the tokenizer.
            automodel_args: Additional arguments for the AutoModel.
            trust_remote_code: Whether to trust remote code in the model.
            revision: Specific model revision to use.
            local_files_only: Whether to only use local model files.
            default_activation_function: Activation function for the output.
            classifier_dropout: Dropout probability for classifier.
        """
        ...

    def predict(
        self,
        sentences: Sequence[Sequence[str]] | Sequence[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        num_workers: int = 0,
        activation_fct: Callable[[object], object] | None = None,
        apply_softmax: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> NDArray[np.float32] | list[float]:
        """Compute similarity scores for sentence pairs.

        Args:
            sentences: List of sentence pairs. Each element should be [query, document]
                       or (query, document) tuple.
            batch_size: Batch size for inference. Default is 32.
            show_progress_bar: Whether to show progress bar. Default is None (auto).
            num_workers: Number of workers for data loading. Default is 0.
            activation_fct: Activation function to apply. Default is None.
            apply_softmax: Whether to apply softmax to output. Default is False.
            convert_to_numpy: Whether to convert output to numpy array. Default is True.
            convert_to_tensor: Whether to convert output to tensor. Default is False.

        Returns:
            Array of similarity scores, one per sentence pair.
            Higher scores indicate more similar/relevant pairs.
        """
        ...

    def rank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int | None = None,
        return_documents: bool = False,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        num_workers: int = 0,
        activation_fct: Callable[[object], object] | None = None,
        apply_softmax: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ) -> list[dict[str, object]]:
        """Rank documents by relevance to a query.

        Args:
            query: The query string.
            documents: List of documents to rank.
            top_k: Return only top_k results. Default is None (return all).
            return_documents: Whether to include document text in results.
            batch_size: Batch size for inference.
            show_progress_bar: Whether to show progress bar.
            num_workers: Number of workers for data loading.
            activation_fct: Activation function to apply.
            apply_softmax: Whether to apply softmax to output.
            convert_to_numpy: Whether to convert output to numpy array.
            convert_to_tensor: Whether to convert output to tensor.

        Returns:
            List of dictionaries with 'corpus_id' and 'score' keys,
            sorted by score descending. If return_documents is True,
            also includes 'text' key.
        """
        ...
