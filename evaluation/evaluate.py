from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

from app.rag_pipeline import (
    hybrid_search,
    rerank_results,
    answer_question,
)

from evaluation.test_cases import test_cases


def run_evaluation():

    prepared_test_cases = []

    for test_case in test_cases:

        print(f"\nEvaluating: {test_case.input}")

        # Step 1: Retrieve relevant documents
        hybrid_results = hybrid_search(
            test_case.input,
            top_k=10
        )

        # Step 2: Rerank retrieved documents
        reranked_results = rerank_results(
            test_case.input,
            hybrid_results,
            top_k=3
        )

        # Step 3: Extract documents for DeepEval
        retrieval_context = [
            document
            for document, score in reranked_results
        ]

        # Step 4: Generate actual answer
        actual_output = answer_question(
            test_case.input
        )

        # Step 5: Create DeepEval test case
        prepared_test_cases.append(
            LLMTestCase(
                input=test_case.input,
                actual_output=actual_output,
                expected_output=test_case.expected_output,
                retrieval_context=retrieval_context,
            )
        )

        print(f"Answer: {actual_output}")

        print(
            f"Retrieved context chunks: "
            f"{len(retrieval_context)}"
        )

    # DeepEval metrics
    metrics = [
        AnswerRelevancyMetric(),
        FaithfulnessMetric(),
        ContextualPrecisionMetric(),
    ]

    print("\nStarting DeepEval evaluation...\n")

    results = evaluate(
        test_cases=prepared_test_cases,
        metrics=metrics,
        async_config=AsyncConfig(
            max_concurrent=1,
            run_async=False
        ),
    )

    print("\nEvaluation completed.")
    print(results)


if __name__ == "__main__":
    run_evaluation()

