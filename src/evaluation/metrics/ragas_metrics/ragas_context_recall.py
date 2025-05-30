import math
from ragas import evaluate, RunConfig
from ragas.metrics import context_recall


from evaluation.metrics.data_types import GradingResult
from evaluation.metrics.utils import to_dataset
from utils import llmaaj_chat_client, llmaaj_embedding_client
from utils import time_function


# def ragas_context_answer_similarity(input, output, reference, metadata, expected) -> float:
# def get_assert(output: str, context) -> Union[bool, float, Dict[str, Any]]:
@time_function
def get_assert(output: str, context) -> GradingResult:
    eval_dataset = to_dataset(output=output, context=context)

    result = evaluate(
        eval_dataset,
        metrics=[context_recall],
        llm=llmaaj_chat_client,
        embeddings=llmaaj_embedding_client,
        run_config=RunConfig(max_workers=64),
    ).to_pandas()
    # 'score': result['answer_similarity'],

    score = float(result["context_recall"])
    # threshold = context["test"]["metadata"]["threshold_ragas_as"]
    threshold = 0

    if math.isnan(score):
        score = 0.0

    return {
        "pass": score > threshold,
        "score": score,
        "reason": f"{score} > {threshold} = {score > threshold}",
    }


if __name__ == "__main__":
    x = get_assert("blop", {"vars": {"ground_truth": "blop"}})

    print("XXXX:", x)
