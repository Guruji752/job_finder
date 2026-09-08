"""
DeepEval evaluation of the profile-digest RAG step — adds two metrics RAGAS
doesn't give us (see ragas_eval.py for the RAGAS-parity metrics):

- ContextualRelevancyMetric: reference-free, judges whether the RETRIEVED
  CONTEXT itself is relevant to the query — a different angle than RAGAS's
  context_precision, which judges retrieval usefulness relative to the final
  answer. Useful given ragas_eval.py found context_precision=0.30 (lots of
  retrieval noise) — this metric isolates whether that noise is a retrieval
  problem specifically.
- GEval (custom criterion): targets the exact hallucination pattern found in
  Q4 during the RAGAS run — the RAG inventing per-role job titles and
  presenting them as fact via "(implied, but not explicitly stated)" hedging,
  when the resume only lists ONE title anywhere. Faithfulness (RAGAS) caught
  this generically (0.44-0.60); this metric checks specifically for it.

Reuses QUESTIONS and REFERENCE_ANSWERS from ragas_eval.py — same dataset,
same ground truth, no duplication.

Standalone script, not wired into the running app:
    uv run python -m app.eval.deepeval_eval
"""
from deepeval.metrics import ContextualRelevancyMetric, GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from app.config import settings
from app.eval.ragas_eval import REFERENCE_ANSWERS
from app.llm.client import get_llm_client
from app.profile.digest import QUESTIONS
from app.rag.client import RAGClient


class HFDeepEvalLLM(DeepEvalBaseLLM):
    """Wraps the existing sync HF client with plain-text generation — unlike
    RAGAS's Instructor-based wrapper, DeepEval just wants a string back, so
    this sidesteps the response_format/Mode.JSON incompatibility entirely."""

    def load_model(self):
        return get_llm_client()

    def generate(self, prompt: str) -> str:
        completion = self.load_model().chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return settings.chat_model


HALLUCINATED_INFERENCE_CRITERIA = (
    "The actual output should ONLY state facts that are explicitly present in "
    "the input/context. If the output presents a guess, assumption, or inferred "
    "detail as if it were a stated fact — including hedged language like "
    "'implied, but not explicitly stated' or 'a common progression' used to "
    "justify a claim — that counts as a violation, even if the hedge itself is "
    "present. A high score means the output stuck to what the source actually "
    "says; a low score means it fabricated or guessed details."
)


def run_eval():
    llm = HFDeepEvalLLM()
    contextual_relevancy = ContextualRelevancyMetric(model=llm)
    no_hallucinated_inference = GEval(
        name="NoHallucinatedInference",
        criteria=HALLUCINATED_INFERENCE_CRITERIA,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=llm,
    )

    rag_client = RAGClient()
    results = []

    for question in QUESTIONS:
        response = rag_client.query(question)
        print(f"\n=== {question} ===")
        print(f"answer: {response.answer[:300]}")

        test_case = LLMTestCase(
            input=question,
            actual_output=response.answer,
            retrieval_context=response.sources,
            expected_output=REFERENCE_ANSWERS.get(question),
        )

        contextual_relevancy.measure(test_case)
        no_hallucinated_inference.measure(test_case)

        print(f"contextual_relevancy: {contextual_relevancy.score:.2f} — {contextual_relevancy.reason}")
        print(f"no_hallucinated_inference: {no_hallucinated_inference.score:.2f} — {no_hallucinated_inference.reason}")

        results.append({
            "question": question,
            "contextual_relevancy": contextual_relevancy.score,
            "no_hallucinated_inference": no_hallucinated_inference.score,
        })

    print("\n=== AGGREGATE ===")
    for metric in ["contextual_relevancy", "no_hallucinated_inference"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        if scores:
            print(f"{metric}: avg={sum(scores) / len(scores):.2f} (n={len(scores)})")


if __name__ == "__main__":
    run_eval()
