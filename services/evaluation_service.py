import random
from typing import List, Dict, Any
from services.knowledge_base import KnowledgeBase
from services.llm_service import client, MODEL

class EvaluationService:
    @staticmethod
    def calculate_metrics(retrieved_results: List[Dict[str, Any]], ground_truth_text: str, top_k: int = 5) -> Dict[str, float]:
        results = retrieved_results[:top_k]
        hits = 0
        first_relevant_rank = None
        for rank, res in enumerate(results, start=1):
            res_text = res.get("text", "")
            if ground_truth_text.strip() in res_text or res_text.strip() in ground_truth_text:
                hits += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank
        precision = hits / top_k
        recall = 1.0 if hits > 0 else 0.0
        hit_rate = 1.0 if hits > 0 else 0.0
        mrr = 1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        return {"precision": precision, "recall": recall, "hit_rate": hit_rate, "mrr": mrr}

    @staticmethod
    def generate_synthetic_test_cases(kb: KnowledgeBase, num_cases: int = 5) -> List[Dict[str, Any]]:
        all_child_records = kb.vector_records
        if not all_child_records:
            return []
        num_cases = min(num_cases, len(all_child_records))
        selected_records = random.sample(all_child_records, num_cases)
        test_cases = []
        for idx, rec in enumerate(selected_records, start=1):
            text = rec["text"]
            chunk_id = rec["chunk"].id
            prompt = f"Given the text segment below, generate exactly one direct, search-style question that can be answered solely using the information in this segment. Return ONLY the generated question. Do not include markdown, explanations, or quotes.\n\nSegment:\n{text}"
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": "You are a QA dataset generation assistant. You output only raw question text."},
                        {"role": "user", "content": prompt}
                    ]
                )
                question = response.choices[0].message.content.strip()
                if question:
                    test_cases.append({"id": idx, "question": question, "ground_truth_text": text, "chunk_id": chunk_id})
            except Exception as e:
                print(f"Error generating test case {idx}: {e}")
        return test_cases

    @staticmethod
    def evaluate_configuration(kb: KnowledgeBase, test_cases: List[Dict[str, Any]], config_overrides: Dict[str, Any], top_k: int = 5) -> Dict[str, float]:
        import config
        original_values = {key: getattr(config, key) for key in config_overrides}
        for key, val in config_overrides.items():
            setattr(config, key, val)
        total_metrics = {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "mrr": 0.0}
        for case in test_cases:
            retrieved = kb.retrieve(case["question"], top_k=top_k)
            metrics = EvaluationService.calculate_metrics(retrieved, case["ground_truth_text"], top_k=top_k)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
        for key, val in original_values.items():
            setattr(config, key, val)
        return {key: val / max(len(test_cases), 1) for key, val in total_metrics.items()}
