import os
from dotenv import load_dotenv

load_dotenv()

from services.knowledge_base import KnowledgeBase
from services.evaluation_service import EvaluationService

def run_benchmarks():
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set.")
        return
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        print(f"Error: {readme_path} not found.")
        return
    with open(readme_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()
    kb = KnowledgeBase()
    kb.add_document(text=corpus_text, metadata={"source": "pdf", "title": "Lumixa Readme", "file": "README.md"})
    test_cases = EvaluationService.generate_synthetic_test_cases(kb, num_cases=5)
    if not test_cases:
        print("Failed to generate test cases.")
        return
    configs = {
        "1. Baseline Vector Search": {
            "ENABLE_HYBRID_SEARCH": False,
            "ENABLE_QUERY_EXPANSION": False,
            "ENABLE_RERANKING": False,
            "RETRIEVAL_STRATEGY": "standard"
        },
        "2. Hybrid Search (FAISS + BM25)": {
            "ENABLE_HYBRID_SEARCH": True,
            "ENABLE_QUERY_EXPANSION": False,
            "ENABLE_RERANKING": False,
            "RETRIEVAL_STRATEGY": "standard"
        },
        "3. Hybrid + Expansion + Reranking": {
            "ENABLE_HYBRID_SEARCH": True,
            "ENABLE_QUERY_EXPANSION": True,
            "ENABLE_RERANKING": True,
            "RETRIEVAL_STRATEGY": "standard"
        },
        "4. Full Pipeline (Parent-Child Strategy)": {
            "ENABLE_HYBRID_SEARCH": True,
            "ENABLE_QUERY_EXPANSION": True,
            "ENABLE_RERANKING": True,
            "RETRIEVAL_STRATEGY": "parent_child"
        }
    }
    results = {}
    top_k = 3
    for name, overrides in configs.items():
        eval_kb = KnowledgeBase()
        import config
        old_strategy = config.RETRIEVAL_STRATEGY
        config.RETRIEVAL_STRATEGY = overrides["RETRIEVAL_STRATEGY"]
        eval_kb.add_document(text=corpus_text, metadata={"source": "pdf", "title": "Lumixa Readme", "file": "README.md"})
        config.RETRIEVAL_STRATEGY = old_strategy
        metrics = EvaluationService.evaluate_configuration(eval_kb, test_cases, overrides, top_k=top_k)
        results[name] = metrics
    print("\n" + "="*80)
    print(f" BENCHMARK COMPARISON RESULTS (Top-K = {top_k})")
    print("="*80)
    print(f"{'Configuration Name':<42} | {'Prec@K':<7} | {'Recall@K':<8} | {'MRR':<5} | {'Hit Rate':<8}")
    print("-"*80)
    for name, metrics in results.items():
        print(f"{name:<42} | {metrics['precision']:.3f}   | {metrics['recall']:.3f}    | {metrics['mrr']:.3f} | {metrics['hit_rate']:.3f}")
    print("="*80)

if __name__ == "__main__":
    run_benchmarks()
