import os
import json
import argparse
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    ResponseRelevancy,
    FactualCorrectness,
    SemanticSimilarity,
    BleuScore,
    RougeScore
)

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Para múltiplos ROUGE
rouge_combinations = {
    "rouge1": ["precision", "recall", "fmeasure"],
    "rouge2": ["precision", "recall", "fmeasure"],
    "rougeL": ["precision", "recall", "fmeasure"],
}


def init_metric_sums(selected_metrics):
    sums = {}
    for m in selected_metrics:
        if m == "rouge":
            # Adiciona submétricas de rouge
            for rtype, modes in rouge_combinations.items():
                for mode in modes:
                    key = f"{rtype}_{mode}"
                    sums[key] = 0.0
        else:
            sums[m] = 0.0
    return sums


def average_metric_sums(metric_sums, count_items):
    if count_items == 0:
        return {k: 0.0 for k in metric_sums}
    else:
        return {k: v / count_items for k, v in metric_sums.items()}


async def evaluate_metrics(
        data,
        question_col="question_text",
        reference_col="answer",
        candidate_col="candidate",
        metrics=None,
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",  # ou text-embedding-3-large / text-embedding-ada-002
        output_json="all_results.json"
):
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=llm_model, temperature=0)
    )
    evaluator_embedding = OpenAIEmbeddings(model=embedding_model)
    embeddings_wrapper = LangchainEmbeddingsWrapper(evaluator_embedding)

    scorer_answer_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=embeddings_wrapper)
    scorer_answer_correctness = FactualCorrectness(llm=evaluator_llm)
    scorer_semantic_similarity = SemanticSimilarity(embeddings=embeddings_wrapper)
    bleu_scorer = BleuScore()

    metric_sums = init_metric_sums(metrics)
    count_items = 0
    total_items = len(data)
    item_results = []

    for idx, item in enumerate(tqdm(data, desc="Processando itens", total=total_items), start=1):
        q_number = item.get("question_number", str(idx))
        q_text = item.get("question_text", item.get(question_col, ""))

        reference = item.get(reference_col, "")
        candidate = item.get(candidate_col, "")

        if not str(candidate).strip():
            tqdm.write(f"[{idx}/{total_items}] Questão {q_number} ignorada (candidate vazio).")
            continue

        sample = SingleTurnSample(
            user_input=q_text,
            response=str(candidate),
            reference=str(reference)
        )

        result_item = {
            "question_number": q_number,
            "question_text": q_text,
            "reference": reference,
            "candidate": candidate
        }

        if "answer_relevancy" in metrics:
            relevancy_score = await scorer_answer_relevancy.single_turn_ascore(sample)
            result_item["answer_relevancy"] = relevancy_score
            metric_sums["answer_relevancy"] += relevancy_score

        if "answer_correctness" in metrics:
            correctness_score = await scorer_answer_correctness.single_turn_ascore(sample)
            result_item["answer_correctness"] = correctness_score
            metric_sums["answer_correctness"] += correctness_score

        if "semantic_similarity" in metrics:
            semsim_score = await scorer_semantic_similarity.single_turn_ascore(sample)
            result_item["semantic_similarity"] = semsim_score
            metric_sums["semantic_similarity"] += semsim_score

        if "bleu" in metrics:
            bleu_score_val = await bleu_scorer.single_turn_ascore(sample)
            result_item["bleu"] = bleu_score_val
            metric_sums["bleu"] += bleu_score_val

        if "rouge" in metrics:
            for rouge_type, modes in rouge_combinations.items():
                for mode in modes:
                    scorer = RougeScore(rouge_type=rouge_type, mode=mode)
                    rouge_val = await scorer.single_turn_ascore(sample)
                    column_name = f"{rouge_type}_{mode}"
                    result_item[column_name] = rouge_val
                    metric_sums[column_name] += rouge_val

        item_results.append(result_item)
        count_items += 1

        individual_filename = os.path.join(results_dir, f"answer_{q_number}.json")
        with open(individual_filename, "w", encoding="utf-8") as f:
            json.dump(result_item, f, indent=2, ensure_ascii=False)

        tqdm.write(f"[{idx}/{total_items}] Questão {q_number} processada e salva em {individual_filename}.")

    final_scores = average_metric_sums(metric_sums, count_items)
    output_data = {
        "item_results": item_results,
        "overall_scores": final_scores
    }

    all_results_filename = os.path.join(results_dir, output_json)
    with open(all_results_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    tqdm.write(f"Todos os resultados foram salvos em {all_results_filename}.")
    return output_data


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Avaliar métricas.")
    parser.add_argument("--filename", "-f", type=str, required=True,
                        help="Nome do arquivo JSON de entrada.")
    parser.add_argument("--reference_column", "-r", type=str, default="answer",
                        help="Nome da coluna contendo a ground truth (default=answer).")
    parser.add_argument("--candidate_column", "-c", type=str, default="candidate",
                        help="Nome da coluna contendo a resposta do modelo (default=candidate).")
    parser.add_argument("--question_column", "-q", type=str, default="question_text",
                        help="Nome da coluna contendo a pergunta (default=question_text).")
    parser.add_argument("--metrics", "-m", nargs="+", default=[],
                        help="Lista de métricas a serem avaliadas. Use 'all' para todas.")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                        help="Modelo do LLM (ex: gpt-4o-mini-2024-07-18).")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small",
                        help="Modelo de embeddings (default=text-embedding-3-small).")
    parser.add_argument("--output_json", type=str, default="all_results.json",
                        help="Arquivo de saída para gravar resultados (JSON).")
    parser.add_argument("--start", type=int, default=None,
                        help="Índice inicial (1-indexado) dos itens a serem processados.")
    parser.add_argument("--end", type=int, default=None,
                        help="Índice final (1-indexado) dos itens a serem processados.")

    args = parser.parse_args()

    all_metrics = ["answer_relevancy", "answer_correctness", "semantic_similarity", "bleu", "rouge"]
    if "all" in args.metrics:
        selected_metrics = all_metrics
    else:
        selected_metrics = args.metrics if len(args.metrics) > 0 else all_metrics

    if not os.path.isfile(args.filename):
        print(f"Arquivo {args.filename} não encontrado!")
        return

    with open(args.filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.start is not None or args.end is not None:
        start_idx = args.start if args.start is not None else 1
        end_idx = args.end if args.end is not None else len(data)
        data = data[start_idx - 1:end_idx]
        print(f"Processando itens de {start_idx} até {end_idx}.")

    results = asyncio.run(evaluate_metrics(
        data=data,
        question_col=args.question_column,
        reference_col=args.reference_column,
        candidate_col=args.candidate_column,
        metrics=selected_metrics,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        output_json=args.output_json
    ))

    print("\n=== MÉTRICAS GLOBAIS ===")
    print(json.dumps(results["overall_scores"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
