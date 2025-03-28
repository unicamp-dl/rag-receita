import os
import json
import glob

def average_metric_sums(metric_sums, count_items):
    if count_items == 0:
        return {k: 0.0 for k in metric_sums}
    else:
        return {k: v / count_items for k, v in metric_sums.items()}

def main():
    results_dir = os.path.join("..", "results")
    file_pattern = os.path.join(results_dir, "answer_*.json")
    file_list = sorted(
        glob.glob(file_pattern),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
    )

    if not file_list:
        print("Nenhum arquivo answer_*.json encontrado em", results_dir)
        return

    item_results = []
    non_metric_keys = {"question_number", "question_text", "reference", "candidate"}
    metric_sums = {}
    count_items = 0

    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            item_results.append(data)
            count_items += 1
            for key, value in data.items():
                if key not in non_metric_keys and isinstance(value, (int, float)):
                    if key not in metric_sums:
                        metric_sums[key] = 0.0
                    metric_sums[key] += value

    overall_scores = average_metric_sums(metric_sums, count_items)

    output_data = {
        "item_results": item_results,
        "overall_scores": overall_scores
    }

    output_filename = os.path.join(results_dir, "all_results.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Todos os resultados foram salvos em {output_filename}.")

if __name__ == "__main__":
    main()
