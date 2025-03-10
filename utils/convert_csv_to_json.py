import csv
import json
import os

def csv_to_json(csv_path, json_path):
    data_list = []
    with open(csv_path, mode='r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            # Ignorar se não tiver resposta na coluna
            candidate_value = row.get("chatgpt(4o/4mini)+search", "")
            if not candidate_value.strip():
                continue

            data_list.append(row)

    with open(json_path, mode='w', encoding='utf-8') as f_out:
        json.dump(data_list, f_out, indent=2, ensure_ascii=False)

def main():
    csv_path = "../files/data.csv"
    json_path = "../files/data.json"
    if not os.path.isfile(csv_path):
        print(f"Arquivo CSV não encontrado em {csv_path}")
        return

    csv_to_json(csv_path, json_path)
    print(f"Conversão concluída! JSON criado em {json_path}")

if __name__ == "__main__":
    main()
