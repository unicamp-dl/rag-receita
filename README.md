
---

# Metric Evaluator with RAGAS

This script compares model-generated responses with reference responses (ground truth) using metrics from the RAGAS package.

## Step-by-step usage:

1. **If you have a CSV file, convert it to JSON** (the `evaluate.py` script expects a JSON file as input).  
   Run the `convert_csv_to_json.py` script located in utils folder to convert the data spreadsheet:

   ```bash
   python convert_csv_to_json.py
   ```

2. **Create a `.env` file**  
   Add your OpenAI key to the `.env` file:
   ```
   OPENAI_API_KEY=your_key_here
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the script to evaluate with all metrics**:

   ```bash
   python evaluate.py --filename data.json --reference_column ground_truth --candidate_column "chatgpt(4o/4mini)+search" --question_column question_text --metrics all --llm_model llm_model --embedding_model embedding_model --output_json "all_results.json"
   ```

5. **Example with specific metrics**:
   ```bash
   python evaluate.py \
     --filename data.json \
     --reference_column ground_truth \
     --candidate_column model_response \
     --question_column question \
     --metrics answer_relevancy semantic_similarity bleu \
     --output_json "all_results.json"
   ```

6. **Example with a specific range**:
   ```bash
   python evaluate.py --filename data.json --start 1 --end 50 --output_json "1_50_results.json"
   ```

Individual results are saved in the `results/` folder, and the overall result in `results/all_results.json`.

Tip: If you have evaluated one or a few responses and want to recalculate the overall metric averages, use the `calculate_overall.py` file from the `utils` folder. It will read all the new answers and recalculate the average and save into a new json file.

---
