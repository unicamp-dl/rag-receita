# Document parsing for questions and references extraction 

This repository contains the source code used to extract the questions and the legislatory references supporting the answers from the [2024 Personal Income Tax Declaration Questions and Ansers](https://www.gov.br/receitafederal/pt-br/centrais-de-conteudo/publicacoes/perguntas-e-respostas/dirpf/pr-irpf-2024.pdf/view) published by the Brazilian Federal Revenue Service (Receita Federal do Brasil ― RFB), as described in the research paper [BR-TaxQA-R: A Dataset for Question Answering with References for Brazilian Personal Income Tax Law, including case law](https://arxiv.org/abs/2505.15916).

## Files description

The files captures the whole process to extract the data from the original document, including some steps to fix some bugs encountered during the development. The final version of the BR-TaxQA-R shall be referred in its [Hugging Face repository](https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R).

The notebooks are listed in the execution order:

* [extract_IRPF_QnA.ipynb](extract_IRPF_QnA.ipynb): extracts the questions, the explicit linked questions and external references.
* [extract_embedded_references.ipynb](extract_embedded_references.ipynb): extracts the implicit linked questions and external references.
* [documents_new_normalization.ipynb](documents_new_normalization.ipynb): consolidates the external references, removing duplicates (same documents with different references).
* [check_RAG.ipynb](check_RAG.ipynb): Fixing parsing issue ― avoid truncating multiple-line questions, part 1/2.
* [verify_wrong_questions.ipynb](verify_wrong_questions.ipynb): Fixing parsing issue ― avoid truncating multiple-line questions part 2/2.
* [RFB_QnA_parsing.py](RFB_QnA_parsing.py): functions to support the document parsing.
* [llm_access.py](llm_access.py): LLM-supported functions: legal references formatting; legal references deduplication; embedded legal references extraction.
* [references_normalization.py](references_normalization.py): auxiliary functions to match the downloaded references file names to the normalized external references, as extracted from the original Q&A document.

