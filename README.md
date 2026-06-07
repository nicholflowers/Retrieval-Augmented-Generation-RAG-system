# 🔎 RAG System Design & Evaluation

**A proof-of-concept Retrieval Augmented Generation system that answers questions for both engineering and marketing teams, grounding each response in a fixed internal document corpus.**

## 📌 Overview

A tech company wanted a better way for its engineering and marketing teams to search internal documents and get answers, particularly about Generative AI and LLM concepts. This project is a proof of concept evaluating whether a RAG system could meet that need, and whether it answers well enough to justify a full build.

## 🎯 Project Goals

- Implement a RAG pipeline in LangChain over a fixed document collection
- Decide how to serve two audiences with opposite needs from one system. (Engineers, who need detailed answers, and marketing staff, who need high-level ones.)
- Define an evaluation metric that measures how closely the system's answers match a set of labeled "gold" answers
- Experiment with hyperparameters (embeddings, chunk size, prompts, language model) to find the strongest configuration
- Deliver a proof-of-concept recommendation, including risks and limitations

## 🗂️ Dataset

| Source | Description |
| --- | --- |
| Generative AI / LLM research papers | A fixed corpus of arXiv papers on Generative AI and LLM topics, indexed in full into the vector store. |
| 75 gold question/answer pairs | Labeled reference answers for both engineering and marketing personas, used as the evaluation benchmark. |

## 🚀 Implementation Details

The pipeline was built within a fixed toolset to mirror a realistic proof of concept, testing established components rather than novel ones.

### 🔹 Retrieval & Indexing

- Embedded document chunks with the `multi-qa-mpnet-base-dot-v1` Sentence Transformers model and stored them in a Qdrant vector store
- Chunked with LangChain's RecursiveCharacterTextSplitter at 448 to 512 tokens with 50 to 64 tokens of overlap, settling there after smaller chunks (128) lost context continuity and larger ones (768) reduced precision
- Tuned top-k retrieval between 8 and 30: higher k improved recall for broad marketing queries, lower k sharpened focus for technical ones
- Reranked candidates with Cohere `rerank-english-v3.0`, and replaced heavy MMR filtering with soft deduplication and per-source caps after aggressive filtering hurt results

### 🔹 Generation

- Compared an open-source model (Mistral 7B Instruct) against a proprietary one (Cohere) as the answer generator
- Used Mistral for early, cost-effective iteration, then selected Cohere for final evaluation for its stronger grounding, tone control, and multi-document synthesis

### 🔹 Handling Two Personas

The central design decision was how to serve two audiences with opposite needs from one system: engineers wanting detailed, technical answers and marketing wanting simplified, high-level ones. Rather than build and maintain two separate pipelines, I kept a single shared retrieval layer and differentiated the personas entirely through prompt design, which kept the system simple while still meeting both teams' needs. Where retrieval trade-offs arose, such as chunk size and top-k, I prioritized technical accuracy for the engineering persona and shaped clarity for marketing through the prompt rather than the retrieval. Both prompts were constrained to answer only from the retrieved context and to say "Not enough information" when the context did not support an answer.

### 🔹 Evaluation

- Combined two metrics: an LLM-as-a-Judge score for semantic correctness and completeness, and RAGAS context precision for retrieval quality
- Chose LLM-as-a-Judge over plain cosine similarity, which does not capture deep semantic correctness, and paired it with precision so retrieval failures could be separated from generation failures
- Scored the engineering and marketing personas separately and ran the best configuration across all 75 gold questions
- RAGAS context precision proved token-heavy at scale, since it issues an LLM call per retrieved chunk, so it was used for focused comparisons rather than the full 75-question batch, which relied on the LLM-as-a-Judge score. The trade-off is real: precision gives a sharper read on retrieval quality but costs too much to run on every evaluation.

## 🏆 Results

- The strongest configuration paired `multi-qa-mpnet-base-dot-v1` embeddings, 448 to 512 token chunks, Cohere reranking, and tuned top-k retrieval with the dual-persona prompts.
- Cohere was selected as the final generation model over Mistral, trading the open-source model's lower cost for noticeably better grounding, tone control, and synthesis across multiple documents.
- The two-metric framework proved its worth during tuning: tracking retrieval precision separately from answer quality made it clear when a weak answer came from poor retrieval versus poor generation, which is what drove the chunking and reranking decisions.

## 🧠 Skills Demonstrated

- **RAG pipeline construction:** assembling embeddings, chunking, a vector store, a retriever, and an LLM into a working LangChain chain
- **Evaluation design:** combining an LLM-as-a-Judge score with RAGAS retrieval precision to measure answer quality and retrieval quality independently
- **Hyperparameter experimentation:** reasoning about how embeddings, chunk size, prompts, and model choice affect retrieval and answer quality
- **Stakeholder communication:** framing a build decision through a proof-of-concept report covering performance, risks, and limitations

## 🧰 Stack

LangChain, Mistral 7B Instruct, Cohere (generation and reranking), Sentence Transformers, Qdrant, and RAGAS.
