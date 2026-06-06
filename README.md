# 🔎 RAG System Design & Evaluation

**A proof-of-concept retrieval-augmented generation system that answers GenAI questions for two audiences, engineers and marketers, from a fixed internal document collection.**

## 📌 Overview

A tech company wanted a better way for its engineering and marketing teams to search internal documents and get answers, particularly about Generative AI and LLM concepts. This project is a proof of concept evaluating whether a retrieval-augmented generation (RAG) system could meet that need, and whether it answers well enough to justify a full build.

## 🎯 Project Goals

- Implement a RAG pipeline in LangChain over a fixed document collection
- Serve two user personas from one system: engineers, who need detailed answers, and marketing staff, who need high-level ones
- Define an evaluation metric that measures how closely the system's answers match a set of labeled "gold" answers
- Experiment with hyperparameters (embeddings, chunk size, prompts, language model) to find the strongest configuration
- Deliver a proof-of-concept recommendation, including risks and limitations

## 🗂️ Dataset

| Source | Description |
| --- | --- |
| Provided GenAI/LLM document collection | A fixed corpus of technical documents on Generative AI and LLM topics, indexed in full into the vector store. |
| 75 gold question/answer pairs | Labeled reference answers for both engineering and marketing personas, used as the evaluation benchmark. |

## 🚀 Implementation Details

The pipeline was built within a fixed toolset to mirror a realistic proof of concept, testing established components rather than novel ones.

### 🔹 Retrieval & Indexing

- Embedded the document chunks with a Sentence Transformers model and stored them in a Qdrant vector store
- Chunked the documents with LangChain's RecursiveCharacterTextSplitter, with chunk size treated as a key tuning parameter
- Retrieved supporting context by semantic similarity at query time

### 🔹 Generation

- Compared an open-source language model (Mistral 7B Instruct) against a proprietary one (Cohere) as the answer generator
- Each query retrieves relevant context, which is passed to the model to ground its answer

### 🔹 Dual-Persona Prompting

- Wrote separate prompts for the two audiences: detailed, technical responses for engineers and concise, high-level responses for marketing
- The same retrieval layer serves both, with the prompt shaping the depth of the answer

### 🔹 Evaluation

- Scored generated answers against the 75 gold answers using an evaluation metric I defined and justified
- Reported performance for the engineering and marketing personas separately
- Ran the best configuration across all 75 questions

## 🏆 Results

_To complete from the POC report: the winning configuration (embedding model, chunk size, language model), its scores against the gold answers, and the build recommendation with risks and limitations._

## 🧠 Skills Demonstrated

- **RAG pipeline construction:** assembling embeddings, chunking, a vector store, a retriever, and an LLM into a working LangChain chain
- **Evaluation design:** defining and defending a metric to measure generated answers against labeled gold data
- **Hyperparameter experimentation:** reasoning about how embeddings, chunk size, prompts, and model choice affect retrieval and answer quality
- **Stakeholder communication:** framing a build decision through a proof-of-concept report covering performance, risks, and limitations

## 🧰 Stack

Includes but is not limited to LangChain, Mistral 7B Instruct, Cohere, Sentence Transformers, and Qdrant.
