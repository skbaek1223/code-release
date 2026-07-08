"""Drop-in monkey-patch: replaces prompts.get_extractor_instruction with a
multi-hop-friendly variant.

Changes vs the original:
  - Framing: "for answering the question" -> "to advance the reasoning toward
    the question"
  - Adds an explicit allowance for intermediate bridge facts that are useful
    for the next reasoning step but do not directly resolve the original
    question.
  - Softens the strict-rejection bias: NONE is reserved for the case where the
    documents contain neither a direct answer nor any usable intermediate fact.

Used only by run_re_guide_extractor_fix.py for the small-scale validation run.
The main pipeline's prompts.py is left untouched so existing main-table and
ablation runs remain reproducible.
"""

import prompts as _orig_prompts


def get_extractor_instruction(question, recent_reasoning, search_query, documents):
    return f"""You are an extraction assistant.

Read the retrieved documents and extract information relevant to the current information need reflected in the search query and recent reasoning, **to advance the reasoning toward** answering the question.

Rules:
- Use only the retrieved documents.
- Do not speculate or use outside knowledge.
- Keep factual details as written in the documents.
- Intermediate bridge facts (e.g., the identity of an entity that connects the question to a later retrieval step) are useful and should be extracted, even when they do not directly answer the original question.
- Output NONE only when the documents contain neither a direct answer nor any intermediate fact usable for the next reasoning step.

Output format:
- If the retrieved documents contain helpful information (direct or intermediate) for the current search query or recent reasoning, provide the information with `**Extracted Information**` as shown below.

**Extracted Information**

[Extracted information]

- If the retrieved documents do not contain any helpful information (direct or intermediate), output the following:

**Extracted Information**

NONE

Question: {question}
Current Search Query: {search_query}
Recent Reasoning: {recent_reasoning}
Retrieved Documents: {documents}

Now you should read the retrieved documents and find helpful information based on the current search query "{search_query}" and recent reasoning, including any intermediate facts that could support the next reasoning step.
"""


# Patch the prompts module so any importer that already grabbed a reference
# to get_extractor_instruction sees the new version.
_orig_prompts.get_extractor_instruction = get_extractor_instruction
