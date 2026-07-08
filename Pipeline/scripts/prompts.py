_QA_INSTRUCTION_COMMON = """You are a question answering assistant with the ability to perform searches and reason.

You can receive assistance from the system guide during the retrieval and reasoning process.
You must provide an accurate answer to the given question by performing searches to retrieve relevant information, and reasoning over the retrieved information to derive the answer.

To perform a search, write a search query in the format <|begin_search_query|> your query here <|end_search_query|>, written as concise keywords only.
Then, the system will then search for relevant content, and return helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.
The system will also provide guidance on whether the search result is sufficient, and on the direction your reasoning should take next, in the format [Reasoning Guide].

The final answer must be in the format \\boxed{YOUR_ANSWER}.
"""

SINGLE_HOP_EXAMPLE = """

Here is an example of a single-hop question that requires one retrieval step:

Question:
who holds the most women's wimbledon titles

You may refer to the retrieval guide below to inform your search strategy, or feel free to take a different approach.

Retrieval Guide:
Find the record holder for the most women's singles titles at Wimbledon.

Following the retrieval guide, I need to find who holds the record for the most women's singles titles at Wimbledon.
<|begin_search_query|> most women's Wimbledon singles titles record holder <|end_search_query|>
<|begin_search_result|> Martina Navratilova holds the record for the most women's singles titles at Wimbledon, having won the championship nine times between 1978 and 1990. <|end_search_result|>
[Reasoning Guide]: Sufficient.
You now have enough information to answer the original question "who holds the most women's wimbledon titles". Provide the final answer in the format \\boxed{YOUR_ANSWER}.

The search result clearly states Martina Navratilova holds the record with nine Wimbledon titles.
\\boxed{Martina Navratilova}
"""


MULTI_HOP_EXAMPLE = """

Here is an example of a multi-hop question that requires multiple retrieval steps:

Question:
The Oberoi family is part of a hotel company that has a head office in what city?

You may refer to the retrieval guide below to inform your search strategy, or feel free to take a different approach.

Retrieval Guide:
1. Identify the hotel company the Oberoi family is part of.
2. Find the city where that hotel company's head office is located.

Following step 1 of the retrieval guide, I need to identify the hotel company the Oberoi family is part of.
<|begin_search_query|> Oberoi family hotel company <|end_search_query|>
<|begin_search_result|> The Oberoi family is an Indian family involved in the hospitality industry. They are the founders and owners of The Oberoi Group, which operates luxury hotels and resorts. <|end_search_result|>
[Reasoning Guide]: Sufficient.
1) If there is still more information to retrieve before fully answering the original question "The Oberoi family is part of a hotel company that has a head office in what city?", derive an intermediate answer based on the retrieved information, then continue reasoning toward the next retrieval. You may refer to the reasoning context above or the retrieval guide to inform your search strategy, or feel free to take a different approach.
2) If you have sufficient information to fully answer the original question, provide the final answer in the format \\boxed{YOUR_ANSWER}.

The Oberoi family is associated with The Oberoi Group. I still need to find the head office location, so I'll continue retrieval.
Following step 2 of the retrieval guide, now I need to find where The Oberoi Group's head office is located.
<|begin_search_query|> The Oberoi Group head office location <|end_search_query|>
<|begin_search_result|> The Oberoi Group is a hotel company founded in 1934. It operates luxury hotels, resorts, and cruisers in several countries. The company's head office is located in Delhi, India. <|end_search_result|>
[Reasoning Guide]: Sufficient.
1) If there is still more information to retrieve before fully answering the original question "The Oberoi family is part of a hotel company that has a head office in what city?", derive an intermediate answer based on the retrieved information, then continue reasoning toward the next retrieval. You may refer to the reasoning context above or the retrieval guide to inform your search strategy, or feel free to take a different approach.
2) If you have sufficient information to fully answer the original question, provide the final answer in the format \\boxed{YOUR_ANSWER}.

The Oberoi family owns The Oberoi Group, whose head office is in Delhi.
\\boxed{Delhi}
"""


INSUFFICIENT_RETRY_EXAMPLE = """

Here is an example where the first retrieval is insufficient and a retry with a differently-angled approach is needed:

Question:
when did women get the vote in the uk

You may refer to the retrieval guide below to inform your search strategy, or feel free to take a different approach.

Retrieval Guide:
Find the year women gained the right to vote in the United Kingdom.

Following the retrieval guide, I need to find when women gained voting rights in the UK.
<|begin_search_query|> UK women vote year <|end_search_query|>
<|begin_search_result|> NONE <|end_search_result|>
[Reasoning Guide]: Insufficient (confidence 100%).
Feel free to explore alternative paths, such as trying a different search query or taking different retrieval steps as needed, to derive an answer to the question: when did women get the vote in the uk. Please think carefully, and use up to 153 words.

The retrieval was entirely insufficient, so I should switch angles entirely. Instead of asking the date directly, search for the broader topic of women's suffrage history in the UK, which is more likely to appear in encyclopedic sources.
<|begin_search_query|> women suffrage UK history <|end_search_query|>
"""

def get_single_qa_instruction():
    return _QA_INSTRUCTION_COMMON + SINGLE_HOP_EXAMPLE + INSUFFICIENT_RETRY_EXAMPLE


def get_multi_qa_instruction():
    return _QA_INSTRUCTION_COMMON + MULTI_HOP_EXAMPLE + INSUFFICIENT_RETRY_EXAMPLE

def get_retrieval_evaluator_instruction(
    question,
    recent_reasoning,
    search_query,
    search_result
):
    return f"""You are a retrieval evaluation assistant.

Your task is to determine whether the given search result is sufficient to satisfy the information need expressed in the most recent reasoning to answer the question.

Rules:
1. Judge only based on the provided search result. Do not infer, speculate, or use outside knowledge.
2. Focus only on the information need implied by the recent reasoning, not the whole question.
3. Output <answer> sufficient </answer> only if the search result provides enough explicit information to satisfy that information need without additional search.
4. Output <answer> insufficient </answer> if the search result is incomplete, ambiguous, indirect, weakly relevant, or missing key information needed to satisfy that information need.
5. After that, output a confidence score from 0% to 100% indicating how confident you are in your sufficient/insufficient decision based only on the given search result.

Output format:
<answer> sufficient </answer>
<confidence> 90% </confidence>

or

<answer> insufficient </answer>
<confidence> 85% </confidence>

Do not output any explanation, reasoning, or extra text.

Question: {question}

Recent Reasoning: {recent_reasoning}

Search Query: {search_query}

Search Result: {search_result}
"""

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

