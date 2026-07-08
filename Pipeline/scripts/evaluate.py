import re
import json
import math
import numpy as np
from collections import Counter
import string
import os, time
from collections import defaultdict
try:
    from lcb_runner.evaluation import codegen_metrics
except Exception:
    codegen_metrics = None
try:
    from utils.math_equivalence import is_equiv
except Exception:
    def is_equiv(a, b):
        return 0


# Mirror of run_re_guide.py's runaway cap (len(s['output']) > 40_000 →
# early termination). Applied here so that older runs generated without
# the cap are scored as if the cap had been in force: anything past this
# many characters is truncated before answer extraction / metrics.
OUTPUT_CHAR_CAP = 40_000


def _cap_output(text):
    if isinstance(text, str) and len(text) > OUTPUT_CHAR_CAP:
        return text[:OUTPUT_CHAR_CAP]
    return text


def extract_answer(output, mode='gen'):
    extracted_text = ''
    if mode == 'codegen':
        # Extract the code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode == 'infogen':
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        pattern_info = "**Final Information**"
        pattern_step = "**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        # Prefer explicit <answer>...</answer> tags if present.
        tag_matches = re.findall(r'<answer>(.*?)</answer>', output, re.DOTALL | re.IGNORECASE)
        if tag_matches:
            return tag_matches[-1].strip()
        # Existing extraction logic for 'gen' and 'choose' modes
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            if mode in ['choose', 'qa']:
                # Handle 'choose' mode
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]  # Take the last match
                extracted_text = extracted_text.strip("()")
    return extracted_text


def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text

def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.strip().split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def evaluate_predictions(output, labeled_answer, mode='gen'):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}
    pred_answer = extract_answer(output, mode=mode)
    if pred_answer != '':
        final_metric["is_valid_answer"] = True

    if mode == 'qa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    else:
        normalized_pred_answer = normalize_answer(pred_answer)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)
    
        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

    # print(em, acc, f1, normalized_pred_answer, '|', normalized_ground_truth)
    return final_metric, pred_answer



_SEARCH_RESULT_RE = re.compile(
    r'<\|begin_search_result\|>.*?end_search_\w+\|?>', re.DOTALL)
_REASONING_GUIDE_RE = re.compile(
    r'\[Reasoning Guide\]:.*?(?:'
    r'\\boxed\{YOUR_ANSWER\}\.?'
    r'|up to \d+ words\.?'
    r'|find new information\.?'
    r'|\Z'
    r')',
    re.DOTALL)


def _strip_non_model_segments(text):
    text = _SEARCH_RESULT_RE.sub('', text)
    text = _REASONING_GUIDE_RE.sub('', text)
    return text


# --- Rumination metrics (DeepSeek-R1 Thoughtology, Sec. 3.3) ---
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _extract_reasoning_chain(text):
    return _strip_non_model_segments(text)


def _tokenize_words(text):
    return [w.lower() for w in _WORD_RE.findall(text)]


def ngram_repetition_rate(words, n=5):
    """Proportion of n-gram occurrences that are repeats (i.e. not first
    occurrence). With n=5 as in Thoughtology, captures verbatim rumination."""
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    total = len(grams)
    unique = len(set(grams))
    return (total - unique) / total


def normalized_lexical_entropy(words):
    """H_norm(T) = -sum_w p(w) log2 p(w) / log2 N,  p(w) = c(w)/N
    (Equation (1) of Thoughtology). Lower => less lexical diversity =>
    more rumination."""
    N = len(words)
    if N <= 1:
        return 0.0
    counts = Counter(words)
    H = -sum((c / N) * math.log2(c / N) for c in counts.values())
    denom = math.log2(N)
    if denom == 0:
        return 0.0
    return H / denom


def compute_rumination(text, n=5):
    chain = _extract_reasoning_chain(text)
    words = _tokenize_words(chain)
    return {
        'rumination_5gram_rep': ngram_repetition_rate(words, n=n),
        'rumination_lex_entropy': normalized_lexical_entropy(words),
    }


def _aggregate_rumination(rum_list):
    if not rum_list:
        return None
    rep = [r['rumination_5gram_rep'] for r in rum_list]
    ent = [r['rumination_lex_entropy'] for r in rum_list]
    return {
        'rumination_5gram_rep': float(np.mean(rep)),
        'rumination_lex_entropy': float(np.mean(ent)),
    }


def _per_q_output_tokens(output_list, tokenizer):
    """Per-question output token count via cap + strip + tokenize.
    Single source of truth shared by avg_output_tokens and
    module_tokens.reasoning.output."""
    if tokenizer is None or not output_list:
        return None
    counts = []
    for r in output_list:
        text = r if isinstance(r, str) else r.outputs[0].text
        text = _cap_output(text)
        text = _strip_non_model_segments(text)
        counts.append(len(tokenizer(text, add_special_tokens=False)['input_ids']))
    return counts


def _avg_output_tokens(output_list, tokenizer):
    counts = _per_q_output_tokens(output_list, tokenizer)
    if not counts:
        return None
    return sum(counts) / len(counts)


def _summarize_module_tokens(per_q_module_tokens):
    """Convert per-question token lists into per-module averages plus a total.

    Two accepted formats:
      flat (legacy):  {'planner': [int]*N, 'reasoning': [int]*N, ...}
                      -> output-only counts; summary value is a float.
      nested:         {'planner': {'input': [int]*N, 'output': [int]*N}, ...}
                      -> per-module dict {'input', 'output'}; total is a dict.
    Empty/None entries are skipped."""
    if not per_q_module_tokens:
        return None
    summary = {}
    n = None
    has_nested = False
    for k, v in per_q_module_tokens.items():
        if isinstance(v, dict):
            has_nested = True
            in_lst = v.get('input') or []
            out_lst = v.get('output') or []
            if not in_lst and not out_lst:
                continue
            entry = {}
            if in_lst:
                entry['input'] = float(np.mean(in_lst))
                n = len(in_lst) if n is None else n
            if out_lst:
                entry['output'] = float(np.mean(out_lst))
                n = len(out_lst) if n is None else n
            summary[k] = entry
        else:
            if not v:
                continue
            n = len(v) if n is None else n
            summary[k] = float(np.mean(v))
    if not summary:
        return None
    if has_nested:
        total_in = sum(e.get('input', 0.0) for e in summary.values()
                       if isinstance(e, dict))
        total_out = sum(e.get('output', 0.0) for e in summary.values()
                        if isinstance(e, dict))
        summary['total'] = {
            'input': total_in,
            'output': total_out,
            'combined': total_in + total_out,
        }
    else:
        summary['total'] = float(sum(v for v in summary.values()
                                     if isinstance(v, float)))
    summary['_num_questions'] = n
    return summary


def _write_cost_report(output_dir, metrics_json_name, dataset_name, overall_metrics):
    """Write a per-run cost analysis (txt + json) next to the metrics file.

    Reads `overall_metrics['module_tokens']` (populated by
    `_summarize_module_tokens`). The two output files share the same
    timestamp prefix as the metrics file:
        {prefix}.cost.txt
        {prefix}.cost.json
    No-op if module_tokens is missing or the metrics filename does not
    match the standard pattern."""
    mt = overall_metrics.get('module_tokens') if isinstance(overall_metrics, dict) else None
    if not mt:
        return
    if not metrics_json_name.endswith('.metrics.json'):
        return
    base = metrics_json_name[:-len('.metrics.json')]
    txt_path = os.path.join(output_dir, f'{base}.cost.txt')
    json_path = os.path.join(output_dir, f'{base}.cost.json')

    has_planner = 'planner' in mt
    has_infogen = 'infogen' in mt
    if has_planner and not has_infogen:
        method = 're_guide'
    elif has_infogen and not has_planner:
        method = 'search_o1_wiki'
    else:
        method = 'unknown'

    n = mt.get('_num_questions')
    total = mt.get('total')
    nested = isinstance(total, dict)

    lines = []
    lines.append('=' * 78)
    lines.append(f'Cost analysis  dataset={dataset_name}  method={method}')
    lines.append('=' * 78)
    lines.append(f'  num_questions:       {n}')
    lines.append(f'  query_latency:       {overall_metrics.get("query_latency", "-")}')
    lines.append(f'  avg_main_agent_tok:  {overall_metrics.get("avg_output_tokens", "-")}')
    lines.append('')

    if nested:
        total_in = total.get('input', 0.0)
        total_out = total.get('output', 0.0)
        total_comb = total.get('combined', total_in + total_out)
        lines.append('  Per-module avg tokens / question (input + output):')
        lines.append(f'    {"module":<12}{"input":>11}{"output":>11}'
                     f'{"sum":>11}{"% of total":>14}')
        lines.append(f'    {"-"*12:<12}{"-"*11:>11}{"-"*11:>11}'
                     f'{"-"*11:>11}{"-"*14:>14}')
        for k in ('planner', 'reasoning', 'extractor', 'evaluator', 'infogen'):
            entry = mt.get(k)
            if not isinstance(entry, dict):
                continue
            v_in = entry.get('input', 0.0)
            v_out = entry.get('output', 0.0)
            v_sum = v_in + v_out
            pct = (v_sum / total_comb * 100.0) if total_comb else 0.0
            lines.append(f'    {k:<12}{v_in:>11.1f}{v_out:>11.1f}'
                         f'{v_sum:>11.1f}{pct:>13.1f}%')
        lines.append(f'    {"-"*12:<12}{"-"*11:>11}{"-"*11:>11}'
                     f'{"-"*11:>11}{"-"*14:>14}')
        lines.append(f'    {"TOTAL":<12}{total_in:>11.1f}{total_out:>11.1f}'
                     f'{total_comb:>11.1f}{100.0:>13.1f}%')
    else:
        total_v = total or 0.0
        breakdown = []
        for k in ('planner', 'reasoning', 'extractor', 'evaluator', 'infogen'):
            v = mt.get(k)
            if v is not None and not isinstance(v, dict):
                pct = (v / total_v * 100.0) if total_v else 0.0
                breakdown.append((k, v, pct))
        lines.append('  Per-module avg output tokens / question:')
        lines.append(f'    {"module":<14}{"avg_tokens":>14}{"% of total":>14}')
        lines.append(f'    {"-"*14:<14}{"-"*14:>14}{"-"*14:>14}')
        for k, v, pct in breakdown:
            lines.append(f'    {k:<14}{v:>14.1f}{pct:>13.1f}%')
        lines.append(f'    {"-"*14:<14}{"-"*14:>14}{"-"*14:>14}')
        lines.append(f'    {"TOTAL":<14}{total_v:>14.1f}{100.0:>13.1f}%')
    lines.append('')
    lines.append('Notes:')
    if nested:
        lines.append('  - Counts are average tokens per question, split by input (prompt) vs output (generated).')
    else:
        lines.append('  - Counts are average OUTPUT tokens per question (model-generated, not prompt).')
    lines.append('  - Re-Guide TOTAL    = planner + reasoning + extractor + evaluator.')
    lines.append('  - Search-o1 TOTAL   = reasoning + infogen.')
    lines.append('  - avg_main_agent_tok counts only the primary agent\'s reasoning tokens.')
    lines.append('=' * 78)
    body = '\n'.join(lines) + '\n'

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(body)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': dataset_name,
            'method': method,
            'num_questions': n,
            'query_latency': overall_metrics.get('query_latency'),
            'avg_main_agent_tokens': overall_metrics.get('avg_output_tokens'),
            'module_tokens': mt,
        }, f, indent=2, ensure_ascii=False)


def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, apply_backoff=False, tokenizer=None, per_q_module_tokens=None):
    per_q_out_tokens = _per_q_output_tokens(output_list, tokenizer)
    avg_tokens = (sum(per_q_out_tokens) / len(per_q_out_tokens)) if per_q_out_tokens else None

    # Unify: module_tokens.reasoning.output uses the same cap+strip+retokenize
    # logic as avg_output_tokens, replacing the raw vLLM-counted value so the
    # two metrics are consistent by construction.
    if per_q_module_tokens and per_q_out_tokens:
        rsn = per_q_module_tokens.get('reasoning')
        if isinstance(rsn, dict):
            if len(per_q_out_tokens) == len(rsn.get('output', [])):
                rsn['output'] = list(per_q_out_tokens)
        elif isinstance(rsn, list) and len(per_q_out_tokens) == len(rsn):
            per_q_module_tokens['reasoning'] = list(per_q_out_tokens)

    module_tokens_summary = _summarize_module_tokens(per_q_module_tokens)
    if dataset_name == 'livecode':
        # Prepare samples and generations for codegen_metrics
        samples_list = []
        generations_list = []

        # Collect difficulty levels for per-domain metrics
        difficulties = []
        per_difficulty_count = {}
        num_valid_answer = 0

        rum_list = []
        rum_per_difficulty = defaultdict(list)
        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            if type(result) == str:
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text
            output_for_eval = _cap_output(item['Output'])
            difficulty = item.get("difficulty", "Unknown")
            difficulties.append(difficulty)
            # Track metrics per domain
            if difficulty not in per_difficulty_count.keys():
                per_difficulty_count[difficulty] = 0

            rum = compute_rumination(output_for_eval)
            item['Rumination'] = rum
            rum_list.append(rum)
            rum_per_difficulty[difficulty].append(rum)

            pred_code = extract_answer(output_for_eval, mode='codegen')
            if pred_code != '':
                num_valid_answer += 1
                per_difficulty_count[difficulty] += 1
            # Assuming each item has 'input_output' with 'inputs' and 'outputs'
            public_test_cases = json.loads(item.get("public_test_cases", "{}"))

            inputs, outputs = [], []
            for case in public_test_cases:
                inputs.append(case["input"])
                outputs.append(case["output"])

            sample = {
                "input_output": json.dumps({
                    "inputs": inputs,
                    "outputs": outputs
                }),
            }

            samples_list.append(sample)
            generations_list.append([pred_code])
            item['Pred_Answer'] = pred_code
            item['Question'] = input_prompt


        # Call codegen_metrics with pass@1
        metrics, results, final_metadata = codegen_metrics(
            samples_list,
            generations_list,
            k_list=[1],  # Evaluate the top 1 generated result
            num_process_evaluate=2,   # Parallel evaluation
            timeout=10,  # Set timeout to 10 seconds
            debug=False,  # Enable debug mode
        )
        # print('samples_list', samples_list)
        # print('generations_list', generations_list)
        # print('metrics', metrics)

        # Extract pass@1
        pass_at_1 = metrics.get('pass@1', 0.0)
        detail_pass_at_1 = metrics['detail']['pass@1']

        for item, pass1, res, meta in zip(filtered_data, detail_pass_at_1.values(), results.values(), final_metadata):
            item['Metrics'] = {'pass@1': pass1}
            item['Results'] = res
            item['Final_metadata'] = meta

        # Initialize per-difficulty metrics
        difficulty_metrics = defaultdict(list)
        for idx, difficulty in enumerate(difficulties):
            pass1 = detail_pass_at_1[idx]
            difficulty_metrics[difficulty].append(pass1)

        # Compute overall pass@1
        overall_metrics = {
            'pass@1': pass_at_1,  # / num_valid_answer * len(input_list),
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
            'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
        }
        if avg_tokens is not None:
            overall_metrics['avg_output_tokens'] = f'{avg_tokens:.1f}'
        if module_tokens_summary is not None:
            overall_metrics['module_tokens'] = module_tokens_summary

        rum_overall = _aggregate_rumination(rum_list)
        if rum_overall is not None:
            overall_metrics['rumination'] = rum_overall

        # Compute per-difficulty pass@1
        per_difficulty_metrics = {}
        for difficulty, passes in difficulty_metrics.items():
            avg_pass = np.mean(passes) if len(passes) > 0 else 0.0
            num_valid_answer = per_difficulty_count[difficulty]
            per_difficulty_metrics[difficulty] = {
                'pass@1': avg_pass,
                'num_valid_answer': f'{num_valid_answer} of {len(passes)}'
            }
            rum_dom = _aggregate_rumination(rum_per_difficulty.get(difficulty, []))
            if rum_dom is not None:
                per_difficulty_metrics[difficulty]['rumination'] = rum_dom

        # Save the metrics
        final_metrics = {
            'overall': overall_metrics,
            'per_domain': per_difficulty_metrics
        }

    else:
        # Existing evaluation for other datasets
        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
        num_valid_answer = 0
        rum_list = []

        # If the dataset is GPQA, track metrics per domain
        domain_metrics = {}

        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            if type(result) == str:
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text
            output_for_eval = _cap_output(item['Output'])
            rum = compute_rumination(output_for_eval)
            item['Rumination'] = rum
            rum_list.append(rum)
            if dataset_name in ['gpqa', 'medmcqa']:
                labeled_answer = item["Correct Choice"]
                # labeled_choice_answer = item["Correct Answer"]
                mode = 'choose'
            elif dataset_name in ['math500', 'aime', 'amc']:
                labeled_answer = item["answer"]
                mode = 'gen'
            elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'ambigqa']:
                labeled_answer = item.get("golden_answers", item["answer"])
                if isinstance(labeled_answer, str):
                    labeled_answer = [labeled_answer]
                mode = 'qa'
            elif dataset_name in ['pubhealth']:
                labeled_answer = item["answer"]
                mode = 'choose'
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")

            metric, pred_answer = evaluate_predictions(output=output_for_eval, labeled_answer=labeled_answer, mode=mode)
            item['Pred_Answer'] = pred_answer
            item['Metrics'] = metric
            item['Question'] = input_prompt

            # Determine the validity of the predicted answer
            my_method_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])
            avg_math.append(metric['math_equal'])

            if my_method_valid:
                num_valid_answer += 1

            # If the dataset is GPQA, attempt to track metrics per domain
            if dataset_name == 'gpqa':
                domain = item.get("High-level domain", "Unknown")
                if domain not in domain_metrics:
                    domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0, 'rum': []}
                domain_metrics[domain]['total_num'] += 1
                domain_metrics[domain]['em'].append(metric['em'])
                domain_metrics[domain]['acc'].append(metric['acc'])
                domain_metrics[domain]['f1'].append(metric['f1'])
                domain_metrics[domain]['math_equal'].append(metric['math_equal'])
                domain_metrics[domain]['rum'].append(rum)
                if my_method_valid:
                    domain_metrics[domain]['num_valid_answer'] += 1

        t = time.localtime()
        result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
        metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

        # Compute overall metrics
        overall_results = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
            'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
            'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
        }
        if avg_tokens is not None:
            overall_results['avg_output_tokens'] = f'{avg_tokens:.1f}'
        if module_tokens_summary is not None:
            overall_results['module_tokens'] = module_tokens_summary

        rum_overall = _aggregate_rumination(rum_list)
        if rum_overall is not None:
            overall_results['rumination'] = rum_overall

        # If the dataset is GPQA, output average metrics per domain
        domain_avg_metrics = {}
        if dataset_name == 'gpqa':
            for dm, m in domain_metrics.items():
                domain_avg_metrics[dm] = {
                    'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                    'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                    'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                    'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                    'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}'
                }
                rum_dom = _aggregate_rumination(m.get('rum', []))
                if rum_dom is not None:
                    domain_avg_metrics[dm]['rumination'] = rum_dom

        # 保存总体和分domain的指标
        final_metrics = {'overall': overall_results}
        if dataset_name == 'gpqa':
            final_metrics['per_domain'] = domain_avg_metrics

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'
    if apply_backoff:
        result_json_name = output_dir
        metrics_json_name = output_dir.replace('.json', '.metrics.backoff.json')

    # Save prediction results and metrics
    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)

    # Per-run cost analysis (no-op if module_tokens is unavailable).
    _write_cost_report(
        output_dir,
        metrics_json_name,
        dataset_name,
        final_metrics.get('overall', final_metrics),
    )



if __name__ == "__main__":
    import argparse

    # Parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Evaluate model outputs with optional backoff.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the model output JSON file.')
    parser.add_argument('--output_metrics_path', type=str, help='Path to save the evaluation metrics.')
    parser.add_argument('--apply_backoff', action='store_true', help='Enable backoff to normal outputs if main output is invalid.')
    args = parser.parse_args()

    output_path = args.output_path
    if args.output_metrics_path:
        output_metrics_path = args.output_metrics_path
    else:
        output_metrics_path = output_path.replace('.json', '.metrics.json')

    # Determine dataset name based on the output path
    # NOTE: To apply back off strategy for retrieval-augmented reasoning methods, please replace normal_output_path with your actual path for results with run_direct_gen.
    if 'gpqa' in output_path:
        dataset_name = 'gpqa'
        normal_output_path = './outputs/gpqa.qwq.direct/diamond.12.13,18:23.json'
        if 'extended' in output_path:
            normal_output_path = './outputs/gpqa.qwq.direct/extended.12.28,15:44.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/gpqa.qwen2.5-32b-instruct.direct/diamond.12.14,20:34.json'
    elif 'math500' in output_path:
        dataset_name = 'math500'
        normal_output_path = './outputs/math500.qwq.direct/test.12.13,18:26.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/math500.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
    elif 'aime' in output_path:
        dataset_name = 'aime'
        normal_output_path = './outputs/aime.qwq.direct/2024.12.13,19:36.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/aime.qwen2.5-32b-instruct.direct/test.12.14,20:28.json'
    elif 'amc' in output_path:
        dataset_name = 'amc'
        normal_output_path = './outputs/amc.qwq.direct/test.12.14,14:31.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/amc.qwen2.5-32b-instruct.direct/test.12.14,20:26.json'
    elif 'livecode' in output_path:
        dataset_name = 'livecode'
        normal_output_path = './outputs/livecode.qwq.direct/test.12.13,21:24.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/livecode.qwen2.5-32b-instruct.direct/test.12.14,20:32.json'
    elif 'nq' in output_path:
        dataset_name = 'nq'
        normal_output_path = './outputs/runs.qa/nq.qwq.direct/test.12.15,14:50.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'triviaqa' in output_path:
        dataset_name = 'triviaqa'
        normal_output_path = './outputs/runs.qa/triviaqa.qwq.direct/test.12.15,15:35.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'hotpotqa' in output_path:
        dataset_name = 'hotpotqa'
        normal_output_path = './outputs/runs.qa/hotpotqa.qwq.direct/test.12.15,14:52.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'musique' in output_path:
        dataset_name = 'musique'
        normal_output_path = './outputs/runs.qa/musique.qwq.direct/test.12.27,16:44.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'bamboogle' in output_path:
        dataset_name = 'bamboogle'
        normal_output_path = './outputs/runs.qa/bamboogle.qwq.direct/test.12.28,9:51.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif '2wiki' in output_path:
        dataset_name = '2wiki'
        normal_output_path = './outputs/runs.qa/2wiki.qwq.direct/test.12.15,15:32.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'medmcqa' in output_path:
        dataset_name = 'medmcqa'
        normal_output_path = './outputs/runs.qa/medmcqa.qwq.direct/test.12.15,16:57.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'pubhealth' in output_path:
        dataset_name = 'pubhealth'
        normal_output_path = './outputs/runs.qa/pubhealth.qwq.direct/test.12.15,20:32.json'
        if 'qwq' not in output_path:
            normal_output_path = ''

    # Load main output data
    with open(output_path, mode='r', encoding='utf-8') as file:
        data = json.load(file)

    # Load main metrics data
    with open(output_metrics_path, mode='r', encoding='utf-8') as file:
        metrics = json.load(file)

    # Extract existing metrics
    if 'overall' in metrics:
        query_latency = metrics['overall']['query_latency']
        original_num_valid_answer = metrics['overall']['num_valid_answer']
        avg_output_tokens = metrics['overall'].get('avg_output_tokens')
    else:
        query_latency = metrics.get('query_latency', 'N/A')
        original_num_valid_answer = metrics.get('num_valid_answer', 'N/A')
        avg_output_tokens = metrics.get('avg_output_tokens')

    # Load normal output data if backoff is enabled
    normal_data = None
    if args.apply_backoff:
        if not os.path.exists(normal_output_path):
            raise FileNotFoundError(f"Normal output file not found at: {normal_output_path}")
        with open(normal_output_path, mode='r', encoding='utf-8') as file:
            normal_data = json.load(file)

    if dataset_name != 'livecode':
        # Existing evaluation for non-livecode datasets
        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
        num_valid_answer = 0
        rum_list = []

        # Initialize per-domain metrics
        domain_metrics = {}

        for i, item in enumerate(data):
            if dataset_name in ['gpqa', 'medmcqa']:
                labeled_answer = item["Correct Choice"]
                domain = item.get("High-level domain", "Unknown")
                mode = 'choose'
            elif dataset_name == 'math500':
                labeled_answer = item["answer"]
                domain = item.get("level", "Unknown")
                mode = 'gen'
            elif dataset_name in ['aime', 'amc']:
                labeled_answer = item["answer"]
                mode = 'gen'
                domain = 'Unknown'
            elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'ambigqa']:
                labeled_answer = item.get("golden_answers", item["answer"])
                if isinstance(labeled_answer, str):
                    labeled_answer = [labeled_answer]
                mode = 'qa'
                domain = 'Unknown'
            elif dataset_name in ['pubhealth']:
                labeled_answer = item["answer"]
                mode = 'choose'
                domain = 'Unknown'
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            output = _cap_output(item['Output'])

            metric, pred_answer = evaluate_predictions(
                output=output,
                labeled_answer=labeled_answer,
                mode=mode,
            )

            # Determine if the main method's answer is valid
            my_method_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

            # If invalid and backoff is enabled, use normal method's output
            if args.apply_backoff and not my_method_valid and normal_data is not None:
                normal_item = normal_data[i]
                if dataset_name in ['gpqa', 'medmcqa']:
                    normal_labeled_answer = normal_item["Correct Choice"]
                    normal_mode = 'choose'
                elif dataset_name == 'math500':
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'gen'
                elif dataset_name in ['aime', 'amc']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'gen'
                elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'ambigqa']:
                    normal_labeled_answer = normal_item.get("golden_answers", normal_item["answer"])
                    if isinstance(normal_labeled_answer, str):
                        normal_labeled_answer = [normal_labeled_answer]
                    normal_mode = 'qa'
                elif dataset_name in ['pubhealth']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'choose'
                else:
                    raise ValueError(f"Unsupported dataset for backoff: {dataset_name}")

                normal_output = _cap_output(normal_item['Output'])

                normal_metric, normal_pred_answer = evaluate_predictions(
                    output=normal_output,
                    labeled_answer=normal_labeled_answer,
                    mode=normal_mode,
                )
                normal_valid = (normal_pred_answer != '' and not (normal_mode == 'choose' and dataset_name == 'gpqa' and len(normal_pred_answer) > 1))

                # Use normal method's result if valid
                if normal_valid:
                    metric = normal_metric
                    pred_answer = normal_pred_answer
                    my_method_valid = True

            # Track metrics per domain
            if domain not in domain_metrics:
                domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0, 'rum': []}
            domain_metrics[domain]['total_num'] += 1

            rum = compute_rumination(output)
            item['Rumination'] = rum
            rum_list.append(rum)
            domain_metrics[domain].setdefault('rum', []).append(rum)

            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])
            avg_math.append(metric['math_equal'])
            domain_metrics[domain]['em'].append(metric['em'])
            domain_metrics[domain]['acc'].append(metric['acc'])
            domain_metrics[domain]['f1'].append(metric['f1'])
            domain_metrics[domain]['math_equal'].append(metric['math_equal'])

            if my_method_valid:
                num_valid_answer += 1
                domain_metrics[domain]['num_valid_answer'] += 1

        # Compute overall metrics
        overall_metrics = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0, 
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0, 
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0, 
            'math_equal': np.mean(avg_math) if len(avg_math) > 0 else 0, 
            'num_valid_answer': f'{num_valid_answer} of {len(data)}',
            'query_latency': query_latency,
        }
        if avg_output_tokens is not None:
            overall_metrics['avg_output_tokens'] = avg_output_tokens
        if args.apply_backoff:
            overall_metrics['original_num_valid_answer'] = original_num_valid_answer

        rum_overall = _aggregate_rumination(rum_list)
        if rum_overall is not None:
            overall_metrics['rumination'] = rum_overall

        # Compute per-domain metrics
        domain_avg_metrics = {}
        for dm, m in domain_metrics.items():
            domain_avg_metrics[dm] = {
                'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}',
            }
            rum_dom = _aggregate_rumination(m.get('rum', []))
            if rum_dom is not None:
                domain_avg_metrics[dm]['rumination'] = rum_dom

        # Prepare final metrics
        final_metrics = {'overall': overall_metrics}
        if dataset_name == 'gpqa':
            final_metrics['per_domain'] = domain_avg_metrics

    else:
        # Evaluation and backoff for livecode dataset
        split = 'test'  # Modify as needed or extract from output_path

        if args.apply_backoff and normal_data is not None:
            # Apply backoff by replacing invalid outputs with normal outputs
            for i, item in enumerate(data):
                # Extract Pred_Answer from main output
                pred_answer = item['Pred_Answer']

                # Check if Pred_Answer is invalid
                if pred_answer == '':
                    # Replace Output with normal output
                    item['Output'] = normal_data[i]['Output']

        # Prepare input_list and output_list for run_evaluation
        input_list = [item['Question'] for item in data]
        output_list = [item['Output'] for item in data]

        # Estimate total_time (if available). Here, set to 0 as a placeholder.
        total_time = 0  # Modify if timing information is available

        # Run evaluation
        run_evaluation(
            filtered_data=data,
            input_list=input_list,
            output_list=output_list,
            dataset_name=dataset_name,
            output_dir=output_path,
            total_time=total_time,
            split=split,
            apply_backoff=True,
        )
        # run_evaluation handles saving the metrics for livecode

    # Save metrics for non-livecode datasets
    if dataset_name != 'livecode' or not args.apply_backoff:
        # If dataset is livecode and backoff was applied, metrics are already saved by run_evaluation
        if args.apply_backoff:
            output_metrics_path = output_metrics_path.replace('.json', '.backoff.json')
        with open(output_metrics_path, mode='w', encoding='utf-8') as json_file:
            json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)

    print(f"Evaluation completed. Metrics saved to {output_metrics_path}")
