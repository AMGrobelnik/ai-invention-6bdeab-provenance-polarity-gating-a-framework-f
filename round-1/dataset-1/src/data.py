#!/usr/bin/env python3
"""Load and standardize datasets to exp_sel_data_out schema for Provenance-Polarity Gate."""

import json
import math
import os
import random
import resource
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/data.log", rotation="30 MB", level="DEBUG")

WORKSPACE = Path(__file__).parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT_PATH = WORKSPACE / "full_data_out.json"
SEED = 42

# ── hardware / memory ─────────────────────────────────────────────────────────

def _detect_cpus() -> int:
    for p, period in [
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ]:
        try:
            if period is None:
                parts = Path(p).read_text().split()
                if parts[0] != "max":
                    return math.ceil(int(parts[0]) / int(parts[1]))
            else:
                q = int(Path(p).read_text())
                per = int(Path(period).read_text())
                if q > 0:
                    return math.ceil(q / per)
        except (FileNotFoundError, ValueError):
            pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


def _set_ram_limit() -> None:
    import psutil
    total = (_container_ram_gb() or psutil.virtual_memory().total / 1e9) * 1e9
    budget = int(total * 0.70)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (budget * 3, budget * 3))
        logger.info(f"RAM budget set: {budget/1e9:.1f} GB (70% of {total/1e9:.1f} GB)")
    except (ValueError, resource.error) as e:
        logger.warning(f"Could not set RAM limit: {e}")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list | dict:
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return json.loads(path.read_text())


def make_example(
    input_text: str,
    output_text: str,
    regime: str,
    source_dataset: str,
    **meta,
) -> dict:
    ex: dict = {
        "input": input_text,
        "output": output_text,
        "metadata_regime": regime,
        "metadata_source_dataset": source_dataset,
    }
    for k, v in meta.items():
        ex[f"metadata_{k}"] = v
    return ex


# ── dataset processors ────────────────────────────────────────────────────────

def process_clutrr(path: Path, n: int = 500) -> list[dict]:
    """CLUTRR v1 — kinship reasoning (universal_rule regime)."""
    rows = load_json(path)
    rng = random.Random(SEED)
    # stratify by task_name (hop depth)
    by_task: dict[str, list] = {}
    for row in rows:
        by_task.setdefault(row.get("task_name", "unknown"), []).append(row)
    sampled = []
    tasks = sorted(by_task)
    per_task = max(1, n // len(tasks))
    for task in tasks:
        bucket = by_task[task]
        rng.shuffle(bucket)
        sampled.extend(bucket[:per_task])
    rng.shuffle(sampled)
    sampled = sampled[:n]
    examples = []
    for row in sampled:
        hop_depth = len(row.get("f_comb", "").split("-"))
        input_text = (
            f"Story: {row['clean_story']}\n"
            f"Query: What is the relationship between {row['query']}?"
        )
        examples.append(make_example(
            input_text=input_text,
            output_text=row["target_text"],
            regime="universal_rule",
            source_dataset="CLUTRR/v1",
            hop_depth=hop_depth,
            f_comb=row.get("f_comb", ""),
            gold_premises=row.get("proof_state", ""),
            task_name=row.get("task_name", ""),
        ))
    logger.info(f"CLUTRR: {len(examples)} examples")
    return examples


def process_proofwriter_owa(depth_paths: list[Path], n_per_depth: int = 125) -> list[dict]:
    """hitachi-nlp ProofWriter OWA by depth (universal_rule regime)."""
    examples = []
    for path in depth_paths:
        rows = load_json(path)
        rng = random.Random(SEED)
        rng.shuffle(rows)
        taken = 0
        for row in rows:
            if taken >= n_per_depth:
                break
            theory = row.get("theory", "")
            questions = row.get("questions", {})
            all_proofs = row.get("allProofs", "")
            max_d = row.get("maxD", 0)
            # pick first non-random question with a definite answer
            for _qk, q in questions.items():
                answer = q.get("answer", "")
                if answer in ("True", "False") and q.get("question"):
                    input_text = (
                        f"Theory:\n{theory}\n\n"
                        f"Question: {q['question']}"
                    )
                    examples.append(make_example(
                        input_text=input_text,
                        output_text=answer,
                        regime="universal_rule",
                        source_dataset="hitachi-nlp/proofwriter_processed_OWA",
                        hop_depth=max_d,
                        proof_tree=all_proofs,
                    ))
                    taken += 1
                    break
        logger.info(f"ProofWriter OWA {path.name}: {taken} examples")
    logger.info(f"ProofWriter OWA total: {len(examples)} examples")
    return examples


def process_entailment_bank(path: Path, n: int = 500) -> list[dict]:
    """nguyen-brat/entailment_bank — science QA with CoT proof (universal_rule)."""
    rows = load_json(path)
    rng = random.Random(SEED)
    rng.shuffle(rows)
    examples = []
    for row in rows[:n]:
        question = row.get("question", "")
        answers = row.get("answer", [])
        cot = row.get("cot", [])
        if not question or not answers:
            continue
        answer_text = answers[0] if isinstance(answers, list) else answers
        proof_text = "\n".join(cot) if isinstance(cot, list) else str(cot)
        examples.append(make_example(
            input_text=f"Question: {question}",
            output_text=answer_text,
            regime="universal_rule",
            source_dataset="nguyen-brat/entailment_bank",
            proof=proof_text,
        ))
    logger.info(f"EntailmentBank: {len(examples)} examples")
    return examples


def process_babi_nli(path: Path, n: int = 500) -> list[dict]:
    """tasksource/babi_nli basic-deduction (universal_rule regime)."""
    rows = load_json(path)
    rng = random.Random(SEED)
    rng.shuffle(rows)
    label_map = {0: "not entailed", 1: "entailed"}
    examples = []
    for row in rows[:n]:
        premise = row.get("premise", "")
        hypothesis = row.get("hypothesis", "")
        label = row.get("label", 0)
        if not premise or not hypothesis:
            continue
        input_text = f"Premises:\n{premise}\n\nHypothesis: {hypothesis}\nDoes the hypothesis follow?"
        examples.append(make_example(
            input_text=input_text,
            output_text=label_map.get(label, str(label)),
            regime="universal_rule",
            source_dataset="tasksource/babi_nli",
            label=label,
        ))
    logger.info(f"bAbI NLI: {len(examples)} examples")
    return examples


def process_synthetic_reasoning(path: Path, n: int = 500) -> list[dict]:
    """lighteval/synthetic_reasoning_natural hard (document_stipulated regime)."""
    rows = load_json(path)
    rng = random.Random(SEED)
    rng.shuffle(rows)
    examples = []
    for row in rows[:n]:
        question = row.get("question", "")
        target = row.get("target", "")
        if not question or not target:
            continue
        examples.append(make_example(
            input_text=question,
            output_text=target,
            regime="document_stipulated",
            source_dataset="lighteval/synthetic_reasoning_natural",
        ))
    logger.info(f"Synthetic Reasoning: {len(examples)} examples")
    return examples


# ── synthetic generation ──────────────────────────────────────────────────────

NONSENSE_TERMS = [
    ("zorbite", "a mineral that conducts vorpal energy"),
    ("blinthorp", "a gas that expands when cooled"),
    ("quellix", "an organism that photosynthesizes via ultrasound"),
    ("dravian", "a particle with negative mass"),
    ("snorkel-moss", "a plant that absorbs light from soil"),
    ("fremium", "an element that decays into stability"),
    ("plonkite", "a crystal that melts at -200°C"),
    ("wazzle", "a compound that neutralises acids by becoming more acidic"),
    ("glormp", "a bacterium that requires no water to survive"),
    ("trelvex", "a wave that slows down in a vacuum"),
]

NONSENSE_AXIOMS = [
    ("All {term} are {prop1}.", "{prop1} things are {prop2}."),
    ("Every {term} exhibits {prop1} when exposed to {cond}.", "Exposure to {cond} triggers {prop2}."),
    ("No {term} can be both {prop1} and {prop2}.", "{term} in state X are {prop1}."),
]

PROPS = ["luminescent", "thermic", "refractive", "osmotic", "catalytic",
         "viscous", "resonant", "inert", "exothermic", "volatile"]
CONDS = ["high pressure", "low temperature", "solar radiation", "magnetic flux", "gravity waves"]


def gen_micro_ontology(n: int = 20) -> list[dict]:
    rng = random.Random(SEED)
    examples = []
    for i in range(n):
        term, defn = NONSENSE_TERMS[i % len(NONSENSE_TERMS)]
        prop1, prop2 = rng.sample(PROPS, 2)
        cond = rng.choice(CONDS)
        # Document: define the term + axioms
        doc = (
            f"MICRO-ONTOLOGY DOCUMENT #{i+1}\n\n"
            f"Definition: A {term} is {defn}.\n"
            f"Axiom 1: All {term}s are {prop1} under standard conditions.\n"
            f"Axiom 2: Any {prop1} substance in {cond} becomes {prop2}.\n"
            f"Axiom 3: A {prop2} substance emits a detectable signal.\n"
        )
        query = f"Given the above definitions, if a {term} is placed in {cond}, will it emit a detectable signal?"
        examples.append(make_example(
            input_text=f"{doc}\nQuery: {query}",
            output_text="Yes",
            regime="document_stipulated",
            source_dataset="synthetic_micro_ontology",
            term=term,
            is_synthetic=True,
        ))
    logger.info(f"Micro-ontology: {len(examples)} examples")
    return examples


CONTRACT_TEMPLATES = [
    (
        'AGREEMENT §{n}\n\nFor purposes of this Agreement, "Qualified Event" means any occurrence '
        "that satisfies all of the following: (a) it involves two or more Parties; (b) it occurs "
        "within the Territory; and (c) it has a documented financial impact exceeding $10,000.\n\n"
        "A \"Triggering Condition\" is met when a Qualified Event is reported in writing to the "
        "Compliance Officer within 30 days of its occurrence.\n\n"
        "When a Triggering Condition is met, the Indemnification Clause (Appendix B) becomes operative.",
        "Scenario: An event involving three parties occurred in the Territory, had a $50,000 financial "
        "impact, and was reported to the Compliance Officer in writing within 30 days.\n"
        "Question: Is the Indemnification Clause now operative?",
        "Yes — the Triggering Condition is met because the Qualified Event conditions are satisfied and "
        "the report was timely.",
    ),
    (
        'AGREEMENT §{n}\n\nAs used herein, "Material Breach" means any failure to perform an '
        "obligation that (i) is not cured within 15 days of written notice and (ii) causes "
        "demonstrable harm.\n\n"
        '"Remediation Period" means the 15-day window after receipt of written notice.\n\n'
        "Upon a Material Breach, the non-breaching Party may seek specific performance.",
        "Scenario: Party A failed to deliver goods. Party B sent written notice. 20 days passed "
        "with no cure and Party B suffered documented losses.\n"
        "Question: May Party B seek specific performance?",
        "Yes — a Material Breach occurred because the failure was uncured past the Remediation Period "
        "and caused demonstrable harm.",
    ),
    (
        'AGREEMENT §{n}\n\n"Confidential Information" means all non-public data exchanged under '
        "this Agreement, excluding information that: (a) was already public at the time of "
        "disclosure; (b) was independently developed; or (c) was received from a third party "
        "without restriction.\n\n"
        "Each Party shall hold Confidential Information in strict confidence for five years.",
        "Scenario: A formula was shared under this Agreement. The same formula was publicly "
        "disclosed by the disclosing party one day before the sharing.\n"
        "Question: Is the formula Confidential Information under this Agreement?",
        "No — it was already public at the time of disclosure, which is an explicit exclusion.",
    ),
]


def gen_contracts(n: int = 15) -> list[dict]:
    rng = random.Random(SEED + 1)
    examples = []
    for i in range(n):
        tmpl = CONTRACT_TEMPLATES[i % len(CONTRACT_TEMPLATES)]
        doc, scenario, answer = tmpl
        doc = doc.replace("{n}", str(100 + i))
        # slight variation
        extra = rng.choice([
            "",
            " Note: all standard definitions apply unless overridden.",
            " This section supersedes any prior oral agreements.",
        ])
        examples.append(make_example(
            input_text=f"{doc}{extra}\n\n{scenario}",
            output_text=answer,
            regime="document_stipulated",
            source_dataset="synthetic_contract",
            is_synthetic=True,
        ))
    logger.info(f"Contracts: {len(examples)} examples")
    return examples


NEAR_MISS_TEMPLATES = [
    # (story, query, bait_answer, correct_answer, bait_premise)
    (
        "Alice and Bob are siblings. Bob has a daughter named Carol.",
        "What is the relationship of Alice to Carol?",
        "aunt",  # correct
        "aunt",
        "Alice is Bob's sibling, Bob is Carol's parent → Alice is Carol's aunt.",
    ),
    (
        "The document states that all Registered Users receive a 10% discount.",
        "Does a Registered User receive a 20% discount?",
        "No",
        "No",
        "Document stipulates 10% discount for Registered Users, not 20%.",
    ),
    (
        "All zorblexes are thermite. This object is a zorblex.",
        "Is this object thermite?",
        "Yes",
        "Yes",
        "Universal rule: All zorblexes are thermite. Object is a zorblex → object is thermite.",
    ),
    (
        "If a creature is nocturnal and warm-blooded, it hibernates in winter.",
        "A bat is nocturnal and warm-blooded. Does it hibernate?",
        "Yes",
        "Yes",
        "Rule covers nocturnal + warm-blooded → hibernates. Bat satisfies both conditions.",
    ),
]


def gen_near_miss_bait(n: int = 20) -> list[dict]:
    """Near-miss bait instances: premise not in document."""
    rng = random.Random(SEED + 2)
    examples = []
    hallucinatory_premises = [
        "The document also states that siblings share all responsibilities.",
        "A 50% discount applies for Premium Users.",
        "All zorblexes are also combustible.",
        "Warm-blooded creatures always migrate south.",
        "The contract specifies a 30-day grace period.",
        "Children of siblings are automatically registered.",
    ]
    for i in range(n):
        base = NEAR_MISS_TEMPLATES[i % len(NEAR_MISS_TEMPLATES)]
        story, query, answer, _correct, real_premise = base
        # inject a hallucinated premise into a variant question
        fake_premise = rng.choice(hallucinatory_premises)
        bait_query = (
            f"{query} (Assume the following additional rule applies: {fake_premise})"
        )
        # correct answer is unchanged; the bait tries to distract
        examples.append(make_example(
            input_text=f"Document:\n{story}\n\nQuery: {bait_query}",
            output_text=answer,
            regime="universal_rule",
            source_dataset="synthetic_near_miss_bait",
            is_hallucination_bait=True,
            real_premise=real_premise,
            hallucinated_premise=fake_premise,
            is_synthetic=True,
        ))
    logger.info(f"Near-miss bait: {len(examples)} examples")
    return examples


PREFLIGHT_TEMPLATES = [
    # (true_class, doc, query, answer)
    # a = document_extensional: answer derivable only from document facts
    # b = universal_rule: answer requires world-knowledge rule
    # c = document_stipulated: answer requires rule stated in document
    (
        "a",
        "The Acme Corp report (Q3 2024) shows revenue of $4.2 million.",
        "What was Acme Corp's Q3 2024 revenue according to the report?",
        "$4.2 million",
    ),
    (
        "b",
        "John is Mary's father. Mary is Tom's mother.",
        "What is the relationship between John and Tom?",
        "grandfather",
    ),
    (
        "c",
        'This policy defines "Active Member" as any member who has attended at least 3 meetings in the past year. '
        "Active Members are entitled to vote.",
        "Is a member who attended 4 meetings this year entitled to vote?",
        "Yes — they qualify as an Active Member per the policy definition.",
    ),
    (
        "a",
        "The experiment used a 0.5 M sodium chloride solution at 37°C.",
        "At what temperature was the sodium chloride solution used?",
        "37°C",
    ),
    (
        "b",
        "Alice is Bob's aunt. Bob has a son named Charlie.",
        "What is Alice's relationship to Charlie?",
        "great-aunt",
    ),
    (
        "c",
        '"Force Majeure Event" means any event beyond a Party\'s reasonable control including '
        "natural disasters, war, or government action. In a Force Majeure Event, obligations are suspended.",
        "An earthquake destroys a Party's factory. Are their contractual obligations suspended?",
        "Yes — an earthquake is a natural disaster qualifying as a Force Majeure Event under this definition.",
    ),
    (
        "a",
        "Table 2: Sample A had pH 6.8; Sample B had pH 7.4; Sample C had pH 5.1.",
        "Which sample had the lowest pH according to Table 2?",
        "Sample C (pH 5.1)",
    ),
    (
        "b",
        "Every mammal is warm-blooded. A whale is a mammal.",
        "Is a whale warm-blooded?",
        "Yes",
    ),
    (
        "c",
        '"Late Payment" means any payment received after the 15th of the month. '
        "Late Payments incur a 5% surcharge.",
        "A payment was received on the 20th. Does a surcharge apply?",
        "Yes — payment on the 20th qualifies as a Late Payment per the definition.",
    ),
    (
        "a",
        "The patient was admitted on March 3 and discharged on March 7.",
        "How many days was the patient admitted?",
        "4 days",
    ),
]


def gen_preflight(n: int = 30) -> list[dict]:
    rng = random.Random(SEED + 3)
    examples = []
    pool = PREFLIGHT_TEMPLATES * (n // len(PREFLIGHT_TEMPLATES) + 1)
    rng.shuffle(pool)
    for i, (true_class, doc, query, answer) in enumerate(pool[:n]):
        examples.append(make_example(
            input_text=f"Document:\n{doc}\n\nQuery: {query}",
            output_text=answer,
            regime="preflight",
            source_dataset="synthetic_preflight",
            true_class=true_class,
            is_synthetic=True,
        ))
    logger.info(f"Preflight: {len(examples)} examples")
    return examples


# ── main ──────────────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    _set_ram_limit()

    datasets: list[dict] = []

    # 1. CLUTRR
    clutrr_path = DATASETS_DIR / "full_kendrivp_CLUTRR_v1_extracted_default_train.json"
    datasets.append({"dataset": "CLUTRR/v1", "examples": process_clutrr(clutrr_path, n=500)})

    # 2. ProofWriter OWA by depth (depths 0-3, 125 each = 500 total)
    pw_paths = [
        DATASETS_DIR / f"full_hitachi-nlp_proofwriter_processed_OWA_depth-{d}_train.json"
        for d in range(4)
    ]
    datasets.append({
        "dataset": "hitachi-nlp/proofwriter_processed_OWA",
        "examples": process_proofwriter_owa(pw_paths, n_per_depth=125),
    })

    # 3. EntailmentBank
    eb_path = DATASETS_DIR / "full_nguyen-brat_entailment_bank_default_train.json"
    datasets.append({"dataset": "nguyen-brat/entailment_bank", "examples": process_entailment_bank(eb_path, n=500)})

    # 4. Synthetic Reasoning Natural (document_stipulated)
    sr_path = DATASETS_DIR / "full_lighteval_synthetic_reasoning_natural_hard_train.json"
    datasets.append({"dataset": "lighteval/synthetic_reasoning_natural", "examples": process_synthetic_reasoning(sr_path, n=500)})

    out = {"datasets": datasets}
    total = sum(len(d["examples"]) for d in datasets)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Saved {total} total examples to {OUTPUT_PATH}")

    for d in datasets:
        logger.info(f"  {d['dataset']}: {len(d['examples'])} examples")


if __name__ == "__main__":
    main()
