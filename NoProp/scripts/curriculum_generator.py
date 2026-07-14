"""
Curriculum dataset generator — thinker-optimized.
Core reasoning/phases get larger budgets. Every sample checkpointed instantly.

Usage:
  python curriculum_generator.py --thinker --samples 1000   # core phases get ~2-4x
  python curriculum_generator.py --phases 3,5 --samples 2000 # reasoning + math
  python curriculum_generator.py --resume                     # continue from last save
"""
import os, sys, json, time, uuid, argparse, re, signal, tempfile

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(PROJ_ROOT, "curriculum_data")
API_BASE = os.environ.get("LLAMA_API", "http://127.0.0.1:8080")
API_URL = f"{API_BASE}/v1/chat/completions"

SYSTEM_PROMPT = "You are a professor generating training data. Output ONLY the JSON requested."

# Global state for graceful shutdown
_graceful_stop = False
_phase_state = {}  # phase_id -> {"written": int, "jsonl_path": str, "count_path": str}


def _signal_handler(sig, frame):
    global _graceful_stop
    if _graceful_stop:
        print("\n  [SIG] Forced exit.")
        sys.exit(1)
    _graceful_stop = True
    print(f"\n  [SIG] Finishing current sample... (Ctrl+C again to force)")
    # Flush phase counts immediately
    for pid, st in _phase_state.items():
        if st["written"] > 0:
            _write_count_atomic(st["count_path"], st["written"])
            print(f"  [SIG] Phase {pid}: saved count={st['written']}")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _write_count_atomic(path: str, value: int):
    """Write count to a temp file then rename — no partial writes on power loss."""
    tmp = path + ".tmp." + uuid.uuid4().hex[:8]
    try:
        with open(tmp, "w") as f:
            f.write(str(value))
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass

# Each phase has a `weight` — effective samples = base_samples * weight in thinker mode.
# Thinker weights: core language/reasoning/math subjects get 2-4x.
# Use --equal to give all phases weight=1.
PHASES = [
    {"id": 0, "name": "Foundation", "domain": "language", "weight": 3,
     "difficulty_base": 1,
     "reasoning_types": ["recognition", "understanding"],
     "topic_groups": [
         {"concept": "Parts of Speech", "subtopics": [
             "nouns — proper, common, abstract, concrete",
             "verbs — transitive, intransitive, auxiliary, phrasal",
             "adjectives — comparative, superlative, attributive, predicative",
             "adverbs — manner, time, place, frequency, degree",
             "pronouns — personal, possessive, reflexive, relative, interrogative",
             "prepositions — time, place, direction, agent, instrument",
             "conjunctions — coordinating, subordinating, correlative",
             "determiners — articles, demonstratives, possessives, quantifiers",
         ]},
         {"concept": "Sentence Structure", "subtopics": [
             "simple sentences — subject-predicate core",
             "compound sentences — coordinating conjunctions",
             "complex sentences — subordinate clauses",
             "compound-complex sentences — multiple clauses",
             "declarative, interrogative, imperative, exclamatory",
             "direct and indirect objects",
             "relative clauses — restrictive vs non-restrictive",
             "participial phrases and absolute constructions",
         ]},
         {"concept": "Verb System", "subtopics": [
             "simple tenses — past, present, future",
             "progressive/continuous aspect",
             "perfect and perfect progressive tenses",
             "conditional sentences — zero through third",
             "active vs passive voice transformations",
             "subject-verb agreement in complex sentences",
             "irregular verb patterns and exceptions",
             "inversion and question formation",
         ]},
         {"concept": "Morphology & Word Formation", "subtopics": [
             "roots and base morphemes",
             "prefixes — negative, temporal, spatial, quantitative",
             "suffixes — noun-forming, adjective-forming, verb-forming",
             "compounding — noun-noun, adjective-noun, verb-particle",
             "blending, clipping, acronyms, and back-formation",
             "derivational vs inflectional morphology",
             "homophones, homographs, and homonyms",
             "collocations — strong vs weak, grammatical vs lexical",
         ]},
         {"concept": "Punctuation & Mechanics", "subtopics": [
             "comma usage — lists, clauses, appositives, interjections",
             "semicolons — joining clauses, complex lists",
             "colons — lists, explanations, quotations",
             "quotation marks — dialogue, titles, scare quotes",
             "apostrophes — possession, contraction, special uses",
             "dashes and hyphens — parenthetical, compound words",
             "capitalization — proper nouns, titles, after colons",
             "paragraph structure — topic sentences, transitions, coherence",
         ]},
     ]},
    {"id": 1, "name": "Knowledge", "domain": "general_knowledge", "weight": 2,
     "difficulty_base": 2,
     "reasoning_types": ["recognition", "understanding"],
     "topic_groups": [
         {"concept": "History", "subtopics": [
             "ancient civilizations — Egypt, Mesopotamia, Indus Valley",
             "classical Greece and Rome — philosophy, governance, warfare",
             "medieval period — feudalism, crusades, black death",
             "renaissance and enlightenment — science, art, philosophy",
             "industrial revolution — steam, factories, urbanization",
             "world wars — causes, major battles, consequences",
             "space exploration — moon landing, ISS, Mars rovers",
             "modern geopolitics — cold war, globalization, climate",
         ]},
         {"concept": "Science", "subtopics": [
             "physics — mechanics, thermodynamics, electromagnetism",
             "chemistry — elements, bonding, reactions, periodic table",
             "biology — cells, genetics, evolution, ecosystems",
             "earth science — geology, meteorology, oceanography",
             "astronomy — stars, planets, galaxies, cosmology",
             "neuroscience — brain structure, neurons, cognition",
             "ecology — food webs, biomes, conservation, climate",
             "materials science — metals, polymers, composites, nanomaterials",
         ]},
         {"concept": "Geography", "subtopics": [
             "physical geography — mountains, rivers, deserts, oceans",
             "human geography — population, migration, urbanization",
             "political geography — countries, borders, capitals",
             "economic geography — resources, trade, development",
             "climate zones — tropical, temperate, arctic, mediterranean",
             "biogeography — species distribution, biomes, endemism",
             "cartography — projections, scales, GIS, remote sensing",
             "geopolitics — strategic resources, territorial disputes",
         ]},
         {"concept": "Social Sciences", "subtopics": [
             "psychology — cognition, behavior, development, disorders",
             "sociology — social structures, institutions, inequality",
             "economics — supply/demand, markets, fiscal policy",
             "anthropology — human evolution, culture, archaeology",
             "political science — systems, ideologies, international relations",
             "linguistics — language structure, acquisition, typology",
             "education — pedagogy, curriculum, learning theories",
             "philosophy — ethics, logic, metaphysics, epistemology",
         ]},
     ]},
    {"id": 2, "name": "Relationships", "domain": "relational_knowledge", "weight": 2,
     "difficulty_base": 3,
     "reasoning_types": ["understanding", "analysis"],
     "topic_groups": [
         {"concept": "Causal Chains", "subtopics": [
             "historical cause and effect — events leading to war",
             "ecological food chains — producers, consumers, decomposers",
             "economic causality — supply shocks, inflation, recession",
             "scientific causality — experimental cause and effect",
             "social causality — policy changes, social movements",
             "technical causality — system failures, cascade effects",
             "psychological causality — trauma, behavior, outcomes",
             "environmental causality — pollution, habitat loss, extinction",
         ]},
         {"concept": "Taxonomies & Hierarchies", "subtopics": [
             "biological taxonomy — domain, kingdom, phylum, class",
             "library classification — Dewey, LC, UDC systems",
             "organizational hierarchies — flat, matrix, hierarchical",
             "knowledge graphs — entities, relations, ontologies",
             "computer file systems — directories, permissions, paths",
             "semantic networks — nodes, edges, inheritance",
             "mathematical hierarchies — sets, groups, fields",
             "concept hierarchies — abstraction, specialization, composition",
         ]},
         {"concept": "Analogies & Mappings", "subtopics": [
             "structural analogies — atom:solar system",
             "functional analogies — heart:pump, brain:computer",
             "cross-domain mappings — biological to computational",
             "proportional analogies — A:B as C:D",
             "metaphorical reasoning — abstract concepts via concrete",
             "mathematical analogies — isomorphisms, homomorphisms",
             "systemic analogies — feedback loops in nature and engineering",
             "analogical problem solving — transfer across domains",
         ]},
         {"concept": "Graphs & Networks", "subtopics": [
             "social networks — friendships, influence, communities",
             "transportation networks — roads, flights, shortest paths",
             "communication networks — internet, protocols, routing",
             "biological networks — protein interaction, neural, gene",
             "economic networks — trade, supply chains, markets",
             "dependency graphs — prerequisites, build systems, scheduling",
             "knowledge graphs — Wikipedia, Wikidata, concept maps",
             "hierarchical clustering — dendrograms, phylogenetics",
         ]},
     ]},
    {"id": 3, "name": "Reasoning", "domain": "logical_reasoning", "weight": 4,
     "difficulty_base": 4,
     "reasoning_types": ["deduction", "induction", "abduction", "analogical", "counterfactual"],
     "topic_groups": [
         {"concept": "Deductive Reasoning", "subtopics": [
             "syllogisms — categorical, hypothetical, disjunctive",
             "modus ponens and modus tollens — if-then inference",
             "transitive inference — A>B, B>C therefore A>C",
             "proof by contradiction — reductio ad absurdum",
             "conditional reasoning — necessary and sufficient conditions",
             "quantifier reasoning — all, some, none, most",
             "formal logic — propositional, predicate, first-order",
             "mathematical deduction — theorem proving, derivation",
         ]},
         {"concept": "Inductive Reasoning", "subtopics": [
             "generalization from specific observations",
             "statistical inference — sampling, confidence intervals",
             "scientific induction — hypothesis formation, testing",
             "analogical induction — reasoning by similarity",
             "probabilistic reasoning — Bayesian inference, priors",
             "causal induction — discovering cause-effect relationships",
             "pattern induction — sequence prediction, rule learning",
             "abstraction — concrete examples to general principles",
         ]},
         {"concept": "Abductive Reasoning", "subtopics": [
             "inference to the best explanation",
             "medical diagnosis — symptoms to underlying causes",
             "scientific explanation — observations to theories",
             "detective reasoning — evidence to perpetrator",
             "diagnostic reasoning — system failures, root causes",
             "theory formation — best explanation for phenomena",
             "hypothesis generation — plausible explanations",
             "model selection — Occam's razor, simplicity criteria",
         ]},
         {"concept": "Counterfactual Reasoning", "subtopics": [
             "what-if analysis — alternative histories",
             "causal counterfactuals — had X not occurred",
             "regret and learning — counterfactual experience",
             "policy evaluation — what if different policies?",
             "intervention analysis — hypothetical interventions",
             "thought experiments — gedankenexperiments",
             "blame assignment — but-for causation",
             "optimization — what changes improve outcome?",
         ]},
         {"concept": "Critical Thinking", "subtopics": [
             "identifying logical fallacies — ad hominem, straw man",
             "evaluating evidence — relevance, reliability, sufficiency",
             "argument analysis — premises, conclusions, assumptions",
             "cognitive biases — confirmation, anchoring, availability",
             "decision theory — rational choice under uncertainty",
             "risk assessment — probability, impact, mitigation",
             "problem decomposition — complex to manageable parts",
             "meta-cognition — thinking about thinking, self-correction",
         ]},
         {"concept": "Mathematical Reasoning", "subtopics": [
             "algebraic reasoning — equations, functions, transformations",
             "geometric reasoning — shapes, proofs, spatial inference",
             "combinatorial reasoning — counting, permutations, graphs",
             "probabilistic reasoning — likelihood, expectation, risk",
             "algorithmic reasoning — procedures, invariants, complexity",
             "logical proofs — direct, indirect, induction",
             "optimization reasoning — maxima, minima, constraints",
             "information-theoretic reasoning — entropy, coding, channels",
         ]},
     ]},
    {"id": 4, "name": "Programming", "domain": "coding", "weight": 2,
     "difficulty_base": 5,
     "reasoning_types": ["deduction", "decomposition", "debugging"],
     "topic_groups": [
         {"concept": "Data Structures", "subtopics": [
             "arrays and strings — indexing, slicing, searching",
             "linked lists — singly, doubly, circular",
             "stacks and queues — LIFO, FIFO, priority queues",
             "hash tables — collision resolution, load factor, resizing",
             "trees — binary, BST, AVL, red-black, B-trees",
             "heaps — min-heap, max-heap, heap sort",
             "graphs — adjacency lists/matrices, DFS, BFS",
             "tries, segment trees, union-find, fenwick trees",
         ]},
         {"concept": "Algorithms", "subtopics": [
             "sorting — quicksort, mergesort, heapsort, radix",
             "searching — binary search, interpolation, exponential",
             "dynamic programming — memoization, tabulation, optimal substructure",
             "greedy algorithms — intervals, scheduling, Huffman",
             "graph algorithms — Dijkstra, A*, Floyd-Warshall, Kruskal",
             "string algorithms — KMP, Rabin-Karp, suffix arrays",
             "divide and conquer — master theorem, recurrences",
             "computational complexity — big-O, NP-completeness, reductions",
         ]},
         {"concept": "Software Design", "subtopics": [
             "design patterns — singleton, factory, observer, strategy",
             "architecture — MVC, microservices, event-driven",
             "testing — unit, integration, property-based, fuzzing",
             "debugging strategies — bisection, logging, traces",
             "refactoring techniques — extract, rename, inline, move",
             "API design — REST, GraphQL, gRPC, versioning",
             "error handling — exceptions, results, panics",
             "documentation — docstrings, specs, architecture docs",
         ]},
         {"concept": "Programming Paradigms", "subtopics": [
             "object-oriented programming — classes, inheritance, polymorphism",
             "functional programming — maps, folds, monads, immutability",
             "declarative programming — SQL, logic programming",
             "concurrent programming — threads, async, actors",
             "metaprogramming — macros, decorators, reflection",
             "type systems — static, dynamic, gradual, dependently typed",
             "memory management — GC, ARC, manual, arenas",
             "program correctness — invariants, pre/post conditions",
         ]},
     ]},
    {"id": 5, "name": "Mathematics", "domain": "mathematics", "weight": 3,
     "difficulty_base": 5,
     "reasoning_types": ["deduction", "induction", "proof"],
     "topic_groups": [
         {"concept": "Algebra", "subtopics": [
             "linear equations — systems, matrices, determinants",
             "polynomials — roots, factoring, interpolation",
             "abstract algebra — groups, rings, fields, modules",
             "vector spaces — basis, dimension, linear transformations",
             "eigenvalues and eigenvectors — diagonalization, SVD",
             "number theory — primes, gcd, modular arithmetic",
             "combinatorics — permutations, combinations, generating functions",
             "graph theory — matchings, colorings, flows, connectivity",
         ]},
         {"concept": "Calculus & Analysis", "subtopics": [
             "limits and continuity — epsilon-delta definition",
             "derivatives — rules, partial, directional, gradients",
             "integrals — definite, indefinite, multiple, line integrals",
             "differential equations — ODEs, PDEs, boundary conditions",
             "series — taylor, fourier, convergence tests",
             "multivariable calculus — gradient, divergence, curl",
             "real analysis — completeness, sequences, topology",
             "complex analysis — analytic functions, residues, contour integrals",
         ]},
         {"concept": "Probability & Statistics", "subtopics": [
             "probability theory — axioms, random variables, distributions",
             "statistical inference — estimation, testing, confidence",
             "Bayesian statistics — priors, posteriors, conjugate families",
             "regression — linear, logistic, regularization",
             "stochastic processes — Markov chains, random walks",
             "information theory — entropy, KL divergence, mutual information",
             "decision theory — expected utility, minimax, regret",
             "machine learning foundations — bias-variance, loss, optimization",
         ]},
         {"concept": "Geometry & Topology", "subtopics": [
             "euclidean geometry — triangles, circles, proofs",
             "analytic geometry — coordinates, conics, transformations",
             "differential geometry — manifolds, curvature, tensors",
             "topology — open sets, compactness, connectedness",
             "algebraic topology — homotopy, homology, fundamental group",
             "measure theory — sigma-algebras, Lebesgue integral",
             "functional analysis — Banach spaces, Hilbert spaces",
             "category theory — objects, morphisms, functors, natural transformations",
         ]},
         {"concept": "Discrete Mathematics", "subtopics": [
             "logic and propositional calculus",
             "set theory — operations, cardinality, axiomatic",
             "relations and functions — properties, composition",
             "mathematical induction — weak, strong, structural",
             "recurrence relations — solving, generating functions",
             "combinatorial design — block designs, Latin squares",
             "coding theory — error correction, Hamming codes",
             "cryptography — encryption, signatures, zero-knowledge",
         ]},
     ]},
    {"id": 6, "name": "Tool_Use", "domain": "tool_use", "weight": 1,
     "difficulty_base": 4,
     "reasoning_types": ["planning", "decomposition", "verification"],
     "topic_groups": [
         {"concept": "Tool Selection", "subtopics": [
             "when to use a tool vs direct reasoning",
             "argument construction for tool calls",
             "interpreting tool results",
             "error recovery from tool failures",
         ]},
         {"concept": "Tool Types", "subtopics": [
             "web search and information retrieval",
             "code interpreter and data analysis",
             "database queries", "API orchestration",
         ]},
     ]},
    {"id": 7, "name": "Long_Context", "domain": "long_context", "weight": 1,
     "difficulty_base": 6,
     "reasoning_types": ["analysis", "synthesis", "evaluation"],
     "topic_groups": [
         {"concept": "Document Analysis", "subtopics": [
             "codebase understanding", "paper comprehension", "book summarization",
         ]},
         {"concept": "Multi-hop Reasoning", "subtopics": [
             "cross-document reasoning", "information synthesis",
             "temporal reasoning",
         ]},
     ]},
    {"id": 8, "name": "Memory", "domain": "memory", "weight": 1,
     "difficulty_base": 5,
     "reasoning_types": ["analysis", "evaluation"],
     "topic_groups": [
         {"concept": "Conversational Memory", "subtopics": [
             "fact extraction", "retrieval and update", "forgetting",
         ]},
         {"concept": "Knowledge Management", "subtopics": [
             "knowledge graphs", "semantic memory", "episodic memory",
         ]},
     ]},
    {"id": 9, "name": "Multi_Agent", "domain": "multi_agent", "weight": 1,
     "difficulty_base": 7,
     "reasoning_types": ["planning", "synthesis", "evaluation"],
     "topic_groups": [
         {"concept": "Agent Collaboration", "subtopics": [
             "planner-critic workflow", "consensus", "role specialization",
         ]},
         {"concept": "Agent Communication", "subtopics": [
             "message passing", "shared context", "conflict resolution",
         ]},
     ]},
    {"id": 10, "name": "Self_Improvement", "domain": "meta_learning", "weight": 2,
     "difficulty_base": 7,
     "reasoning_types": ["evaluation", "synthesis", "creation"],
     "topic_groups": [
         {"concept": "Self-Critique", "subtopics": [
             "generate-critique-improve cycles",
             "comparing alternatives systematically",
             "error detection and self-correction",
         ]},
         {"concept": "Curriculum Design", "subtopics": [
             "identifying knowledge gaps",
             "adaptive learning paths",
             "benchmark-driven improvement",
         ]},
     ]},
]


def call_api(prompt: str, max_tokens: int = 1536, temp: float = 0.7,
             retries: int = 3) -> str | None:
    import httpx
    for attempt in range(retries):
        try:
            r = httpx.post(API_URL, json={
                "model": "local",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temp,
                "top_p": 0.95,
            }, timeout=300)
            r.raise_for_status()
            c = r.json()["choices"][0]["message"]["content"]
            return c.strip() if c else None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None


def build_prompt(phase, tg, sub, diff: int, rtype: str) -> str:
    return (
        f'Generate a training sample. Phase: {phase["name"]}. '
        f'Domain: {phase["domain"]}. Concept: {tg["concept"]}. '
        f'Subtopic: {sub}. Difficulty: {diff}/7. Type: {rtype}.\n\n'
        f'JSON format:\n'
        f'{{"id":"...","domain":"{phase["domain"]}","difficulty":{diff},'
        f'"concepts":["{tg["concept"]}","{sub}"],'
        f'"dependencies":[],"requires_memory":false,"requires_tools":false,'
        f'"reasoning_type":"{rtype}","input":"...","analysis":"...",'
        f'"verification":"...","final_answer":"...","quality":0.95,"teacher":"Gemma4"}}\n\n'
        f'Rules: input=question, analysis=step-by-step (200-500 words), '
        f'verification=how to check, final_answer=definitive answer. '
        f'Output ONLY the JSON.'
    )


def parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                text = "\n".join(lines[i+1:]).strip()
                break
        end = text.find("```")
        if end >= 0:
            text = text[:end].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None


def gen_phase(phase, output_dir: str, target: int, resume: bool):
    """Generate `target` samples for a phase (target already includes weight scaling)."""
    global _graceful_stop, _phase_state

    phase_id = phase["id"]
    phase_dir = os.path.join(output_dir, f"phase{phase_id:02d}_{phase['name']}")
    os.makedirs(phase_dir, exist_ok=True)
    jsonl_path = os.path.join(phase_dir, "samples.jsonl")
    count_path = os.path.join(phase_dir, "_count.txt")

    written = 0
    if resume:
        if os.path.exists(count_path):
            with open(count_path) as f:
                try:
                    written = int(f.read().strip())
                except (ValueError, OSError):
                    written = 0
        if os.path.exists(jsonl_path):
            actual = sum(1 for _ in open(jsonl_path, encoding="utf-8") if _.strip())
            if actual > written:
                written = actual
        if written > 0:
            print(f"  Resume: {written}/{target}")

    if written >= target:
        print(f"  Complete: {written}/{target}")
        return written

    # Register in global state for signal handler
    _phase_state[phase_id] = {"written": written, "jsonl_path": jsonl_path, "count_path": count_path}

    combos = [(tg, sub) for tg in phase["topic_groups"] for sub in tg["subtopics"]]
    attempts = written
    max_attempts = max(target * 4, written + 10000)
    phase_t0 = time.time()
    api_fails = 0
    parse_fails = 0
    server_reconnects = 0

    while written < target and attempts < max_attempts:
        if _graceful_stop:
            _write_count_atomic(count_path, written)
            print(f"  [STOP] Phase {phase_id}: saved {written}/{target}")
            return written

        idx = attempts % len(combos)
        tg, sub = combos[idx]
        diff = min(phase["difficulty_base"] + (attempts // len(combos)) % 3, 7)
        rtype = phase["reasoning_types"][attempts % len(phase["reasoning_types"])]

        prompt = build_prompt(phase, tg, sub, diff, rtype)
        temp = 0.7 + (written % 5) * 0.05

        progress = (written / max(target, 1)) * 100
        bar_len = 20
        filled = int(bar_len * progress / 100)
        bar = "#" * filled + "." * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {written:>3}/{target:<3} {sub[:44]:44s} ")
        sys.stdout.flush()
        t0 = time.time()

        response = call_api(prompt, max_tokens=2048, temp=temp, retries=5)
        elapsed = time.time() - t0

        if not response:
            api_fails += 1
            server_reconnects += 1
            print(f"\r  [{bar}] {written}/{target} {sub[:44]:44s} FAIL ({elapsed:.1f}s) retry={server_reconnects}")
            attempts += 1
            if server_reconnects >= 50:
                print(f"  [FATAL] Too many consecutive server failures. Save progress and abort.")
                _write_count_atomic(count_path, written)
                return written
            continue

        server_reconnects = 0  # Reset on success

        parsed = parse_json(response)
        if parsed is None:
            parse_fails += 1
            print(f"\r  [{bar}] {written}/{target} {sub[:44]:44s} BAD-JSON ({elapsed:.1f}s)")
            attempts += 1
            continue

        parsed["id"] = f"{phase['name']}_{written:06d}_{uuid.uuid4().hex[:8]}"
        parsed["teacher"] = "Gemma4"
        parsed["quality"] = min(float(parsed.get("quality", 0.85)), 0.99)

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
        written += 1
        _write_count_atomic(count_path, written)
        _phase_state[phase_id]["written"] = written

        rate = written / (time.time() - phase_t0) if written > 0 else 0
        eta = (target - written) / rate if rate > 0 else 0
        eta_str = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
        print(f"\r  [{bar}] {written}/{target} {sub[:44]:44s} OK ({elapsed:.1f}s, {rate:.2f}/s, ETA {eta_str})")

    phase_t = time.time() - phase_t0
    print(f"  -> {written} samples ({phase_t/60:.1f}m), api_fails={api_fails}, parse={parse_fails}")
    _write_count_atomic(count_path, written)
    return written


def main():
    global API_BASE, API_URL
    parser = argparse.ArgumentParser(description="Thinker-optimized curriculum generator")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=50,
                        help="Base samples per phase (multiplied by phase weight)")
    parser.add_argument("--phases", default=None,
                        help="Comma-separated IDs, e.g. 0,1,2")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from saved count (default)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh — ignore saved progress, overwrite existing")
    parser.add_argument("--equal", action="store_true",
                        help="Give all phases weight=1 (ignore thinker scaling)")
    parser.add_argument("--api", default=None)
    args = parser.parse_args()

    if args.api:
        API_BASE = args.api
        API_URL = f"{API_BASE}/v1/chat/completions"

    # --fresh overrides --resume
    if args.fresh:
        args.resume = False

    phases = PHASES
    if args.phases:
        ids = {int(x.strip()) for x in args.phases.split(",")}
        phases = [p for p in phases if p["id"] in ids]
        if not phases:
            print(f"No phases for IDs: {args.phases}")
            sys.exit(1)

    print(f"API: {API_URL}  ", end="")
    try:
        import httpx
        h = httpx.get(f"{API_BASE}/v1/models", timeout=10)
        h.raise_for_status()
        print("OK")
    except Exception:
        print("WARNING — server unreachable")
        try:
            if input("Continue? (y/N): ").lower() != "y":
                sys.exit(1)
        except EOFError:
            print("  (non-interactive, continuing...)")

    # Compute per-phase targets
    total_target = 0
    phase_targets = []
    for ph in phases:
        w = 1 if args.equal else ph["weight"]
        n = max(1, int(args.samples * w))
        phase_targets.append(n)
        total_target += n

    print(f"\n{'='*60}")
    print(f"  THINKER CURRICULUM")
    print(f"  Mode: {'balanced (--equal)' if args.equal else 'thinker-weighted'}")
    print(f"  Base: {args.samples} samples/phase")
    print(f"  Resume: {'yes' if args.resume else 'no (--fresh)'}")
    print(f"  Total samples across {len(phases)} phases: {total_target}")
    for ph, n in zip(phases, phase_targets):
        w = 1 if args.equal else ph["weight"]
        tag = " [CORE]" if w >= 3 else (" [HIGH]" if w >= 2 else "")
    print(f"    Phase {ph['id']}: {ph['name']:20s} {n:5d} samples (x{w}){tag}")
    total_est = total_target * 21
    print(f"  Estimated time: {total_est/60:.0f}m ({total_est/3600:.1f}h)")
    print(f"{'='*60}\n")

    os.makedirs(args.output, exist_ok=True)
    grand_total = 0
    grand_t0 = time.time()

    for ph, target in zip(phases, phase_targets):
        w = 1 if args.equal else ph["weight"]
        tag = " [CORE]" if w >= 3 else (" [HIGH]" if w >= 2 else "")
        print(f"\n{'='*60}")
        print(f"  Phase {ph['id']}: {ph['name']}{tag} - {target} samples")
        print(f"{'='*60}")
        n = gen_phase(ph, args.output, target, args.resume)
        grand_total += n

    elapsed = time.time() - grand_t0
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fn in os.walk(args.output) for f in fn if f.endswith(".jsonl"))

    print(f"\n{'='*60}")
    print(f"  GENERATION COMPLETE")
    print(f"  Samples: {grand_total}")
    print(f"  Time: {elapsed/60:.1f}m ({elapsed/3600:.1f}h)")
    print(f"  Data: {size>>20} MB")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")

    json.dump({
        "samples": grand_total, "time_s": elapsed, "size_mb": size>>20,
        "phases": [{"id": p["id"], "name": p["name"], "samples": t}
                    for p, t in zip(phases, phase_targets)],
    }, open(os.path.join(args.output, "_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
