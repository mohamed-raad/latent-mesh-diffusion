"""
Generate high-quality AGK datasets using a local GGUF model (gemma-4 via llama-cli.exe).
Unlimited generation — no API rate limits since everything runs locally.
Output goes to agk_llm/ (separate from template-based agk_data/).

Usage:
  uv run --no-sync --package noprop-mesh python scripts/generate_with_llm.py
  uv run --no-sync --package noprop-mesh python scripts/generate_with_llm.py --samples 500 --output agk_llm
  uv run --no-sync --package noprop-mesh python scripts/generate_with_llm.py --phases physics,coding,reasoning --samples 1000
"""
import os
import sys
import subprocess
import json
import argparse
import time
import re
import atexit

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LLAMA_DIR = os.environ.get("LLAMA_DIR") or os.path.join(os.path.dirname(_PROJ_ROOT), "LLAMA")
if not os.path.exists(_LLAMA_DIR):
    _LLAMA_DIR = os.path.join(_PROJ_ROOT, "..", "LLAMA")
if not os.path.exists(_LLAMA_DIR):
    _LLAMA_DIR = r"E:\my apps\LLAMA"
LLAMA_CLI = os.path.join(_LLAMA_DIR, "llama-cli.exe")
DEFAULT_MODEL = os.path.join(_LLAMA_DIR, "gemma-4-E2B-it-Q4_K_S.gguf")
DEFAULT_OUTPUT = os.path.join(_PROJ_ROOT, "agk_llm")

# Multimodal tools for vision/audio demo
LLAMA_MTMD = os.path.join(_LLAMA_DIR, "llama-mtmd-cli.exe")
LLAMA_TTS = os.path.join(_LLAMA_DIR, "llama-tts.exe")

SYSTEM_PROMPT = (
    "You are a helpful AI assistant that generates high-quality educational content. "
    "Your output must be valid Markdown with clear section headers. "
    "Be precise, detailed, and factually accurate."
)

TOPICS = [
    # ===== FOUNDATIONS: Language, Grammar, Communication =====
    {
        "id": 1, "name": "Language_Grammar", "count": 1000,
        "prompts": [
            "Write a comprehensive grammar lesson about {topic}. Cover definitions, rules, correct usage, and common mistakes with examples.",
            "Create a detailed guide to {topic}. Include clear explanations, example sentences, and usage notes.",
            "Generate a thorough reference on {topic}. Explain the concept, provide examples, and highlight common errors.",
        ],
        "topics": [
            "parts of speech — nouns, verbs, adjectives, adverbs", "verb tenses — past, present, future, perfect, progressive",
            "subject-verb agreement rules", "pronoun cases and antecedents",
            "prepositions and prepositional phrases", "conjunctions and sentence connectors",
            "articles — definite, indefinite, and zero article", "modifiers — adjectives, adverbs, and dangling modifiers",
            "sentence structure — simple, compound, complex, compound-complex", "punctuation — commas, periods, semicolons, colons",
            "apostrophes — possession and contractions", "quotation marks and dialogue formatting",
            "capitalization rules", "parallel structure in sentences",
            "active vs passive voice usage", "conditional sentences — zero, first, second, third",
            "relative clauses — defining and non-defining", "reported speech and indirect statements",
            "word order in questions and statements", "commonly confused words — their/there/they're, your/you're, its/it's",
        ],
    },
    {
        "id": 2, "name": "Conversation_Patterns", "count": 1000,
        "prompts": [
            "Write a detailed guide to conversational AI. Explain how to handle {topic} effectively with examples of good and bad responses.",
            "Create a comprehensive reference on communication patterns. Cover {topic} with example dialogues and best practices.",
            "Generate a thorough explanation of AI conversation techniques for {topic}. Include sample exchanges and rationale.",
        ],
        "topics": [
            "greeting users and establishing context", "handling ambiguous or unclear requests",
            "asking clarifying questions effectively", "providing concise and accurate answers",
            "explaining complex concepts in simple terms", "admitting uncertainty and limitations gracefully",
            "maintaining conversation coherence and topic focus", "handling multiple questions in one message",
            "following up and checking understanding", "using examples to illustrate points",
            "structuring responses with clear sections", "matching user tone — formal vs casual",
            "recovering from misunderstandings", "setting expectations about capabilities",
            "encouraging user engagement and follow-up questions", "politely correcting user misconceptions",
            "providing step-by-step instructions", "summarizing long discussions",
            "handling sensitive or controversial topics", "closing conversations naturally",
        ],
    },
    {
        "id": 3, "name": "Writing_Style", "count": 1000,
        "prompts": [
            "Write a comprehensive style guide for {topic}. Include rules, examples, and detailed explanations.",
            "Create a detailed reference on {topic} for clear professional writing. Show before/after examples.",
            "Generate a thorough guide to {topic} with practical tips, examples, and common pitfalls to avoid.",
        ],
        "topics": [
            "writing clear and concise sentences", "organizing ideas into coherent paragraphs",
            "using transitions for smooth flow between ideas", "tone — professional, friendly, instructional",
            "writing effective introductions and conclusions", "using bullet points and lists effectively",
            "choosing precise vocabulary over vague language", "avoiding jargon and explaining technical terms",
            "writing for different audiences — expert vs beginner", "using analogies and metaphors to explain ideas",
            "structuring arguments with claims and evidence", "writing actionable instructions",
            "formatting for readability — headings, spacing, emphasis", "revising and editing your own writing",
            "using data and examples to support claims", "writing persuasive and informative content",
            "balancing detail with brevity", "creating scannable content for quick comprehension",
            "writing inclusive and unbiased language", "developing a consistent voice and style",
        ],
    },
    # ===== LOGICAL REASONING =====
    {
        "id": 4, "name": "EN_Reasoning", "count": 1000,
        "prompts": [
            "Generate a detailed logical reasoning problem with step-by-step solution. Include premises, deduction steps, and a final conclusion. Topic: {topic}.",
            "Create a comprehensive syllogism problem. State premises clearly, walk through the logical deduction, and give the conclusion. Topic: {topic}.",
            "Write a thorough critical thinking puzzle. Present the scenario, reasoning steps, and solution. Topic: {topic}.",
        ],
        "topics": [
            "modus ponens and modus tollens", "hypothetical syllogism", "disjunctive syllogism",
            "transitive inference", "counterfactual reasoning", "analogical reasoning",
            "abductive reasoning", "temporal reasoning", "spatial reasoning", "causal reasoning",
            "deductive validity", "inductive strength", "logical fallacies", "proof by contradiction",
            "Bayesian reasoning", "decision theory", "game theory basics", "moral reasoning",
            "scientific reasoning", "statistical reasoning",
        ],
    },
    {
        "id": 5, "name": "Linguistics_Advanced", "count": 500,
        "prompts": [
            "Generate a comprehensive explanation of a linguistic concept. Include definition, examples, and key rules. Topic: {topic}.",
            "Create a grammar lesson covering usage, common errors, and correct examples. Topic: {topic}.",
            "Write a detailed analysis of a language phenomenon with examples from multiple languages. Topic: {topic}.",
        ],
        "topics": [
            "syntax and sentence structure", "morphology and word formation",
            "phonetics and phonology", "semantics and meaning",
            "pragmatics and context", "historical linguistics",
            "language acquisition", "bilingualism and multilingualism",
            "grammatical tense and aspect", "grammatical case systems",
            "discourse analysis", "sociolinguistics",
            "comparative linguistics", "corpus linguistics",
            "translation theory and practice",
        ],
    },
    {
        "id": 6, "name": "EN_Coding", "count": 1000,
        "prompts": [
            "Generate a coding problem with complete solution. Include problem description, solution code in Python, and complexity analysis. Topic: {topic}.",
            "Create an algorithm challenge with implementation. Describe the algorithm, provide well-commented code, and analyze time/space complexity. Topic: {topic}.",
            "Write a data structures problem. Show the problem, the data structure choice, implementation, and analysis. Topic: {topic}.",
        ],
        "topics": [
            "dynamic programming", "graph algorithms", "tree traversal",
            "hash table design", "sorting algorithms", "search algorithms",
            "string manipulation", "recursion", "greedy algorithms", "backtracking",
            "linked list operations", "stack and queue", "heap priority queue",
            "trie data structure", "union-find / DSU", "segment tree",
            "bit manipulation", "sliding window", "two-pointer technique",
            "divide and conquer",
        ],
    },
    {
        "id": 7, "name": "EN_Physics", "count": 500,
        "prompts": [
            "Generate a physics problem with complete solution. Include problem statement, relevant formulas, derivation, and final answer. Topic: {topic}.",
            "Create an in-depth explanation of a physical concept. Cover the underlying principles, mathematical formulation, and real-world applications. Topic: {topic}.",
            "Write a step-by-step worked example in physics. Show the setup, equations, algebraic manipulation, numerical calculation, and result. Topic: {topic}.",
        ],
        "topics": [
            "Newtonian mechanics — forces and motion", "energy conservation and work",
            "thermodynamics and heat transfer", "electromagnetism and Maxwell's equations",
            "quantum mechanics fundamentals", "special relativity",
            "wave physics and optics", "fluid dynamics",
            "statistical mechanics", "nuclear physics",
            "particle physics and the Standard Model", "condensed matter physics",
            "astrophysics and cosmology", "chaos theory and nonlinear dynamics",
            "computational physics and simulation",
        ],
    },
    {
        "id": 8, "name": "EN_Maths", "count": 1000,
        "prompts": [
            "Generate a mathematics problem with full step-by-step solution. Include the problem statement, derivation steps, and final answer. Topic: {topic}.",
            "Create a calculus/mathematical analysis problem. Show the working and reasoning clearly. Topic: {topic}.",
            "Write an applied mathematics problem. Include real-world context, mathematical formulation, and solution. Topic: {topic}.",
        ],
        "topics": [
            "derivatives and differentiation", "integrals and integration techniques",
            "differential equations", "linear algebra — matrix operations",
            "eigenvalues and eigenvectors", "probability distributions",
            "statistical inference", "optimization theory",
            "number theory", "combinatorics",
            "graph theory", "information theory",
            "calculus of variations", "numerical methods",
            "Fourier analysis",
        ],
    },
    {
        "id": 9, "name": "Multimodal_Vision", "count": 200,
        "prompts": [
            "Explain in detail how a vision AI model would analyze a scene. Describe the objects, spatial relationships, lighting, and semantic understanding involved. Topic: {topic}.",
            "Describe a visual concept comprehensively in text. Include shape, color, texture, composition, and how a multimodal AI would process it. Topic: {topic}.",
            "Generate a detailed visual understanding essay. Cover object detection, segmentation, depth estimation, and scene graph generation for: {topic}.",
        ],
        "topics": [
            "object detection and recognition", "image segmentation",
            "scene understanding and reasoning", "visual question answering",
            "optical character recognition", "face detection and analysis",
            "depth estimation and 3D reconstruction", "image captioning",
            "visual grounding and referring expressions", "video understanding",
        ],
    },
    {
        "id": 10, "name": "Agentic_EN", "count": 500,
        "prompts": [
            "Generate a multi-step agent tool-use conversation. The agent has access to tools: web_search, calculator, code_interpreter, database_query, file_reader, email_sender. Show the user request, tool calls with results, and final response. Scenario: {topic}.",
            "Create an AI assistant conversation that uses multiple tools to solve a complex task. Include at least 3 tool calls with realistic results. Scenario: {topic}.",
            "Write an agent workflow that uses tool calls to achieve a goal. Show planning, execution, and error recovery. Scenario: {topic}.",
        ],
        "topics": [
            "researching a topic and summarizing findings",
            "calculating financial projections and sending an email report",
            "querying a database and generating a chart",
            "debugging code by reading files and running analysis",
            "planning a trip with weather checks and calculator",
            "analyzing survey data with statistical tools",
            "writing and testing a script for data processing",
            "comparing product prices and creating a summary table",
            "fetching news articles and extracting key insights",
            "solving a multi-step math word problem with verification",
            "multi-agent collaboration on a research task",
            "tool-use with error recovery and retry logic",
            "web scraping and data extraction pipeline",
            "automated report generation from multiple sources",
            "orchestrating cloud APIs for data processing",
        ],
    },
    {
        "id": 11, "name": "Multimodal_Audio", "count": 200,
        "prompts": [
            "Explain how an AI system processes audio and speech. Cover the acoustic features, model architecture, and recognition pipeline. Topic: {topic}.",
            "Describe audio processing concepts in detail including sampling, feature extraction, and model inference. Topic: {topic}.",
            "Generate a comprehensive explanation of speech/audio AI. Include signal processing, neural architectures, and applications. Topic: {topic}.",
        ],
        "topics": [
            "automatic speech recognition", "text-to-speech synthesis",
            "speaker identification and diarization", "audio event detection",
            "music information retrieval", "emotion recognition from speech",
            "voice activity detection", "audio source separation",
            "multilingual speech processing", "end-to-end speech models",
        ],
    },
    {
        "id": 12, "name": "AI_LLM_Research", "count": 1000,
        "prompts": [
            "Generate a comprehensive explanation of a key AI/ML concept. Cover the intuition, mathematical formulation, training methodology, and impact. Topic: {topic}.",
            "Write a survey-style overview of a deep learning area. Include key papers, architectural innovations, training techniques, and open challenges. Topic: {topic}.",
            "Create a detailed technical explanation of a neural network architecture. Describe the design motivation, components, training dynamics, and applications. Topic: {topic}.",
        ],
        "topics": [
            "transformer architecture and attention mechanisms", "large language model training and scaling",
            "reinforcement learning from human feedback (RLHF)", "retrieval-augmented generation (RAG)",
            "mixture-of-experts (MoE) architectures", "speculative decoding and draft models",
            "diffusion models for text generation", "multimodal LLMs and vision-language models",
            "parameter-efficient fine-tuning (LoRA, adapters)", "prompt engineering and in-context learning",
            "chain-of-thought reasoning", "tool-use and function calling in LLMs",
            "multi-agent systems and agentic workflows", "LLM benchmarking and evaluation",
            "quantization and model compression", "distributed training and data parallelism",
            "attention variants (multi-query, grouped-query, sliding window)", "KV cache optimization for inference",
            "alignment and safety in LLMs", "open-source LLM ecosystem (Llama, Mistral, Gemma, Qwen)",
        ],
    },
    {
        "id": 13, "name": "AI_Agent_Systems", "count": 500,
        "prompts": [
            "Generate a detailed explanation of an AI agent system architecture. Cover components, communication patterns, memory, and tool integration. Topic: {topic}.",
            "Write a comprehensive overview of agent framework design. Include planning, execution, reflection loops, and multi-agent coordination. Topic: {topic}.",
            "Create a technical deep-dive into agentic AI patterns. Describe the implementation, challenges, and best practices. Topic: {topic}.",
        ],
        "topics": [
            "autonomous agent architectures", "agent planning and reasoning loops",
            "tool-use and function calling pipelines", "multi-agent collaboration patterns",
            "agent memory and state management", "agent observability and debugging",
            "ReAct (Reasoning + Acting) framework", "tree-of-thoughts for agents",
            "agentic RAG and knowledge retrieval", "code generation and execution agents",
            "web navigation and automation agents", "conversational agent design",
            "agent safety and constraints", "evaluation of agent systems",
            "scaling agent systems to production",
        ],
    },
    {
        "id": 14, "name": "AI_ML_Papers", "count": 500,
        "prompts": [
            "Write a detailed paper summary and analysis. Cover the problem, proposed method, experimental setup, key results, and impact. Focus on: {topic}.",
            "Generate a critical review of a seminal AI paper. Explain the contribution, methodology, results, and why it matters. Paper focus: {topic}.",
            "Create a technical breakdown of a recent AI research result. Cover the architecture, training details, benchmarks, and ablation studies. Topic: {topic}.",
        ],
        "topics": [
            "Attention Is All You Need (Vaswani et al.)", "GPT-3: Language Models are Few-Shot Learners",
            "LLaMA: Open and Efficient Foundation Models", "Gemma: Open Models Based on Gemini Research",
            "DeepSeek-R1: Reasoning via Reinforcement Learning", "Mixture of Experts in Large Language Models",
            "RLHF and InstructGPT", "Direct Preference Optimization (DPO)",
            "Retrieval-Augmented Generation (Lewis et al.)", "Speculative Decoding for LLM Inference",
            "Eagle: Speculative Sampling with Prediction Heads", "Diffusion Models (Ho et al. DDPM)",
            "Vision Transformers (Dosovitskiy et al.)", "CLIP: Learning Visual Representations from Text",
            "Large Language Models are Zero-Shot Reasoners", "Tree of Thoughts (Yao et al.)",
            "ReAct: Synergizing Reasoning and Acting", "Constitutional AI and Safety",
            "FlashAttention: Fast and Memory-Efficient Attention", "LoRA: Low-Rank Adaptation of LLMs",
        ],
    },
    {
        "id": 15, "name": "Multimodal_Integration", "count": 300,
        "prompts": [
            "Explain how a multimodal AI integrates vision, language, and audio. Describe the architecture, fusion strategies, and training methodology. Topic: {topic}.",
            "Generate a detailed explanation of cross-modal learning. Cover alignment, contrastive learning, and joint embedding spaces. Topic: {topic}.",
            "Write about E2B (Everything-to-Bytes) multimodal architectures. Explain tokenization, attention across modalities, and generation. Topic: {topic}.",
        ],
        "topics": [
            "vision-language models and CLIP", "multimodal transformers",
            "cross-modal alignment and contrastive learning", "joint embedding spaces",
            "E2B unified multimodal tokenization", "multimodal generation and captioning",
            "video-language understanding", "multimodal retrieval",
            "zero-shot multimodal transfer", "scaling multimodal models",
        ],
    },
]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


_SERVER_PROCESS = None


def _start_server(model: str):
    """Start llama-server in background for persistent inference."""
    global _SERVER_PROCESS
    if _SERVER_PROCESS is not None:
        return
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sock.connect_ex(('127.0.0.1', 8081)) == 0:
        sock.close()
        return
    sock.close()
    server_exe = os.path.join(_LLAMA_DIR, "llama-server.exe")
    _SERVER_PROCESS = subprocess.Popen(
        [server_exe, "-m", model, "--port", "8081", "-ngl", "99", "--ctx-size", "4096"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import time
    time.sleep(3)


def _stop_server():
    global _SERVER_PROCESS
    if _SERVER_PROCESS is not None:
        _SERVER_PROCESS.terminate()
        _SERVER_PROCESS = None


def _call_openai_api(prompt: str, max_tokens: int, temp: float, port: int,
                     model_name: str = "local") -> str | None:
    """Try calling an OpenAI-compatible API on the given port."""
    import httpx
    try:
        messages = [
            {"role": "system", "content": (
                "You are a helpful AI assistant. Provide accurate, well-structured responses."
            )},
            {"role": "user", "content": prompt},
        ]
        r = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": 0.95,
        }, timeout=120)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        if content is None:
            return None
        end_marker = "<end_of_turn>"
        if end_marker in content:
            content = content.split(end_marker)[0].strip()
        return content.strip()
    except Exception:
        return None


def run_llama(prompt: str, max_tokens: int = 2048, temp: float = 0.7,
              model: str = "", use_server: bool = True) -> str:
    # Try local llama-server first (port 8080 — no content filters)
    if use_server:
        result = _call_openai_api(prompt, max_tokens, temp, 8080, model_name="local")
        if result:
            return result

    # Fallback: direct llama-cli invocation
    gemma_prompt = (
        "<start_of_turn>user\n"
        + prompt
        + "<end_of_turn>\n<start_of_turn>model\n"
    )
    cmd = [
        LLAMA_CLI,
        "-m", model or DEFAULT_MODEL,
        "-p", gemma_prompt,
        "-n", str(max_tokens),
        "--temp", str(temp),
        "--top-k", "40",
        "--top-p", "0.95",
        "--repeat-penalty", "1.1",
        "--no-display-prompt",
        "-e",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = r.stdout.strip()
        if not output:
            return ""
        end_marker = "<end_of_turn>"
        if end_marker in output:
            output = output.split(end_marker)[0].strip()
        return output
    except subprocess.TimeoutExpired:
        print(f"  [WARN] LLM generation timed out (300s)")
        return ""
    except FileNotFoundError:
        print(f"  [ERROR] llama-cli.exe not found at {LLAMA_CLI}")
        raise
    except Exception as e:
        print(f"  [ERROR] LLM invocation failed: {e}")
        return ""


def format_doc(topic_name: str, response: str, topic_str: str) -> str:
    header = topic_str.rstrip(".").strip().capitalize()
    tag = topic_name.lower().replace(" ", "_").replace("-", "_")
    lines = [
        f"# {topic_name}: {header}",
        "",
        "## Content",
        "",
        response.strip(),
        "",
        f"Tags: #llm #{tag} #up_to_date #generated",
    ]
    return "\n".join(lines)


def generate_dataset(model: str, output_dir: str, samples_per_topic: int | None = None,
                     phases_filter: list[str] | None = None, resume: bool = False):
    print("=" * 60)
    print("LLM Dataset Generator (local, no rate limits)")
    print(f" Model: {os.path.basename(model)}")
    print(f" Output: {output_dir}")
    if resume:
        print(" Resume mode: skipping existing files")
    print("=" * 60)
    print()

    _ensure_dir(output_dir)
    total = 0
    topics_to_gen = TOPICS

    if phases_filter:
        phase_names = {t["name"].lower(): t for t in TOPICS}
        topics_to_gen = []
        for p in phases_filter:
            if p.lower() in phase_names:
                topics_to_gen.append(phase_names[p.lower()])
            else:
                matches = [t for t in TOPICS if p.lower() in t["name"].lower()]
                if matches:
                    topics_to_gen.extend(matches)
                else:
                    print(f"  [WARN] Unknown phase '{p}', skipping")

    for topic_def in topics_to_gen:
        name = topic_def["name"]
        n = samples_per_topic or topic_def["count"]
        prompts = topic_def["prompts"]
        topics = topic_def["topics"]
        phase_dir = os.path.join(output_dir, f"phase{topic_def['id']:02d}_{name}")
        _ensure_dir(phase_dir)

        existing = set()
        if resume:
            for f in os.listdir(phase_dir):
                if f.endswith(".md"):
                    existing.add(f)

        print(f"Phase {topic_def['id']}: {name} ({n} samples, {len(existing)} existing)")

        for i in range(n):
            filename = f"{name.lower()}_llm_{i+1:04d}.md"
            if filename in existing:
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{n}] SKIP (exists)", end="\r")
                continue

            topic_str = topics[i % len(topics)]
            prompt_template = prompts[i % len(prompts)]
            full_prompt = prompt_template.format(topic=topic_str)
            full_prompt += "\n\nUse proper Markdown formatting with ## headings. Include the complete solution. Be comprehensive — produce at least 800 words of detailed content suitable for training a language model."

            print(f"  [{i+1}/{n}] {topic_str} ...", end=" ", flush=True)
            t0 = time.time()

            response = run_llama(
                prompt=full_prompt,
                max_tokens=4096,
                temp=0.7 + (i % 5) * 0.05,
                model=model,
                use_server=True,
            )

            elapsed = time.time() - t0
            if response and len(response) > 100:
                doc = format_doc(name, response, topic_str)
                filepath = os.path.join(phase_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(doc)
                total += 1
                print(f"OK ({len(response)}c, {elapsed:.1f}s)")
            else:
                print(f"SHORT ({len(response) if response else 0}c, {elapsed:.1f}s)")

        existing_count = len([f for f in os.listdir(phase_dir) if f.endswith(".md")])
        file_sizes = sum(os.path.getsize(os.path.join(phase_dir, f)) for f in os.listdir(phase_dir) if f.endswith(".md"))
        print(f"  -> {existing_count} files ({file_sizes >> 10} KB) in {phase_dir}")
        print()

    # Final summary
    total_files = 0
    total_bytes = 0
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".md"):
                total_files += 1
                total_bytes += os.path.getsize(os.path.join(root, f))

    print("=" * 60)
    print(f"LLM dataset generation complete — {total} new files")
    print(f" Total in {output_dir}: {total_files} files, {total_bytes >> 10} KB, {total_bytes >> 20} MB")
    print(f" Phases: {', '.join(t['name'] for t in TOPICS)}")
    print("=" * 60)
    return total


def multimodal_demo():
    """Demonstrate multimodal capabilities via llama-mtmd-cli and llama-tts."""
    print("=" * 60)
    print("Multimodal Capability Demo")
    print("=" * 60)

    if os.path.exists(LLAMA_MTMD):
        print(f"\n Vision: llama-mtmd-cli available at {LLAMA_MTMD}")
        print("  Usage: llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf --image photo.jpg -p 'Describe this image'")
    else:
        print("\n Vision: llama-mtmd-cli not found")

    if os.path.exists(LLAMA_TTS):
        print(f"\n Audio: llama-tts available at {LLAMA_TTS}")
        print("  Usage: llama-tts -m tts_model.gguf -p 'Hello world' --output speech.wav")
    else:
        print("\n Audio: llama-tts not found")

    print("\nText datasets for multimodal concepts are already being generated")
    print("in the Multimodal_Vision, Multimodal_Audio, and Multimodal_Integration phases.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate AGK datasets using local GGUF model (no rate limits)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory (default: agk_llm)")
    parser.add_argument("--samples", type=int, default=None, help="Samples per phase (default: per-phase count, up to 1000)")
    parser.add_argument("--phases", nargs="+", default=None,
                        help="Phases to generate: EN_Reasoning, EN_Coding, EN_Physics, AI_LLM_Research, etc.")
    parser.add_argument("--resume", action="store_true", help="Skip existing files")
    parser.add_argument("--server", action="store_true", help="Use llama-server for persistent inference (faster)")
    parser.add_argument("--multimodal-demo", action="store_true", help="Show multimodal tool capabilities")
    args = parser.parse_args()

    if args.server:
        _start_server(args.model or DEFAULT_MODEL)
        atexit.register(_stop_server)

    if args.multimodal_demo:
        multimodal_demo()
        return

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)

    generate_dataset(
        model=args.model,
        output_dir=args.output,
        samples_per_topic=args.samples,
        phases_filter=args.phases,
        resume=args.resume,
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
