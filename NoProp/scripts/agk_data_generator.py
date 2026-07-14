"""
AGK — Automated General Knowledge dataset generator for the NoProp Mesh.
Generates multi-phase, multi-language training data covering reasoning, coding,
maths, physics, agentic tool-use, and toolset definitions.

Phases:
  Phase 1  EN_Reasoning      English logical reasoning chains
  Phase 2  AR_Reasoning      Arabic logical reasoning chains
  Phase 3  EN_Coding         English code snippets + explanations
  Phase 4  AR_Coding          Arabic coding content
  Phase 5  EN_Maths          English math problems + solutions
  Phase 6  AR_Maths          Arabic math problems + solutions
  Phase 7  Agentic_EN        English agent/tool-use conversations
  Phase 8  Agentic_AR        Arabic agent/tool-use conversations
  Phase 9  Toolsets          Tool definitions, function schemas, API docs
  Phase 10 EN_Physics        English physics problems + solutions
  Phase 11 Web_Supplement    Web-fetched real-world data (if online)
"""
import os
import sys
import json
import math
import random
import hashlib
import textwrap
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_HTTP = None
try:
    import httpx
    _HTTP = "httpx"
except ImportError:
    try:
        import requests
        _HTTP = "requests"
    except ImportError:
        _HTTP = None


# =============================================================================
# Utility helpers
# =============================================================================

def _fetch(url: str, timeout: float = 15.0) -> str | None:
    if _HTTP == "httpx":
        try:
            r = httpx.get(url, timeout=timeout, follow_redirects=True)
            r.raise_for_status()
            return r.text
        except Exception:
            return None
    elif _HTTP == "requests":
        try:
            import requests
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception:
            return None
    return None


def _fetch_json(url: str, timeout: float = 15.0) -> dict | list | None:
    text = _fetch(url, timeout)
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return None


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_doc(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip())
        f.write("\n")


def _uid() -> str:
    return hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]


# =============================================================================
# Phase 1 — EN Reasoning
# =============================================================================

EN_REASONING_TEMPLATES = [
    "Logical deduction problem: {premises}\n\nChain of thought:\n{steps}\n\nConclusion: {conclusion}",
    "If {a} and {b}, then what follows?\n\n{explanation}\n\nAnswer: {answer}",
    "Consider the following statements:\n{statements}\n\nReason step by step:\n{reasoning}\n\nTherefore: {conclusion}",
]


def _generate_en_reasoning(n: int) -> list[str]:
    docs = []
    topics = [
        ("all men are mortal", "Socrates is a man", "Socrates is mortal"),
        ("if it rains the ground gets wet", "it is raining", "the ground is wet"),
        ("all birds have feathers", "penguins are birds", "penguins have feathers"),
        ("every square is a rectangle", "shape X is a square", "shape X is a rectangle"),
        ("if a number is even it is divisible by 2", "10 is even", "10 is divisible by 2"),
        ("all mammals breathe air", "whales are mammals", "whales breathe air"),
        ("if a triangle has three equal sides it is equilateral",
         "triangle T has three equal sides", "triangle T is equilateral"),
        ("every prime greater than 2 is odd", "7 is prime and greater than 2", "7 is odd"),
    ]
    for i in range(n):
        a, b, c = topics[i % len(topics)]
        doc = textwrap.dedent(f"""\
        # Reasoning Problem {i+1}

        Premise 1: {a}
        Premise 2: {b}

        Step 1: Identify the logical form — modus ponens from Premise 1 and Premise 2.
        Step 2: Apply the rule: if P→Q and P, then Q.
        Step 3: {c}.

        Conclusion: {c}
        This is a valid deductive argument. If both premises are true, the conclusion must be true.
        """)
        docs.append(doc)
    return docs


# =============================================================================
# Phase 2 — AR Reasoning
# =============================================================================

AR_REASONING_TOPICS = [
    ("كل الطيور لها ريش", "العصفور طائر", "العصفور له ريش", "الطيور", "birds"),
    ("كل المستطيلات لها أربع زوايا قائمة", "المربع مستطيل", "المربع له أربع زوايا قائمة", "المستطيلات", "rectangles"),
    ("إذا كان العدد زوجي فهو يقبل القسمة على 2", "العدد 14 زوجي", "العدد 14 يقبل القسمة على 2", "الأعداد الزوجية", "even numbers"),
    ("جميع الثدييات تتنفس الهواء", "الحوت ثديي", "الحوت يتنفس الهواء", "الثدييات", "mammals"),
    ("كل المربعات هي متوازيات أضلاع", "الشكل س مربع", "الشكل س متوازي أضلاع", "الأشكال الهندسية", "geometry"),
]


def _generate_ar_reasoning(n: int) -> list[str]:
    docs = []
    for i in range(n):
        premise1, premise2, conclusion, tag, en_tag = AR_REASONING_TOPICS[i % len(AR_REASONING_TOPICS)]
        doc = textwrap.dedent(f"""\
        # مسألة استدلال منطقي {i+1}

        المقدمة الأولى: {premise1}
        المقدمة الثانية: {premise2}

        خطوات الاستدلال:
        ١. تحديد الشكل المنطقي: قياس اقتراني شرطي من المقدمة الأولى والثانية.
        ٢. تطبيق القاعدة: إذا كان P→Q و P فإن Q.
        ٣. النتيجة: {conclusion}

        إذن: {conclusion}
        هذه حجة استنتاجية صحيحة. إذا كانت المقدمتان صحيحتين، فإن النتيجة صحيحة حتماً.

        الوسوم: #{tag} #{en_tag} #استدلال_منطقي #arabic #reasoning
        """)
        docs.append(doc)
    return docs


# =============================================================================
# Phase 3 — EN Coding
# =============================================================================

EN_CODE_PROBLEMS = [
    {
        "title": "Reverse a linked list",
        "lang": "python",
        "code": "def reverse_linked_list(head):\n    prev = None\n    curr = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev",
        "explanation": "Iteratively reverse pointers. Maintain prev/curr/nxt pointers and flip each node's next pointer backward.",
    },
    {
        "title": "Binary search",
        "lang": "python",
        "code": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        "explanation": "Divide and conquer on sorted array. Repeatedly halve the search space by comparing middle element with target.",
    },
    {
        "title": "Quick sort",
        "lang": "python",
        "code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)",
        "explanation": "Recursively partition around a pivot element. Elements smaller than pivot go left, larger go right.",
    },
    {
        "title": "Breadth-first search",
        "lang": "python",
        "code": "def bfs(graph, start):\n    visited = set()\n    queue = [start]\n    visited.add(start)\n    while queue:\n        node = queue.pop(0)\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return visited",
        "explanation": "Level-order traversal using a queue. Explore all neighbors at the current depth before moving deeper.",
    },
    {
        "title": "Merge sort",
        "lang": "python",
        "code": "def mergesort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = mergesort(arr[:mid])\n    right = mergesort(arr[mid:])\n    return merge(left, right)\n\ndef merge(a, b):\n    res = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] < b[j]:\n            res.append(a[i]); i += 1\n        else:\n            res.append(b[j]); j += 1\n    return res + a[i:] + b[j:]",
        "explanation": "Divide and conquer. Split array in half, recursively sort each half, then merge the sorted halves.",
    },
    {
        "title": "Dijkstra's shortest path",
        "lang": "python",
        "code": "import heapq\n\ndef dijkstra(graph, start):\n    dist = {node: float('inf') for node in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in graph[u]:\n            nd = d + w\n            if nd < dist[v]:\n                dist[v] = nd\n                heapq.heappush(pq, (nd, v))\n    return dist",
        "explanation": "Greedy shortest-path algorithm using a priority queue. Always expand the node with the smallest known distance.",
    },
    {
        "title": "Dynamic programming — Fibonacci",
        "lang": "python",
        "code": "def fib(n):\n    if n <= 1:\n        return n\n    dp = [0] * (n + 1)\n    dp[1] = 1\n    for i in range(2, n + 1):\n        dp[i] = dp[i - 1] + dp[i - 2]\n    return dp[n]",
        "explanation": "Bottom-up dynamic programming. Build the solution from base cases using previously computed subproblems.",
    },
    {
        "title": "Two-sum problem",
        "lang": "python",
        "code": "def two_sum(nums, target):\n    seen = {}\n    for i, v in enumerate(nums):\n        complement = target - v\n        if complement in seen:\n            return [seen[complement], i]\n        seen[v] = i\n    return []",
        "explanation": "Use a hash map to track seen values. For each element, check if its complement (target - value) has been seen.",
    },
]


def _generate_en_coding(n: int) -> list[str]:
    docs = []
    for i in range(n):
        p = EN_CODE_PROBLEMS[i % len(EN_CODE_PROBLEMS)]
        doc = textwrap.dedent(f"""\
        # Coding: {p['title']}

        ## Problem
        Implement {p['title'].lower()} in {p['lang']}.

        ## Solution
        ```{p['lang']}
        {p['code']}
        ```

        ## Explanation
        {p['explanation']}

        ## Complexity
        - Time: O(n) or O(n log n) depending on implementation
        - Space: O(n) in the worst case

        Tags: #{p['lang']} #coding #algorithms #{p['title'].lower().replace(' ', '_')}
        """)
        docs.append(doc)
    return docs


# =============================================================================
# Phase 4 — AR Coding
# =============================================================================

AR_CODE_INTROS = [
    "عكس قائمة مرتبطة: استخدم مؤشرات prev و curr و nxt لعكس اتجاه الروابط.",
    "البحث الثنائي: قسم المصفوفة المرتبة إلى نصفين متكررين حتى العثور على الهدف.",
    "الفرز السريع: اختر عنصر محور وقسم المصفوفة حوله.",
    "البحث بالعرض: استخدم طابور لاستكشاف الرسم البياني مستوى بمستوى.",
    "الفرز بالدمج: قسم المصفوفة إلى نصفين ورتب كل نصف ثم ادمجهم.",
]


def _generate_ar_coding(n: int) -> list[str]:
    docs = []
    for i in range(n):
        p = EN_CODE_PROBLEMS[i % len(EN_CODE_PROBLEMS)]
        intro = AR_CODE_INTROS[i % len(AR_CODE_INTROS)]
        doc = textwrap.dedent(f"""\
        # برمجة: {p['title']}

        ## المسألة
        {intro}

        ## الحل بلغة {p['lang']}
        ```{p['lang']}
        {p['code']}
        ```

        ## شرح
        {p['explanation']}

        الوسوم: #{p['lang']} #برمجة #خوارزميات #{p['title'].lower().replace(' ', '_')} #arabic #coding
        """)
        docs.append(doc)
    return docs


# =============================================================================
# Phase 5 — EN Maths
# =============================================================================

MATH_PROBLEMS = [
    ("derivative of x^2", "2x", "Apply the power rule: d/dx(x^n) = n*x^(n-1)"),
    ("integral of 2x dx", "x^2 + C", "Apply the power rule in reverse: ∫x^n dx = x^(n+1)/(n+1) + C"),
    ("solve 2x + 5 = 13", "x = 4", "Subtract 5 from both sides → 2x = 8 → divide by 2 → x = 4"),
    ("probability of rolling a 6 on a fair die", "1/6", "One favorable outcome out of six equally likely outcomes"),
    ("area of a circle with radius 3", "9π ≈ 28.27", "A = πr² = π(3)² = 9π"),
    ("Pythagorean theorem: find hypotenuse if legs are 3 and 4", "5", "c² = a² + b² = 9 + 16 = 25 → c = 5"),
    ("log₂(32)", "5", "2⁵ = 32, therefore log₂(32) = 5"),
    ("sum of angles in a triangle", "180°", "For any triangle, interior angles sum to 180 degrees"),
    ("derivative of sin(x)", "cos(x)", "Standard trigonometric derivative from the limit definition"),
    ("solve x² - 4 = 0", "x = 2 or x = -2", "Factor: (x-2)(x+2) = 0 → x = 2 or x = -2"),
]


def _generate_en_maths(n: int) -> list[str]:
    docs = []
    for i in range(n):
        problem, answer, steps = MATH_PROBLEMS[i % len(MATH_PROBLEMS)]
        doc = textwrap.dedent(f"""\
        # Maths Problem {i+1}

        Problem: {problem}

        Step-by-step:
        {steps}

        Answer: {answer}

        Tags: #mathematics #problem_solving #{'algebra' if i % 2 == 0 else 'geometry'} #{'calculus' if i < 3 else 'algebra'}
        """)
        docs.append(doc)
    return docs


# =============================================================================
# Phase 6 — AR Maths
# =============================================================================

AR_MATH_PROBLEMS = [
    ("اشتقاق x²", "2x", "طبق قاعدة القوة: d/dx(xⁿ) = n·xⁿ⁻¹"),
    ("تكامل 2x dx", "x² + C", "طبق قاعدة القوة العكسية: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C"),
    ("حل المعادلة 2x + 5 = 13", "x = 4", "اطرح 5 من الطرفين → 2x = 8 → اقسم على 2 → x = 4"),
    ("احتمال ظهور الرقم 6 عند رمي حجر نرد", "1/6", "نتيجة واحدة مواتية من ست نتائج متساوية الاحتمال"),
    ("مساحة دائرة نصف قطرها 3", "9π ≈ 28.27", "المساحة = πr² = π × 9 = 9π"),
]


def _generate_ar_maths(n: int) -> list[str]:
    docs = []
    for i in range(n):
        problem, answer, steps = AR_MATH_PROBLEMS[i % len(AR_MATH_PROBLEMS)]
        doc = textwrap.dedent(f"""\
        # مسألة رياضيات {i+1}

        المسألة: {problem}

        خطوات الحل:
        {steps}

        الإجابة: {answer}

        الوسوم: #رياضيات #حل_مسائل #arabic #maths
        """)
        docs.append(doc)
    return docs


# =============================================================================
# Phase 7 — Agentic EN (tool-use conversations)
# =============================================================================

AGENTIC_EN_TEMPLATES = [
    {
        "tools": ["web_search", "calculator", "code_interpreter"],
        "conversation": [
            ("user", "What's 3847 * 9521 and then search for the current population of Tokyo?"),
            ("assistant", "I'll handle both tasks. First, let me calculate.\n\n```tool\ncalculator.run(expression='3847 * 9521')\n```\nResult: 36,630,287\n\nNow searching for Tokyo's population:\n```tool\nweb_search.query('current population of Tokyo 2024')\n```\nTokyo has approximately 14 million residents in the 23 special wards, and over 37 million in the greater metropolitan area.",
             "The assistant used two tools: calculator for arithmetic and web_search for factual lookup."),
        ],
    },
    {
        "tools": ["code_interpreter", "file_reader"],
        "conversation": [
            ("user", "Read the CSV file and plot the data."),
            ("assistant", "Let me read the file first.\n\n```tool\nfile_reader.read(path='data.csv')\n```\nFound 3 columns: date, sales, profit with 100 rows.\n\nNow plotting:\n```tool\ncode_interpreter.run(code='''\nimport pandas as pd\nimport matplotlib.pyplot as plt\ndf = pd.read_csv('data.csv')\nplt.plot(df['date'], df['sales'], label='Sales')\nplt.plot(df['date'], df['profit'], label='Profit')\nplt.legend()\nplt.savefig('output.png')\n''')\n```\nPlot saved to output.png.",
             "The assistant used file_reader to inspect the data, then code_interpreter to generate a plot."),
        ],
    },
    {
        "tools": ["web_search", "summarizer", "translator"],
        "conversation": [
            ("user", "Find the latest research on LLM agents and summarize it in Arabic."),
            ("assistant", "Searching for recent research:\n\n```tool\nweb_search.query('LLM agents recent research 2025 2026')\n```\nFound 3 relevant papers.\n\nSummarizing:\n```tool\nsummarizer.summarize(text=research_texts, max_length=200)\n```\nRecent advances include tool-augmented LLMs, multi-agent collaboration frameworks, and memory-augmented agents.\n\nTranslating to Arabic:\n```tool\ntranslator.translate(text=summary, target='ar')\n```\nالترجمة: تشمل التطورات الحديثة تعزيز النماذج اللغوية الكبيرة بالأدوات، أطر التعاون متعدد الوكلاء، والوكلاء المعززين بالذاكرة.",
             "The assistant chained three tools: web_search → summarizer → translator for a complex multi-step task."),
        ],
    },
    {
        "tools": ["database_query", "chart_generator", "email_sender"],
        "conversation": [
            ("user", "Get Q3 sales data, create a bar chart, and email it to the team."),
            ("assistant", "Querying the database:\n\n```tool\ndatabase_query.run(sql='SELECT region, SUM(sales) FROM sales WHERE quarter=3 GROUP BY region')\n```\nGot data for 4 regions.\n\nGenerating chart:\n```tool\nchart_generator.create(kind='bar', data=results, title='Q3 Sales by Region')\n```\nChart saved.\n\nSending email:\n```tool\nemail_sender.send(to='team@company.com', subject='Q3 Sales Report', attachments=['q3_sales.png'])\n```\nEmail sent successfully.",
             "The assistant orchestrated a 3-tool pipeline with data flowing between steps."),
        ],
    },
]


def _generate_agentic_en(n: int) -> list[str]:
    docs = []
    for i in range(n):
        t = AGENTIC_EN_TEMPLATES[i % len(AGENTIC_EN_TEMPLATES)]
        lines = [f"# Agent Conversation {i+1}", "",
                 f"Available tools: {', '.join(t['tools'])}", ""]
        analysis_parts = []
        for turn in t["conversation"]:
            role, msg = turn[0], turn[1]
            analysis = turn[2] if len(turn) > 2 else ""
            lines.append(f"**{role.capitalize()}**: {msg}")
            lines.append("")
            if analysis:
                analysis_parts.append(analysis)
        if analysis_parts:
            lines.append("--- Analysis ---")
            for a in analysis_parts:
                lines.append(a)
        lines.append("")
        lines.append(f"Tags: #agentic #tool_use #{' '.join(t['tools'])} #conversation")
        docs.append("\n".join(lines))
    return docs


# =============================================================================
# Phase 8 — Agentic AR
# =============================================================================

AR_AGENTIC_SCENARIOS = [
    {
        "scenario": "بحث وحساب",
        "tools": ["بحث_الويب", "آلة_حاسبة"],
        "turns": [
            ("مستخدم", "احسب ٣٨٤٧ × ٩٥٢١ ثم ابحث عن عدد سكان طوكيو"),
            ("مساعد", "سأحسب أولاً:\n\n```tool\nآلة_حاسبة.احسب(expression='3847 * 9521')\n```\nالنتيجة: ٣٦,٦٣٠,٢٨٧\n\nسأبحث الآن:\n```tool\nبحث_الويب.استعلم(query='عدد سكان طوكيو 2024')\n```\nيبلغ عدد سكان طوكيو حوالي ١٤ مليون نسمة في الأحياء الـ ٢٣ الخاصة."),
        ],
    },
]


def _generate_agentic_ar(n: int) -> list[str]:
    docs = []
    for i in range(n):
        t = AR_AGENTIC_SCENARIOS[i % len(AR_AGENTIC_SCENARIOS)]
        lines = [f"# محادثة وكيل {i+1}", "",
                 f"الأدوات المتاحة: {', '.join(t['tools'])}", ""]
        for role, msg in t["turns"]:
            lines.append(f"**{role}**: {msg}")
            lines.append("")
        lines.append(f"الوسوم: #وكيل #استخدام_أدوات #{' '.join(t['tools'])} #محادثة #arabic #agentic")
        docs.append("\n".join(lines))
    return docs


# =============================================================================
# Phase 9 — Toolsets (schemas, definitions, API docs)
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": "Search the web for current information",
        "schema": json.dumps({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }, indent=2),
        "example": 'web_search(query="latest AI research 2026")',
    },
    {
        "name": "code_interpreter",
        "description": "Execute Python code in a sandboxed environment",
        "schema": json.dumps({
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "Execute Python code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"}
                    },
                    "required": ["code"]
                }
            }
        }, indent=2),
        "example": 'code_interpreter(code="print(sum(range(100)))")',
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "schema": json.dumps({
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate mathematical expressions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        }, indent=2),
        "example": "calculator(expression='2**10 - 1')",
    },
    {
        "name": "translator",
        "description": "Translate text between languages",
        "schema": json.dumps({
            "type": "function",
            "function": {
                "name": "translator",
                "description": "Translate text to a target language",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "target": {"type": "string", "enum": ["en", "ar", "fr", "de", "ja", "zh"]}
                    },
                    "required": ["text", "target"]
                }
            }
        }, indent=2),
        "example": 'translator(text="Hello world", target="ar")',
    },
    {
        "name": "database_query",
        "description": "Query a SQL database",
        "schema": json.dumps({
            "type": "function",
            "function": {
                "name": "database_query",
                "description": "Execute SQL query against the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL query"}
                    },
                    "required": ["sql"]
                }
            }
        }, indent=2),
        "example": "database_query(sql='SELECT * FROM users LIMIT 10')",
    },
    {
        "name": "file_reader",
        "description": "Read files from the filesystem",
        "schema": json.dumps({
            "type": "function",
            "function": {
                "name": "file_reader",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"}
                    },
                    "required": ["path"]
                }
            }
        }, indent=2),
        "example": 'file_reader(path="config.json")',
    },
    {
        "name": "email_sender",
        "description": "Send emails with optional attachments",
        "schema": json.dumps({
            "type": "function",
            "function": {
                "name": "email_sender",
                "description": "Send an email message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string", "optional": True},
                        "attachments": {"type": "array", "items": {"type": "string"}, "optional": True}
                    },
                    "required": ["to", "subject"]
                }
            }
        }, indent=2),
        "example": 'email_sender(to="user@example.com", subject="Report", attachments=["report.pdf"])',
    },
    {
        "name": "chart_generator",
        "description": "Generate charts and visualizations",
        "schema": json.dumps({
            "type": "function",
            "function": {
                "name": "chart_generator",
                "description": "Create a chart from data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["bar", "line", "scatter", "pie"]},
                        "data": {"type": "object"},
                        "title": {"type": "string"}
                    },
                    "required": ["kind", "data"]
                }
            }
        }, indent=2),
        "example": 'chart_generator(kind="bar", data={"x": [1,2,3], "y": [4,5,6]}, title="Demo")',
    },
    {
        "name": "summarizer",
        "description": "Summarize long text passages",
        "schema": json.dumps({
            "type": "function",
            "function": {
                "name": "summarizer",
                "description": "Generate a concise summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "max_length": {"type": "integer", "optional": True}
                    },
                    "required": ["text"]
                }
            }
        }, indent=2),
        "example": 'summarizer(text="Long document...", max_length=100)',
    },
]


def _generate_toolsets(n: int) -> list[str]:
    docs = []
    for i in range(n):
        td = TOOL_DEFINITIONS[i % len(TOOL_DEFINITIONS)]
        doc = textwrap.dedent(f"""\
        # Tool: {td['name']}

        ## Description
        {td['description']}

        ## API Schema (OpenAI-compatible function calling format)
        ```json
        {td['schema']}
        ```

        ## Usage Example
        ```
        {td['example']}
        ```

        ## Best Practices
        - Always validate inputs before calling the tool
        - Handle errors gracefully and report them to the user
        - Chain multiple tool calls when a task requires it
        - Cache results when appropriate to reduce latency

        Tags: #tool #{td['name']} #api #function_calling #toolset
        """)
        docs.append(doc)
    return docs


# =============================================================================
# Phase 10 — EN Physics
# =============================================================================

PHYSICS_PROBLEMS = [
    ("A 2 kg block slides down a frictionless incline of 30 degrees. What is its acceleration?",
     "4.9 m/s\u00b2",
     "Force along incline = mg\u00b7sin(\u03b8) = 2\u00b79.8\u00b7sin(30\u00b0) = 9.8 N; a = F/m = 9.8/2 = 4.9 m/s\u00b2"),
    ("How much heat is needed to raise 500 g of water from 20\u00b0C to 80\u00b0C? (c_water = 4186 J/kg\u00b7K)",
     "125.58 kJ",
     "Q = mc\u0394T = 0.5\u00b74186\u00b760 = 125,580 J = 125.58 kJ"),
    ("A 12 V battery is connected to a 4 \u03a9 resistor. What is the current and power dissipated?",
     "I = 3 A, P = 36 W",
     "I = V/R = 12/4 = 3 A; P = I\u00b2R = 9\u00b74 = 36 W"),
    ("Calculate the wavelength of a photon with energy 3.0 eV. (1 eV = 1.6e-19 J, h = 6.63e-34 J\u00b7s, c = 3e8 m/s)",
     "414 nm",
     "E = 3.0\u00b71.6e-19 = 4.8e-19 J; \u03bb = hc/E = (6.63e-34\u00b73e8)/(4.8e-19) = 4.14e-7 m = 414 nm"),
    ("A ball is thrown upward at 15 m/s from ground level. How high does it go? (g = 9.8 m/s\u00b2)",
     "11.48 m",
     "v\u00b2 = u\u00b2 + 2as \u2192 0 = 15\u00b2 + 2(-9.8)h \u2192 h = 225/(19.6) = 11.48 m"),
    ("What is the kinetic energy of a 1000 kg car moving at 20 m/s?",
     "200 kJ",
     "KE = \u00bdmv\u00b2 = 0.5\u00b71000\u00b7400 = 200,000 J = 200 kJ"),
    ("A gas expands from 2 L to 5 L at constant pressure of 1 atm. How much work is done? (1 atm\u00b7L = 101.325 J)",
     "303.98 J",
     "W = P\u0394V = 1\u00b73 = 3 atm\u00b7L = 3\u00b7101.325 = 303.98 J"),
    ("Calculate the gravitational force between two 50 kg masses 1 m apart. (G = 6.67e-11 N\u00b7m\u00b2/kg\u00b2)",
     "1.67e-7 N",
     "F = G\u00b7m\u2081m\u2082/r\u00b2 = 6.67e-11\u00b72500/1 = 1.6675e-7 N"),
    ("How long does it take for a 10 \u03bcF capacitor to charge to 63% through a 1 M\u03a9 resistor?",
     "10 s",
     "\u03c4 = RC = 1e6\u00b710e-6 = 10 s; one time constant charges to 63%"),
    ("A 2 m long wire carries 5 A perpendicular to a 0.1 T magnetic field. What magnetic force acts on it?",
     "1 N",
     "F = BIL = 0.1\u00b75\u00b72 = 1 N"),
    ("Calculate the escape velocity of Earth. (M = 5.97e24 kg, R = 6.37e6 m, G = 6.67e-11)",
     "11.2 km/s",
     "v_esc = \u221a(2GM/R) = \u221a(2\u00b76.67e-11\u00b75.97e24/6.37e6) = \u221a(1.25e8) = 1.12e4 m/s = 11.2 km/s"),
    ("A 5 kg object experiences a net force of 20 N. What is its acceleration?",
     "4 m/s\u00b2",
     "F = ma \u2192 a = F/m = 20/5 = 4 m/s\u00b2"),
    ("What is the momentum of a 60 kg person walking at 1.5 m/s?",
     "90 kg\u00b7m/s",
     "p = mv = 60\u00b71.5 = 90 kg\u00b7m/s"),
    ("A spring with k = 500 N/m is compressed by 0.1 m. What elastic potential energy is stored?",
     "2.5 J",
     "PE = \u00bdkx\u00b2 = 0.5\u00b7500\u00b70.01 = 2.5 J"),
    ("Calculate the period of a 1 m long pendulum. (g = 9.8 m/s\u00b2)",
     "2.01 s",
     "T = 2\u03c0\u221a(L/g) = 2\u03c0\u221a(1/9.8) = 2\u03c0\u00b70.319 = 2.01 s"),
]


def _generate_en_physics(n: int) -> list[str]:
    docs = []
    for i in range(n):
        problem, answer, steps = PHYSICS_PROBLEMS[i % len(PHYSICS_PROBLEMS)]
        doc = textwrap.dedent(f"""\
        # Physics Problem {i+1}

        Problem: {problem}

        Step-by-step:
        {steps}

        Answer: {answer}

        Tags: #physics #{'mechanics' if i % 3 == 0 else 'thermodynamics' if i % 3 == 1 else 'electromagnetism'} #problem_solving
        """)
        docs.append(doc)
    return docs


# =============================================================================
# Phase 11 — Web Supplement (live internet fetch)
# =============================================================================

WEB_SOURCES = [
    {
        "name": "GitHub trending",
        "url": "https://api.github.com/search/repositories?q=language:python+created:>2025-01-01&sort=stars&per_page=5",
        "parser": "json",
    },
    {
        "name": "HuggingFace papers",
        "url": "https://huggingface.co/api/daily_papers?limit=5",
        "parser": "json",
    },
]


def _fetch_web_supplement(output_dir: str) -> int:
    count = 0
    for source in WEB_SOURCES:
        print(f"  Fetching {source['name']}...")
        data = _fetch_json(source["url"])
        if data is None:
            print(f"    (skipped — no connection)")
            continue
        filename = f"web_{source['name'].lower().replace(' ', '_')}.json"
        filepath = os.path.join(output_dir, filename)
        content = json.dumps(data, indent=2, ensure_ascii=False)
        _write_doc(filepath, content)
        count += 1
        print(f"    saved {filepath} ({len(content)} bytes)")
    return count


# =============================================================================
# Main orchestrator
# =============================================================================

PHASES = {
    1:  ("EN_Reasoning",   _generate_en_reasoning,   50),
    2:  ("AR_Reasoning",   _generate_ar_reasoning,   25),
    3:  ("EN_Coding",      _generate_en_coding,      50),
    4:  ("AR_Coding",      _generate_ar_coding,      15),
    5:  ("EN_Maths",       _generate_en_maths,       40),
    6:  ("AR_Maths",       _generate_ar_maths,       20),
    7:  ("Agentic_EN",     _generate_agentic_en,     20),
    8:  ("Agentic_AR",     _generate_agentic_ar,     10),
    9:  ("Toolsets",       _generate_toolsets,       20),
    10: ("EN_Physics",     _generate_en_physics,      40),
    11: ("Web_Supplement", None,                       1),
}


def run_all(output_dir: str, phases: list[int] | None = None,
            samples_per_phase: int | None = None, skip_web: bool = False):
    print("=" * 60)
    print("AGK Data Generator — Automated General Knowledge")
    print("=" * 60)
    print()

    _ensure_dir(output_dir)
    phase_ids = phases or sorted(PHASES.keys())

    total_files = 0
    for pid in phase_ids:
        if pid not in PHASES:
            print(f"Unknown phase {pid}, skipping")
            continue

        name, generator, default_n = PHASES[pid]
        n = samples_per_phase or default_n

        if pid == 11 and skip_web:
            print(f"Phase 11 (Web_Supplement): skipped (--skip-web)")
            continue

        phase_dir = os.path.join(output_dir, f"phase{pid:02d}_{name}")
        _ensure_dir(phase_dir)

        print(f"Phase {pid}/10: {name} ({n} samples)")

        if generator is not None:
            docs = generator(n)
            for idx, doc in enumerate(docs):
                filename = f"{name.lower()}_{idx+1:04d}.md"
                _write_doc(os.path.join(phase_dir, filename), doc)
            total_files += len(docs)
            print(f"  -> {len(docs)} files written to {phase_dir}")

        elif pid == 10:
            count = _fetch_web_supplement(phase_dir)
            total_files += count

        print()

    print("=" * 60)
    print(f"AGK generation complete — {total_files} files in {output_dir}")
    print("Run train_from_text.bat to train the mesh on this data.")
    print("=" * 60)
    return total_files


if __name__ == "__main__":
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    import argparse
    parser = argparse.ArgumentParser(description="AGK Data Generator for NoProp Mesh")
    parser.add_argument("--output", default="agk_data",
                        help="Output directory (default: agk_data)")
    parser.add_argument("--phases", type=int, nargs="+", default=None,
                        help="Phases to run, e.g. --phases 1 3 5 (default: all 11)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Samples per phase (default: phase-specific)")
    parser.add_argument("--skip-web", action="store_true",
                        help="Skip Phase 11 (web supplement)")
    args = parser.parse_args()

    run_all(
        output_dir=args.output,
        phases=args.phases,
        samples_per_phase=args.samples,
        skip_web=args.skip_web,
    )
