<div align="center">

# LangChain Agents — From Tools to Autonomous ReAct Agents

### A hands-on journey through **Tools → Tool Calling → AI Agents** in LangChain

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Agents-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

*Notes, code, and diagrams from my journey learning LangChain Agents.*

</div>

---

## Table of Contents

1. [About This Repo](#about-this-repo)
2. [Learning Roadmap](#learning-roadmap)
3. [What You'll Learn](#what-youll-learn)
4. [Repo Structure](#repo-structure)
5. [Part 1 — Tools](#part-1--tools)
6. [Part 2 — Tool Calling](#part-2--tool-calling)
7. [Part 3 — AI Agents](#part-3--ai-agents)
8. [The ReAct Pattern](#the-react-pattern)
9. [Agent Executor Flow](#agent-executor-flow)
10. [Getting Started](#getting-started)
11. [Tech Stack](#tech-stack)

---

## About This Repo

This repository is my personal lab for understanding **how LLMs stop being "just chatbots"** and start becoming **autonomous agents** that can think, decide, and act using external tools.

Every notebook builds on the previous one — from the smallest building block (a **Tool**) up to a fully working **ReAct AI Agent** that reasons step-by-step, calls tools, observes results, and produces a final answer.

> **An AI agent is an LLM-powered system that can autonomously think, decide, and take actions using external tools or APIs to achieve a goal.**

---

## Learning Roadmap

```mermaid
flowchart LR
    A["01 · Tools<br/>Built-in & Custom"] --> B["02 · Tool Calling<br/>Bind → Call → Execute"]
    B --> C["03 · AI Agent<br/>ReAct + AgentExecutor"]

    style A fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style B fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    style C fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f
```

---

## What You'll Learn

| Concept | What It Means |
|---|---|
| **Tools** | Python functions / APIs packaged so the LLM can call them |
| **Built-in Tools** | Pre-built, production-ready tools shipped by LangChain |
| **Custom Tools** | Your own functions exposed to the LLM via `@tool`, `StructuredTool`, `BaseTool` |
| **Toolkits** | Bundled collections of related tools (e.g. `GoogleDriveToolkit`) |
| **Tool Binding** | Registering tools with an LLM so it knows their name, description, and schema |
| **Tool Calling** | The LLM decides *which* tool to call and *with what arguments* — it does not execute |
| **Tool Execution** | You (or LangChain) actually run the Python function with the LLM's suggested args |
| **AI Agent** | Goal-driven, autonomous, tool-using, context-aware, reasoning-capable, adaptive |
| **ReAct** | **Rea**soning + **Act**ing — interleaves `Thought → Action → Observation` loops |
| **AgentExecutor** | The orchestrator that runs the ReAct loop until the agent produces a final answer |

---

## Repo Structure

```
LangChain_Agents/
├── 01_LangChain_Tools/
│   ├── LangChain_Built-in_Tools.ipynb      # DuckDuckGo search, Shell, etc.
│   └── LangChain_Custom_Tools.ipynb        # @tool, StructuredTool, BaseTool
│
├── 02_LangChain_Tool_Calling/
│   ├── LangChain_Tool_Calling.ipynb        # Binding + calling + executing
│   └── LangChain_Tool_Calling_Currency_Converter.ipynb
│
├── 03_LangChain_AI_Agent/
│   └── ai_agent_in_langchain.ipynb         # ReAct agent + AgentExecutor
│
└── requirements.txt
```

---

## Part 1 — Tools

> A **Tool** is just a Python function (or API) packaged in a way the LLM can understand and call when needed.

LLMs are great at **reasoning (thinking)** and **language generation (speaking)** — but on their own they *can't* access live data, do reliable math, call APIs, run code, or query databases. **Tools are the bridge.**

```mermaid
flowchart LR
    LLM["LLM<br/><i>good at reasoning & speaking</i>"]
    LLM --> T1["DuckDuckGo Search"]
    LLM --> T2["Shell Tool"]
    LLM --> T3["External APIs"]
    LLM --> T4["Databases"]

    style LLM fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f
    style T1 fill:#d1fae5,stroke:#059669,color:#064e3b
    style T2 fill:#d1fae5,stroke:#059669,color:#064e3b
    style T3 fill:#d1fae5,stroke:#059669,color:#064e3b
    style T4 fill:#d1fae5,stroke:#059669,color:#064e3b
```

### How Tools fit into an Agent

An agent is simply **LLM + Tools**: the LLM handles reasoning and decision making, the tools handle action.

```mermaid
flowchart LR
    subgraph AGENT["AI Agent"]
        direction LR
        R["Reasoning &<br/>Decision Making<br/><b>(LLM)</b>"] --> A["Action<br/><b>(Tools)</b>"]
    end
    style AGENT fill:#ecfdf5,stroke:#059669,stroke-width:2px
    style R fill:#fef3c7,stroke:#d97706,color:#78350f
    style A fill:#fef3c7,stroke:#d97706,color:#78350f
```

### Types of Tools

```mermaid
flowchart TD
    T["Tools"] --> B["Built-in Tools<br/><i>pre-built, just import & use</i>"]
    T --> C["Custom Tools<br/><i>you write the logic</i>"]
    T --> K["Toolkits<br/><i>bundles of related tools</i>"]
    C --> C1["@tool decorator"]
    C --> C2["StructuredTool<br/><i>Pydantic schema</i>"]
    C --> C3["BaseTool<br/><i>abstract base class</i>"]
    C3 -. "parent of" .-> C1
    C3 -. "parent of" .-> C2

    style T fill:#e0e7ff,stroke:#4f46e5,stroke-width:2px,color:#312e81
    style B fill:#d1fae5,stroke:#059669,color:#064e3b
    style C fill:#d1fae5,stroke:#059669,color:#064e3b
    style K fill:#d1fae5,stroke:#059669,color:#064e3b
```

| Built-in Tools | Custom Tools |
|---|---|
| Pre-built & production-ready | You define them yourself |
| Minimal / no setup | Full control over logic |
| Just `import` and use | `@tool`, `StructuredTool`, `BaseTool` |

> Notebooks: [`01_LangChain_Tools/`](01_LangChain_Tools/)

---

## Part 2 — Tool Calling

Once you have tools, the LLM needs to know **which ones exist** and **how to use them**. That happens in three steps:

```mermaid
flowchart LR
    A["1 · Tool Binding<br/><i>register tools with the LLM</i>"] --> B["2 · Tool Calling<br/><i>LLM suggests tool + args</i>"]
    B --> C["3 · Tool Execution<br/><i>you / LangChain run it</i>"]

    style A fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    style B fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    style C fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
```

### Step 1 — Tool Binding

Binding tells the LLM three things about every tool:

```mermaid
flowchart LR
    TOOL["Tool"] --> N["name<br/><i>what tools are available</i>"]
    TOOL --> D["description<br/><i>what each tool does</i>"]
    TOOL --> S["input schema<br/><i>what input format to use</i>"]
    N & D & S --> LLM["llm.bind_tools([...])"]

    style TOOL fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
    style LLM fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f
```

### Step 2 & 3 — Tool Calling and Tool Execution

> The LLM **does not run the tool**. It only *suggests* the tool and its arguments. The actual execution is done by LangChain or by you.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant L as LLM (with bound tools)
    participant T as multiply()

    U->>L: "What's 8 multiplied by 7?"
    Note over L: Tool Calling
    L-->>U: { "tool": "multiply", "args": { "a": 8, "b": 7 } }
    Note over U,T: Tool Execution
    U->>T: multiply(a=8, b=7)
    T-->>U: 56
    U->>L: ToolMessage(56)
    L-->>U: "8 × 7 = 56"
```

### Mini Project — Currency Converter

A multi-step tool-calling demo. The second tool needs the **output of the first tool**, so the conversion rate is injected at runtime (the LLM is told *not* to fill that argument itself).

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant L as LLM
    participant F as get_conversion_factor()
    participant C as convert()

    U->>L: "Convert 10 USD to INR"
    Note over L: Thinks: I don't know the rate yet
    L->>F: base = USD, target = INR
    F-->>L: 85.3415
    Note over L: Thinks: now I can convert
    L->>C: amount = 10, rate = 85.3415 (injected)
    C-->>L: 853.415
    L-->>U: "10 USD is 853.415 INR at the current rate"
```

> Notebooks: [`02_LangChain_Tool_Calling/`](02_LangChain_Tool_Calling/)

---

## Part 3 — AI Agents

An AI agent wraps everything above in an **autonomous loop**. You give it a goal — it works out the steps.

```mermaid
mindmap
  root((AI Agent))
    Goal-driven
      Tell it what, not how
    Autonomous planning
      Breaks down the problem
    Tool-using
      APIs, search, calculators
    Context-aware
      Memory across steps
    Reasoning-capable
      Decides what to do next
    Adaptive
      Re-plans when things fail
```

**Example:** *"Plan a budget trip from Delhi to Goa, 1st–7th May."* An agent would understand the intent, research trains and flights, find affordable stays, plan local travel and activities, sum up the budget, and confirm bookings — calling the right tool at each step and checking back with the user.

---

## The ReAct Pattern

**ReAct = Reasoning + Acting.** Instead of answering in one shot, the model **thinks step by step**, decides what to do next, and optionally calls tools to help. It was introduced in *"ReAct: Synergizing Reasoning and Acting in Language Models"* (Yao et al., 2022).

```mermaid
flowchart TD
    Q["User Query"] --> TH["Thought<br/><i>what do I need next?</i>"]
    TH --> AC["Action<br/><i>pick a tool + input</i>"]
    AC --> OB["Observation<br/><i>tool result</i>"]
    OB --> D{"Enough info?"}
    D -- No --> TH
    D -- Yes --> FA["Final Answer"]

    style Q fill:#e0e7ff,stroke:#4f46e5,color:#312e81
    style TH fill:#fef3c7,stroke:#d97706,color:#78350f
    style AC fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    style OB fill:#d1fae5,stroke:#059669,color:#064e3b
    style FA fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#831843
```

**Example trace:**

```text
Thought:      I need to find the capital of France.
Action:       search_tool
Action Input: "capital of France"
Observation:  Paris
Thought:      Now I need the population of Paris.
Action:       search_tool
Action Input: "population of Paris"
Observation:  2.1 million
Thought:      I now know the final answer.
Final Answer: Paris is the capital of France and has a population of ~2.1 million.
```

ReAct is useful for **multi-step problems**, **tool-augmented tasks** (web search, database lookups), and making the agent's reasoning **transparent and auditable**.

---

## Agent Executor Flow

The **Agent** only *decides* the next step. The **AgentExecutor** *runs* the whole loop:

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant E as AgentExecutor
    participant A as Agent (LLM + ReAct prompt)
    participant T as search_tool

    U->>E: user query
    loop until Final Answer
        E->>A: query + scratchpad (thought trail)
        A-->>E: AgentAction (tool, tool_input, log)
        E->>T: run tool with tool_input
        T-->>E: observation
        Note over E: append observation to scratchpad
    end
    A-->>E: AgentFinish (return_values)
    E-->>U: final output
```

### Full Control Flow

```mermaid
flowchart TD
    S["AgentExecutor"] --> R["Receive User Query"]
    R --> P["Pass Query + Agent Scratchpad to Agent"]
    P --> D{"Agent Response<br/><i>(Thought)</i>"}

    D -- AgentAction --> AA["Agent Action<br/><i>tool, tool_input, log</i>"]
    AA --> EX["Execute Tool"]
    EX --> CO["Collect Observation"]
    CO --> UP["Update Scratchpad"]
    UP --> P

    D -- AgentFinish --> AF["Agent Finish<br/><i>return_values, log</i>"]
    AF --> OUT["Return Final Output"]

    style S fill:#e0e7ff,stroke:#4f46e5,stroke-width:2px,color:#312e81
    style D fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f
    style AA fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    style EX fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    style CO fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    style UP fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    style AF fill:#d1fae5,stroke:#059669,color:#064e3b
    style OUT fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#831843
```

**In code**, this is just two pieces:

```python
from langchain_classic.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm=llm, tools=[search_tool], prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

agent_executor.invoke({"input": "What is the capital of France and its population?"})
```

> Notebook: [`03_LangChain_AI_Agent/ai_agent_in_langchain.ipynb`](03_LangChain_AI_Agent/ai_agent_in_langchain.ipynb)

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/LangChain_Agents.git
cd LangChain_Agents
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API key
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 5. Launch Jupyter
```bash
jupyter notebook
```

Open any notebook and run the cells top to bottom.

---

## Tech Stack

| Layer | Tooling |
|---|---|
| **LLM** | Google Gemini (via `langchain-google-genai`) |
| **Framework** | LangChain + LangChain Community + LangChain Experimental |
| **Validation** | Pydantic |
| **Config** | `python-dotenv` |
| **Notebooks** | Jupyter |

---

<div align="center">

### If you found this helpful

Give the repo a star — it helps others find these notes.

**Built while learning. Shared for learning.**

</div>
