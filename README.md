# Financial Researcher — A CrewAI Multi-Agent System

A multi-agent AI system built with [CrewAI](https://crewai.com) that researches any company you name and produces a polished, professional financial-style report. You type in a company (for example, `uEngage`), and a small "crew" of AI agents goes to work: one searches the web and gathers up-to-date facts, the other analyzes those findings and writes a structured report to `output/report.md`.

This README is also a teaching document. Beyond getting the project running, it explains **what CrewAI is**, **how the agent architecture is wired**, **how many agents and tasks exist and what each does**, and — importantly — **how we use *context* to pass information from one agent to the next**.

---

## Table of Contents

1. [What is CrewAI?](#1-what-is-crewai)
2. [The Mental Model: Agents, Tasks, Tools, Crews](#2-the-mental-model-agents-tasks-tools-crews)
3. [Architecture of This Project](#3-architecture-of-this-project)
4. [The Agents (2)](#4-the-agents-2)
5. [The Tasks (2)](#5-the-tasks-2)
6. [How We Use Context](#6-how-we-use-context-the-key-concept)
7. [Process: Why Sequential?](#7-process-why-sequential)
8. [Project Structure](#8-project-structure)
9. [Installation](#9-installation)
10. [Running the Project](#10-running-the-project)
11. [Customizing](#11-customizing)
12. [Sample Output](#12-sample-output)

---

## 1. What is CrewAI?

**CrewAI** is an open-source Python framework for building **multi-agent AI systems** — applications where several specialized LLM-powered "agents" collaborate to complete a task that would be hard for a single agent (or a single prompt) to do well.

The core idea is borrowed from how human teams work. Instead of one generalist doing everything, you assemble a **crew** of specialists, give each a clear **role**, **goal**, and **backstory**, hand them **tasks**, and let them pass their work down the line. CrewAI handles the orchestration: deciding which agent runs when, feeding each agent the right context, invoking tools, and collecting the outputs.

CrewAI's design philosophy is **configuration over code**. The *personalities* of your agents and the *definitions* of your tasks live in plain YAML files (`agents.yaml`, `tasks.yaml`), while the Python code (`crew.py`) just wires them together. This separation makes the system easy to read, tune, and extend — you can completely change an agent's behavior by editing a few lines of YAML, no Python required.

---

## 2. The Mental Model: Agents, Tasks, Tools, Crews

Four concepts make up the entire framework. Understanding them is understanding CrewAI:

| Concept | What it is | In this project |
|---------|------------|-----------------|
| **Agent** | An autonomous AI worker defined by a `role`, `goal`, and `backstory`, backed by an LLM. It's the *"who."* | `researcher` and `analyst` |
| **Task** | A unit of work with a `description` and an `expected_output`, assigned to one agent. It's the *"what."* | `research_task` and `analysis_task` |
| **Tool** | A capability an agent can call (web search, file read, API calls). It extends what an agent can *do* beyond pure text. | `SerperDevTool` (Google search) |
| **Crew** | The container that holds agents + tasks and defines the **process** (how they run — sequentially or hierarchically). It's the *"how."* | `FinancialResearcher` crew |

A useful analogy: **agents are employees**, **tasks are tickets on a board**, **tools are the software those employees can use**, and the **crew is the team plus the workflow rules** that say who does what and in what order.

---

## 3. Architecture of This Project

This project uses the **`@CrewBase` decorator pattern** — CrewAI's recommended structure for production crews. Here is the high-level flow:

```
                    ┌─────────────────────────────────────┐
   User input  ───► │   main.py  (run)                     │
  "uEngage" +       │   inputs = {company, current_date}   │
  current_date      └──────────────────┬──────────────────┘
                                        │ kickoff(inputs)
                                        ▼
                    ┌─────────────────────────────────────┐
                    │   crew.py  (FinancialResearcher)     │
                    │   Process.sequential                 │
                    └──────────────────┬──────────────────┘
                                        │
              ┌─────────────────────────┴───────────────────────┐
              ▼                                                   ▼
   ┌────────────────────────┐    research_task        ┌────────────────────────┐
   │  AGENT 1: researcher    │   output becomes        │  AGENT 2: analyst       │
   │  - role: Sr. Financial  │   ───── context ──────► │  - role: Market Analyst │
   │    Researcher           │                         │    & Report Writer      │
   │  - tool: SerperDevTool  │                         │  - no tools             │
   │    (live web search)    │                         │  - writes report.md     │
   └────────────────────────┘                          └────────────────────────┘
        runs research_task                                  runs analysis_task
                                                                   │
                                                                   ▼
                                                          output/report.md
```

**In words:**

1. `main.py` collects the company name from the user and stamps the current date, then calls `kickoff()` on the crew with those inputs.
2. The **researcher** agent runs `research_task` first. It uses the `SerperDevTool` to perform live Google searches and compiles a structured research document.
3. That research document is passed as **context** into the **analyst** agent's `analysis_task`.
4. The **analyst** reads the research, adds analysis and insight, and writes the final, formatted report to `output/report.md`.

### Architecture summary

- **Type:** Sequential, two-stage research → analysis pipeline
- **Number of agents:** **2** (researcher, analyst)
- **Number of tasks:** **2** (research_task, analysis_task)
- **Process:** `Process.sequential` (tasks run one after another, in order)
- **Tools:** 1 (`SerperDevTool` for web search, attached to the researcher only)
- **Context flow:** output of `research_task` → input context of `analysis_task`
- **LLM:** `openai/gpt-5.4-mini` for both agents
- **Tracing:** enabled (`tracing=True`) for observability of each step

---

## 4. The Agents (2)

Both agents are defined declaratively in [`src/financial_researcher/config/agents.yaml`](src/financial_researcher/config/agents.yaml) and instantiated in [`crew.py`](src/financial_researcher/crew.py) using the `@agent` decorator.

### Agent 1 — `researcher`

```yaml
researcher:
  role: Senior Financial Researcher for {company}
  goal: Research latest company information, news and potential for {company}.
        ...up to date as of {current_date}.
  backstory: You're a seasoned financial researcher with a talent for finding
             the most relevant information about {company}...
  llm: openai/gpt-5.4-mini
```

```python
@agent
def researcher(self) -> Agent:
    return Agent(
        config=self.agents_config['researcher'],
        verbose=True,
        tools=[SerperDevTool()]   # ← gives this agent live web search
    )
```

- **Responsibility:** Gather raw, current facts about the company.
- **Key feature:** It is the only agent with a **tool** — `SerperDevTool`, which lets it run real Google searches via the [Serper](https://serper.dev) API. This is what makes its research *up to date* rather than limited to the model's training data.

### Agent 2 — `analyst`

```yaml
analyst:
  role: Market Analyst and Report writer focused on {company}
  goal: Analyze company {company} and create a comprehensive, well-structured report...
  backstory: You're a meticulous, skilled analyst with a background in financial
             analysis and company research...
  llm: openai/gpt-5.4-mini
```

```python
@agent
def analyst(self) -> Agent:
    return Agent(
        config=self.agents_config['analyst'],
        verbose=True
    )
```

- **Responsibility:** Turn the researcher's raw findings into a polished, structured report with an executive summary, analysis, and market outlook.
- **Key feature:** It has **no tools**. It deliberately does not search the web — it works *only* from the context handed to it by the researcher. This keeps the two jobs cleanly separated: one finds facts, the other reasons about them.

> **Note on `{company}` and `{current_date}`:** Those curly-brace placeholders are **interpolation variables**. CrewAI automatically substitutes them at runtime with the `inputs` dictionary you pass to `kickoff()`. This is how the same generic crew works for *any* company you type in.

---

## 5. The Tasks (2)

Tasks are defined in [`src/financial_researcher/config/tasks.yaml`](src/financial_researcher/config/tasks.yaml) and wired up in `crew.py` with the `@task` decorator. Each task has a `description` (what to do) and an `expected_output` (what good output looks like), and is bound to exactly one agent.

### Task 1 — `research_task`

```yaml
research_task:
  description: >
    Conduct thorough and up-to-date research on company {company}. Focus on:
    1. Current company status and health
    2. Historical company performance
    3. Major challenges and opportunities
    4. Recent news and events
    5. Future outlook and potential developments
  expected_output: >
    A comprehensive research document with well-organized sections...
  agent: researcher
```

- Assigned to the **researcher**.
- Produces a structured research document. This document's text becomes the **context** for the next task.

### Task 2 — `analysis_task`

```yaml
analysis_task:
  description: >
    Analyze the research findings and create a comprehensive report on {company}.
    1. Begin with an executive summary
    2. Include all key information from the research
    3. Provide insightful analysis of trends and patterns
    4. Offer a market outlook ... (not for trading decisions)
    5. Be formatted in a professional, easy-to-read style
  expected_output: >
    A polished, professional report on {company}...
  agent: analyst
  context:
    - research_task          # ← THIS is the context link
  output_file: output/report.md
```

- Assigned to the **analyst**.
- Declares `context: [research_task]` — the line that connects the two halves of the pipeline.
- Writes its final result straight to disk via `output_file: output/report.md`.

---

## 6. How We Use Context (The Key Concept)

**Context** is how one agent's output is fed as input to another agent. It is the single most important wiring decision in this project, so it's worth a dedicated section.

### Two layers of context in this project

**Layer 1 — Input interpolation (`{...}` variables).**
Before any agent runs, CrewAI injects the values you pass to `kickoff()` into every `{placeholder}` in the YAML. In [`main.py`](src/financial_researcher/main.py):

```python
inputs = {
    'company': company,                       # e.g. "uEngage"
    'current_date': str(datetime.now().date())
}
FinancialResearcher().crew().kickoff(inputs=inputs)
```

So `{company}` and `{current_date}` are replaced everywhere — in roles, goals, backstories, and task descriptions. This is **input context**: the same variables shared across the whole crew.

**Layer 2 — Inter-task context (output → input).**
This is the part that makes it a *pipeline* rather than two unrelated jobs. In `tasks.yaml`:

```yaml
analysis_task:
  ...
  context:
    - research_task
```

That `context` key tells CrewAI: *"Before the analyst runs `analysis_task`, take the completed output of `research_task` and place it into the analyst's prompt as background information."*

Concretely:

1. The researcher finishes `research_task` and produces, say, 1,500 words of findings about the company.
2. Because `analysis_task` lists `research_task` in its `context`, CrewAI automatically appends that 1,500-word document to the analyst's prompt.
3. The analyst never has to search the web — it already *has* the facts. It just reasons over them and writes the report.

### Why this matters

- **Separation of concerns.** Searching and writing are different skills. The researcher specializes in finding; the analyst specializes in synthesizing. Context is the clean handoff between them.
- **No information loss.** The analyst sees the *full* research, not a lossy summary, because CrewAI passes the complete task output.
- **No redundant work.** Because the analyst has no tools and relies entirely on context, it can't go off and re-search — it stays focused on analysis, which keeps cost down and output consistent.
- **Composable.** If you added a third task (say, a fact-checker), you could give it `context: [research_task, analysis_task]` to see *both* prior outputs. Context is how you build longer chains.

```
research_task output  ──(context)──►  analysis_task prompt  ──►  report.md
   (the facts)                          (facts + analysis instructions)
```

---

## 7. Process: Why Sequential?

In [`crew.py`](src/financial_researcher/crew.py), the crew uses:

```python
return Crew(
    agents=self.agents,
    tasks=self.tasks,
    process=Process.sequential,   # tasks run in order, one after another
    verbose=True,
    tracing=True,
)
```

CrewAI offers two main process types:

- **`Process.sequential`** (used here) — tasks execute in the order they're listed. The output of each task is available as context to the next. This is the natural fit for a **research → analysis** pipeline, where the second step *depends entirely* on the first.
- **`Process.hierarchical`** — a "manager" LLM dynamically delegates tasks to agents and decides ordering. This is more powerful for open-ended problems but adds complexity and cost. The commented line in `crew.py` shows how you'd switch to it.

Because our two tasks have a strict dependency (you can't analyze research that doesn't exist yet), **sequential is the correct and simplest choice**.

`tracing=True` enables CrewAI's tracing/observability so you can inspect each agent's reasoning, tool calls, and outputs step by step — useful for debugging and understanding what the crew actually did.

---

## 8. Project Structure

```
financial_researcher/
├── README.md                         ← you are here
├── pyproject.toml                    ← dependencies & CLI entry points
├── knowledge/
│   └── user_preference.txt           ← example knowledge source
├── output/
│   └── report.md                     ← the generated report lands here
└── src/financial_researcher/
    ├── main.py                       ← entry point: collects input, runs crew
    ├── crew.py                       ← wires agents + tasks into a Crew
    ├── config/
    │   ├── agents.yaml               ← agent definitions (role/goal/backstory)
    │   └── tasks.yaml                ← task definitions (description/output/context)
    └── tools/
        └── custom_tool.py            ← scaffold for writing your own tools
```

**Where to look for what:**
- Want to change an agent's personality or model? → `config/agents.yaml`
- Want to change what a task does or how context flows? → `config/tasks.yaml`
- Want to add a tool or change orchestration? → `crew.py`
- Want to change inputs or how it's launched? → `main.py`

---

## 9. Installation

Ensure you have **Python >=3.10 <3.14** installed. This project uses [UV](https://docs.astral.sh/uv/) for dependency management.

Install `uv` if you don't have it:

```bash
pip install uv
```

Install dependencies (this also locks them):

```bash
crewai install
```

Then add your API keys to a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...        # for the LLM (openai/gpt-5.4-mini)
SERPER_API_KEY=...           # for SerperDevTool web search — get one at serper.dev
```

> Both keys are required: `OPENAI_API_KEY` powers the agents' reasoning, and `SERPER_API_KEY` powers the researcher's live web search.

---

## 10. Running the Project

From the project root:

```bash
crewai run
```

You'll be prompted:

```
Enter the company name for research: uEngage
```

The crew then runs end to end. With `verbose=True`, you'll watch the researcher perform searches and the analyst compose the report in real time. When it finishes, the final report is written to **`output/report.md`**.

### Other entry points

Defined in `pyproject.toml` (see `[project.scripts]`):

| Command | What it does |
|---------|--------------|
| `crewai run` / `run_crew` | Run the crew normally (prompts for a company) |
| `train` | Train the crew for N iterations to improve outputs |
| `replay` | Replay execution from a specific task id |
| `test` | Run and evaluate the crew with an eval LLM |
| `run_with_trigger` | Run with a JSON trigger payload (for automation/webhooks) |

---

## 11. Customizing

The framework's "configuration over code" philosophy means most changes are YAML edits:

- **Change the agents** → edit [`config/agents.yaml`](src/financial_researcher/config/agents.yaml). Try swapping the `llm`, sharpening a `goal`, or rewriting a `backstory`.
- **Change the tasks / context flow** → edit [`config/tasks.yaml`](src/financial_researcher/config/tasks.yaml). Add a task, change the `context` links, or adjust the `expected_output`.
- **Add tools or change orchestration** → edit [`crew.py`](src/financial_researcher/crew.py). For example, give the analyst a tool, or switch to `Process.hierarchical`.
- **Change inputs** → edit [`main.py`](src/financial_researcher/main.py).

---

## 12. Sample Output

Running the crew on **uEngage** produced a full multi-section report in [`output/report.md`](output/report.md), including:

- An executive summary
- Company overview and product modules
- Operating health and historical performance
- Competitive landscape
- Opportunities, risks, and a market outlook (with an explicit "not for trading decisions" disclaimer)
- A "key facts at a glance" appendix

This demonstrates the full pipeline in action: the **researcher** gathered live, dated facts via web search, and the **analyst** — using only the research handed to it via **context** — produced the polished report.

---

## Learn More About CrewAI

- [CrewAI Documentation](https://docs.crewai.com)
- [Crews & Processes](https://docs.crewai.com/concepts/crews)
- [Tasks & Context](https://docs.crewai.com/concepts/tasks)
- [Knowledge Sources](https://docs.crewai.com/concepts/knowledge)
- [GitHub](https://github.com/crewAIInc/crewAI)

---

*Built as a learning project to explore CrewAI's multi-agent orchestration, the agent/task/tool/crew model, and — above all — how **context** turns a set of independent agents into a coordinated pipeline.*
