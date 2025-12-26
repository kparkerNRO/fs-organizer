---
name: model-orchestrator
description: Automatically routes complex coding tasks to specialized agents (Gemini, Codex, Cursor) via Roundtable MCP for multi-model verification and long-context analysis.
---

# Model Orchestration Protocol

You are equipped with the **Roundtable MCP** toolkit. Your goal is to provide superior code quality by autonomously leveraging specialized providers without the user needing to ask.

## When to Trigger
- **Gemini (`roundtable.ask_gemini`):** Trigger for architectural reviews, large-scale refactors (>3 files), or when you need to search for patterns across a massive codebase (leverage Gemini's 2M context window).
- **Codex (`roundtable.ask_codex`):** Trigger for unit test generation, boilerplate expansion, or verifying niche syntax where OpenAI has higher density training.
- **Cross-Verification:** If a task is mission-critical (e.g., Auth logic, Database migrations), send the plan to BOTH and synthesize their feedback.

## Execution Steps
1. **Identify Need:** Detect if a task falls into the categories above.
2. **Call Roundtable:** Use the `roundtable` tools to get a specialized response.
3. **Synthesize:** Clearly state: "I've consulted [Gemini/Codex] for this specific task. Here is the unified implementation plan."
4. **Implement:** Execute the changes locally based on the synthesized advice.