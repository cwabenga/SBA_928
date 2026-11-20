```markdown
# SBA 928: Enhancing Market Research with AI Prompt Engineering

An official Per Scholas capstone assessment demonstrating curated market research prompt engineering, PyTorch/Hugging Face model fine-tuning, and bias mitigation inside a `uv`-managed Python package.

## Key Components Implemented

- **Curated Market Research Prompt Dataset:** 18 balanced prompts saved to `market_research_prompts.json` across `Consumer Behavior`, `Market Trends`, and `Competitor Analysis`.
- **Prompt Variations Across Scenarios (LangChain + Groq):** Comparative evaluation of Zero-Shot, Role + Structured Competitor Matrix, and Regional/Fairness-Constrained prompts.
- **Language Model Fine-Tuning & Comparison (PyTorch + Hugging Face):** Fine-tuning `distilbert-base-uncased` over 12 epochs (`1.1247` to `0.0481` training loss) and comparing unseen prompt classification confidence against the untrained baseline.
- **Bias Assessment & Fairness Mitigation:** Auditing socioeconomic and urban framing bias against demographic- and region-inclusive prompt alternatives.

## Setup & Execution Instructions

1. **Sync project dependencies:**
	```bash
	uv sync
	```

2. **Configure environment variables:**
	Create a `.env` file in the project root with your Groq API key:
	```env
	GROQ_API_KEY=your_groq_api_key_here
	```

3. **Run the connected package entry point:**
	```bash
	uv run sba-928
	```
```
