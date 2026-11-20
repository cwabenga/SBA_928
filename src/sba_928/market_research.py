import os
import json
from pathlib import Path
from dotenv import load_dotenv

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

LABELS = ["Consumer Behavior", "Market Trends", "Competitor Analysis"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}


def create_curated_dataset() -> list[dict]:
	print("=================================================================")
	print("REQUIREMENT 1: CURATED MARKET RESEARCH PROMPT DATASET")
	print("=================================================================")

	dataset = [
		{"prompt": "Analyze why Gen Z shoppers abandon mobile carts during checkout.", "category": "Consumer Behavior"},
		{"prompt": "Identify key drivers behind repeat purchases in sustainable apparel.", "category": "Consumer Behavior"},
		{"prompt": "Evaluate how free shipping thresholds influence average order value.", "category": "Consumer Behavior"},
		{"prompt": "Examine customer price sensitivity for subscription meal kits.", "category": "Consumer Behavior"},
		{"prompt": "Assess how social proof and verified reviews impact first-time buyers.", "category": "Consumer Behavior"},
		{"prompt": "Investigate purchasing motivations among suburban families buying bulk groceries.", "category": "Consumer Behavior"},
		{"prompt": "Forecast demand growth for eco-friendly home goods over the next 12 months.", "category": "Market Trends"},
		{"prompt": "Identify emerging seasonal fashion trends in North America and Europe.", "category": "Market Trends"},
		{"prompt": "Analyze the shift from brick-and-mortar retail to social commerce.", "category": "Market Trends"},
		{"prompt": "Evaluate inflation impacts on discretionary spending in consumer electronics.", "category": "Market Trends"},
		{"prompt": "Track rising adoption of buy-now-pay-later services among millennials.", "category": "Market Trends"},
		{"prompt": "Summarize post-pandemic growth trends in at-home fitness equipment.", "category": "Market Trends"},
		{"prompt": "Compare pricing tiers and loyalty perks across top three athletic wear brands.", "category": "Competitor Analysis"},
		{"prompt": "Benchmark our delivery speed and return policy against leading online retailers.", "category": "Competitor Analysis"},
		{"prompt": "Identify product gaps in competitors' budget-friendly skincare lines.", "category": "Competitor Analysis"},
		{"prompt": "Analyze customer complaints directed at rival meal-delivery apps.", "category": "Competitor Analysis"},
		{"prompt": "Evaluate competitor marketing positioning in the mid-tier smart home market.", "category": "Competitor Analysis"},
		{"prompt": "Assess market share strengths of top regional coffee chains.", "category": "Competitor Analysis"},
	]

	Path("market_research_prompts.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
	print(f"Saved {len(dataset)} curated prompts across {len(LABELS)} categories to 'market_research_prompts.json'.\n")
	for sample in dataset[:3]:
		print(f"  - [{sample['category']}] {sample['prompt']}")
	return dataset


def run_prompt_variations() -> None:
	print("\n=================================================================")
	print("REQUIREMENT 2: PROMPT VARIATIONS FOR MARKET RESEARCH SCENARIOS")
	print("=================================================================")

	if not os.getenv("GROQ_API_KEY"):
		print("Skipping Groq API calls: GROQ_API_KEY not found in .env.")
		return

	llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

	variations = {
		"Variation 1 (Zero-Shot Direct)": (
			"Analyze consumer behavior trends for sustainable apparel in 2026. Keep under 90 words."
		),
		"Variation 2 (Role + Structured Competitor Matrix)": (
			"Act as a Lead Market Research Analyst. Compare our sustainable apparel line against "
			"two mid-tier competitors across Pricing, Target Demographic, and Key Differentiator "
			"in a concise 3-row Markdown table."
		),
		"Variation 3 (Scenario + Regional & Fairness Constraints)": (
			"Act as a retail strategist advising a brand expanding into both urban and rural U.S. markets. "
			"Identify 3 regional demand trends for affordable eco-friendly clothing while avoiding "
			"high-income urban bias. Keep under 100 words."
		),
	}

	chain = ChatPromptTemplate.from_template("{q}") | llm | StrOutputParser()
	for name, prompt_text in variations.items():
		print(f"\n--- {name} ---")
		print(f"Prompt  : {prompt_text}")
		res = chain.invoke({"q": prompt_text}).strip()
		print(f"Response:\n{res}")


class PromptDataset(Dataset):
	def __init__(self, texts: list[str], labels: list[int], tokenizer):
		self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
		self.labels = torch.tensor(labels, dtype=torch.long)

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, idx: int) -> dict:
		item = {key: val[idx] for key, val in self.encodings.items()}
		item["labels"] = self.labels[idx]
		return item


def run_finetuning_and_comparison(dataset: list[dict]) -> None:
	print("\n=================================================================")
	print("REQUIREMENTS 3 & 4: FINE-TUNING EXPERIMENT & BASE MODEL COMPARISON")
	print("=================================================================")

	model_name = "distilbert-base-uncased"
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	torch.manual_seed(42)
	model = AutoModelForSequenceClassification.from_pretrained(
		model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
	)

	test_prompts = [
		"Why do budget-conscious shoppers switch to store-brand groceries during inflation?",
		"Project EV charging station adoption growth across suburban shopping centers by 2027.",
		"Compare subscription renewal rates and app ratings of our top two streaming rivals.",
	]
	expected_labels = ["Consumer Behavior", "Market Trends", "Competitor Analysis"]

	def predict(target_model, prompts: list[str]) -> list[tuple[str, float]]:
		target_model.eval()
		inputs = tokenizer(prompts, truncation=True, padding=True, max_length=64, return_tensors="pt")
		with torch.no_grad():
			logits = target_model(**inputs).logits
			probs = torch.softmax(logits, dim=-1)
			scores, preds = torch.max(probs, dim=-1)
		return [(ID2LABEL[p.item()], s.item()) for p, s in zip(preds, scores)]

	before_preds = predict(model, test_prompts)

	texts = [item["prompt"] for item in dataset]
	labels = [LABEL2ID[item["category"]] for item in dataset]
	train_loader = DataLoader(PromptDataset(texts, labels, tokenizer), batch_size=6, shuffle=True)

	optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
	model.train()
	epochs = 12
	print(f"Fine-tuning '{model_name}' for {epochs} epochs on {len(dataset)} curated market research prompts...")

	for epoch in range(epochs):
		total_loss = 0.0
		for batch in train_loader:
			optimizer.zero_grad()
			outputs = model(**batch)
			loss = outputs.loss
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		if (epoch + 1) in (1, 6, 12):
			print(f"  Epoch {epoch + 1:02d}/{epochs:02d} | Training Loss: {total_loss / len(train_loader):.4f}")

	after_preds = predict(model, test_prompts)

	print("\n--- Side-by-Side Comparison: Original Base Model vs. Fine-Tuned Model ---")
	for idx, prompt in enumerate(test_prompts):
		b_label, b_conf = before_preds[idx]
		a_label, a_conf = after_preds[idx]
		print(f"\nTest Prompt {idx + 1} : {prompt}")
		print(f"Expected      : {expected_labels[idx]}")
		print(f"Original Model: {b_label} (Confidence: {b_conf:.2%})")
		print(f"Fine-Tuned    : {a_label} (Confidence: {a_conf:.2%})")


def run_bias_and_fairness_audit() -> None:
	print("\n=================================================================")
	print("REQUIREMENTS 5 & 6: BIAS ASSESSMENT & FAIRNESS MITIGATION")
	print("=================================================================")

	if not os.getenv("GROQ_API_KEY"):
		print("Skipping bias and fairness audit: GROQ_API_KEY not found in .env.")
		return

	llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)

	biased_prompt = "Why do wealthy urban professionals prefer luxury organic skincare over cheap drugstore brands?"
	fair_prompt = (
		"Analyze skincare purchasing preferences across diverse income brackets, age groups, and both "
		"urban and rural regions. Compare motivations for choosing premium organic products versus "
		"affordable mass-market brands without assuming one demographic is superior. Keep under 100 words."
	)

	chain = ChatPromptTemplate.from_template("{q}") | llm | StrOutputParser()
	print(f"Biased Prompt   : {biased_prompt}")
	print(f"Biased Output   : {chain.invoke({'q': biased_prompt}).strip()}\n")
	print(f"Mitigated Prompt: {fair_prompt}")
	print(f"Fair Output     : {chain.invoke({'q': fair_prompt}).strip()}")
	print("=================================================================")


def run_sba_928() -> None:
	dataset = create_curated_dataset()
	run_prompt_variations()
	run_finetuning_and_comparison(dataset)
	run_bias_and_fairness_audit()


if __name__ == "__main__":
	run_sba_928()
