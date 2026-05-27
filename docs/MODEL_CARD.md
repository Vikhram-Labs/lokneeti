---
language:
- en
- hi
- bn
- te
- ta
- gu
- kn
- ml
- mr
- pa
license: apache-2.0
library_name: transformers
tags:
- governance
- constitutional-ai
- policy-reasoning
- indian-constitution
- qlora
- peft
- sovereign-ai
- multilingual
base_model: Qwen/Qwen2.5-3B-Instruct
datasets:
- vikhram-labs/lokneeti-governance-dataset
pipeline_tag: text-generation
---

# Lokneeti-3B

> **Reasoning for Democratic Governance**

Lokneeti-3B is a constitutional and governance reasoning Small Language Model (SLM)
developed by **Vikhram Labs**, fine-tuned on Qwen2.5-3B-Instruct using QLoRA.

It specializes in **Indian constitutional reasoning**, **public policy analysis**, and
**democratic governance intelligence** — not general-purpose conversation.

---

## Model Description

| Property         | Value                                              |
|------------------|----------------------------------------------------|
| Base Model       | Qwen/Qwen2.5-3B-Instruct                          |
| Fine-tuning      | QLoRA (4-bit NF4, LoRA r=16, α=32)               |
| Parameters       | ~3B (base) + LoRA adapter                         |
| Languages        | English + 10 Indian languages                     |
| Context Length   | 2048 tokens                                        |
| License          | Apache 2.0                                         |
| Training Compute | Google Colab T4 GPU (15GB VRAM)                   |

---

## Core Capability: Constitutional Chain Compression (C³)

Lokneeti implements a novel reasoning paradigm called **Constitutional Chain Compression (C³)**
that encodes the pathway from a raw governance scenario to a constitutional conclusion
via ordered symbolic reasoning hops.

### Example

**Input:**
```
A welfare scheme excludes biometric-failure citizens from receiving food rations.
```

**C³ Chain:**
```
Article_21 -> welfare_access
biometric_exclusion_risk -> vulnerable_groups
Implementation_gap -> constitutional_vulnerability
```

**Output:**
```
[Constitutional Chain Compression]

CHAIN:
  Article_21 -> welfare_access_risk
  Article_21 -> constitutional_protection
  Implementation_gap -> vulnerable_groups

RISK LEVEL: HIGH

ARTICLES: Article 21, Article 14

CONCLUSION:
This policy presents HIGH constitutional risk under Article 21.
The implementation gap may cause substantive violation of the right to
life and personal liberty for affected citizens, warranting urgent review.
Secondary constitutional implications detected under: Article 14.
```

---

## Use Cases

✅ **Supported Tasks:**
- Constitutional article Q&A
- Policy contradiction detection
- Welfare scheme risk analysis
- Federal conflict reasoning
- Citizen grievance abstraction
- Implementation gap detection
- Institutional reasoning

❌ **Not Designed For:**
- General conversation
- Coding assistance
- Political campaigning
- Legal advice (use at own risk — not a substitute for legal counsel)

---

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "vikhram-labs/Lokneeti-3B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

system_prompt = (
    "You are Lokneeti, a constitutional governance reasoning system. "
    "Analyze Indian public policy using Constitutional Chain Compression methodology."
)

scenario = "A welfare scheme excludes biometric-failure citizens from receiving food rations."

messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": (
            "Analyse the following governance scenario using Constitutional Chain "
            "Compression. Identify constitutional risks, enumerate the reasoning "
            f"chain, and provide a structured governance conclusion.\n\n{scenario}"
        ),
    },
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)

print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Using with lokneeti library

```python
from lokneeti.inference.pipeline import LoknetiPipeline

pipeline = LoknetiPipeline.from_pretrained("vikhram-labs/Lokneeti-3B")
result = pipeline.analyze("A welfare scheme excludes biometric-failure citizens.")
print(result.output)
```

---

## Training Details

### Dataset
- Indian Constitution (articles 1–395 + Schedules)
- Synthetic governance instruction-tuning data (7 task categories)
- Constitutional Chain Compression (C³) examples
- Parliamentary Q&A extracts
- NITI Aayog policy document chunks
- Welfare scheme descriptions

### Fine-tuning Configuration
```yaml
base_model: Qwen/Qwen2.5-3B-Instruct
method: QLoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
quantization: 4-bit NF4
batch_size: 2 (×8 gradient accumulation = 16 effective)
learning_rate: 2e-4
scheduler: cosine
epochs: 3
max_seq_length: 2048
optimizer: adamw_8bit
```

---

## Evaluation

| Metric                        | Score  |
|-------------------------------|--------|
| Constitutional Consistency    | ~0.78  |
| Article Recall                | ~0.71  |
| ROUGE-L (vs reference)        | ~0.43  |
| Hallucination Rate            | ~0.04  |

*Evaluated on held-out test set from Lokneeti governance benchmark.*

---

## Limitations

- Not a legal advice tool
- Constitutional interpretations are approximate — verify with legal experts
- English-dominant; Indic language performance varies by script
- Knowledge cutoff limited by training data
- May reflect biases in policy text sources

---

## Citation

```bibtex
@misc{lokneeti2024,
  title        = {Lokneeti-3B: Constitutional Governance Reasoning SLM for Democratic Intelligence},
  author       = {Vikhram Labs},
  year         = {2024},
  publisher    = {HuggingFace},
  url          = {https://huggingface.co/vikhram-labs/Lokneeti-3B},
  note         = {Apache 2.0 License}
}
```

---

*Developed by Vikhram Labs · Apache 2.0 · Not a substitute for legal advice*
