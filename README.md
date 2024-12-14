**Telugu_LLm**

**1.Install the needed libraries**.

pip install transformers
pip install torch

**2.Import the needed dependencies.**

from transformers import AutoTokenizer, AutoModelForCausalLM

**3. Load the model and tokenizer.**

tokenizer = AutoTokenizer.from_pretrained("Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0")
model = AutoModelForCausalLM.from_pretrained("Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0")

**4. Define your question and encode the question using the tokenizer.**

question = "భారతదేశ రాజధాని ఏమిటి?"
Input_ids = tokenizer.encode(question, return_tensors="pt")

**5.Generate the response using the model.**

outputs = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2)

**6.Decode the generated tokens back to text and print the answer.**

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)

