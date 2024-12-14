def answer_query(query):
    input_ids = tokenizer.encode(query, return_tensors="pt")
    outputs = model.generate(
        input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer