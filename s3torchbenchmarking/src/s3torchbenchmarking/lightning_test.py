from transformers import AutoTokenizer, AutoModelForCausalLM


def run():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto")

    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt")
    print(f"{input_ids}")


if __name__ == "__main__":
    run()
