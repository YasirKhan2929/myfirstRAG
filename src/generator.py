
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Initial threshold for this small demonstration.
# We'll evaluate and tune it later.
MIN_SIMILARITY = 0.30

UNKNOWN_ANSWER = "I don't know based on the available information."


class Generator:
    """Generate answers grounded in retrieved document context."""

    def __init__(self):
        print("Loading the Qwen language model...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )

        self.model.eval()
        print("Qwen model loaded successfully.")

    def answer(self, question, retrieved_chunks):
        """Answer only when relevant context is available."""

        if not retrieved_chunks:
            return UNKNOWN_ANSWER

        # Check whether the best retrieved chunk is relevant enough.
        best_score = max(
            result["score"] for result in retrieved_chunks
        )

        if best_score < MIN_SIMILARITY:
            return UNKNOWN_ANSWER

        context = "\n\n".join(
            result["chunk"] for result in retrieved_chunks
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful store assistant. "
                    "Answer questions using only the supplied context. "
                    "The context may not contain the answer. "
                    "If the answer is not explicitly supported by the "
                    "context, reply exactly: "
                    "'I don't know based on the available information.' "
                    "Never guess, infer unsupported facts, or answer "
                    "yes or no without evidence. "
                    "Keep supported answers brief."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer briefly using only the context."
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        return answer or UNKNOWN_ANSWER
