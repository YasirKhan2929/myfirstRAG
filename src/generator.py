
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MIN_SIMILARITY = 0.25

UNKNOWN_ANSWER = "I don't know based on the available information."


class Generator:
    """Generate answers grounded in retrieved document context."""

    def __init__(self):
        print("Loading the Qwen language model...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Load directly onto a real device instead of using device_map="auto".
        model_dtype = (
            torch.float16 if self.device.type == "cuda"
            else torch.float32
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=model_dtype,
        )

        self.model.to(self.device)
        self.model.eval()

        print("Qwen model loaded successfully.")

    def answer(self, question, retrieved_chunks):
        """Answer only when relevant context is available."""

        if not retrieved_chunks:
            return UNKNOWN_ANSWER

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
                    "Answer using only the supplied context. "
                    "If the answer is not supported by the context, "
                    "reply exactly: "
                    "I don't know based on the available information. "
                    "Never guess or invent facts. Keep answers brief."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer using only the context."
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
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        return answer or UNKNOWN_ANSWER
