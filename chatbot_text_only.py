import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import threading
import sys
import json


class SimpleChatBot:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        print("Loading LLM model Parameters from config.json...")
        self.llm_config = config["llm"]

        print("Loading LLM model:", self.llm_config["model_id"])
        print("LLM settings:", {
              k: v for k, v in self.llm_config.items() if k != "model_id"})

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_config["model_id"], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_config["model_id"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on device: {device}")

    def generate_prompt(self, user_input):
        prompt = f"User: {user_input}\nAssistant:"
        return prompt

    def generate_response(self, user_input):
        prompt = self.generate_prompt(user_input)
        inputs = self.tokenizer(
            prompt, return_tensors="pt").to(self.model.device)

        eos_token_id = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.llm_config.get("max_new_tokens", 150),
            temperature=self.llm_config.get("temperature", 0.7),
            top_p=self.llm_config.get("top_p", 0.9),
            do_sample=self.llm_config.get("do_sample", True),
            eos_token_id=eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        # Clean unwanted tags
        response = response.replace(
            "</think>", "").replace("<think>", "").strip()
        response = response.split("User:")[0].strip()

        return response

    def animate_typing(self, stop_event):
        animation = ["|", "/", "-", "\\"]
        idx = 0
        while not stop_event.is_set():
            sys.stdout.write(
                f"Bot is typing... {animation[idx % len(animation)]}\r")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.2)
        sys.stdout.write(" " * 30 + "\r")
        sys.stdout.flush()

    def chat(self):
        print("Simple ChatBot is ready. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            bot_response = None

            def generate():
                nonlocal bot_response
                bot_response = self.generate_response(user_input)

            gen_thread = threading.Thread(target=generate)
            stop_event = threading.Event()
            animation_thread = threading.Thread(
                target=self.animate_typing, args=(stop_event,))

            gen_thread.start()
            animation_thread.start()

            gen_thread.join()
            stop_event.set()
            animation_thread.join()

            print("Bot:", bot_response)


if __name__ == "__main__":
    chatbot = SimpleChatBot()
    chatbot.chat()
