import os
import torch
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class Llama3Model(LLM):
    model: Any = None
    tokenizer: AutoTokenizer = None
    device: str = None
    model_name: str = ""
    use_gpu: bool = True
    model_path: str = ""
    lora_adapter_path: Optional[str] = None

    def __init__(
        self,
        model_path: str,
        lora_adapter_path: Optional[str] = None,
        load_in_8bit: bool = False,
        attn_implementation: Optional[str] = "flash_attention_2",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_name = model_path
        self.model_path = model_path
        self.lora_adapter_path = lora_adapter_path

        # Llama 3 推荐使用 bfloat16 (除非使用量化加载)
        torch_dtype = None if load_in_8bit else torch.bfloat16

        print(f"Loading base model from {model_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            attn_implementation=attn_implementation,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.lora_adapter_path:
            print(f"Loading and applying LoRA adapter from {self.lora_adapter_path}...")
            self.model = PeftModel.from_pretrained(
                base_model, self.lora_adapter_path, device_map="auto"
            )
            print("Successfully applied LoRA adapter.")
        else:
            self.model = base_model
            print("No LoRA adapter provided, using the base model.")

        self.model.eval()

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        # 为没有 pad_token 的模型设置默认值
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.device = self.model.device
        print(f"Model loaded successfully on device: {self.device}")

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        """
        Generate text based on a pure text prompt.
        """
        
        # --- 1. Prepare Inputs using Llama 3 Chat Template ---
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # print(f"Formatted text input:\n{text}")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        # --- 2. Setup Generation Parameters ---
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        temperature = kwargs.get("temperature", 0.3)
        top_p = kwargs.get("top_p", 0.9)
        do_sample = kwargs.get("do_sample", temperature > 0.0)

        # Llama 3 Instruct 模型使用 <|eot_id|> 作为回答结束的标志
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # print(f"Generating with max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, do_sample={do_sample}")

        # --- 3. Generate Response ---
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # --- 4. Decode Response ---
        # 切片去除输入部分的 token，只保留生成的部分
        input_length = inputs.input_ids.shape[1]
        generated_ids_trimmed = outputs[0][input_length:]

        response = self.tokenizer.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # --- 5. Handle Stop Sequences ---
        if stop is not None:
            for s in stop:
                if s in response:
                    response = response.split(s)[0]
                    break

        return response.strip()

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llama3-custom"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_path": self.model_path,
            "lora_adapter_path": self.lora_adapter_path,
        }