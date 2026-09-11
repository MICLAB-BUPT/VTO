import os
import re  # Import regex for image/video path parsing
import torch  # Import torch
from PIL import Image  # Import Image for loading images
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from peft import PeftModel
from transformers import AutoTokenizer
from transformers import (
    modeling_utils,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info  # Make sure this utility is available

# Handle modeling_utils for compatibility
if (
    not hasattr(modeling_utils, "ALL_PARALLEL_STYLES")
    or modeling_utils.ALL_PARALLEL_STYLES is None
):
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

os.makedirs("image", exist_ok=True)


class Qwen25VLModel(LLM):
    model: Any = None
    processor: AutoProcessor = None
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
        attn_implementation: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_name = model_path
        self.model_path = model_path
        self.lora_adapter_path = lora_adapter_path

        torch_dtype = None if load_in_8bit else "auto"

        # torch_dtype=torch.bfloat16
        attn_implementation = "flash_attention_2"
        print(f"Loading base model from {model_path}...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

        print("Loading processor and tokenizer...")
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.device = self.model.device
        print(f"Model loaded successfully on device: {self.device}")

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        """
        Generate text based on a prompt that can include a local image or video file path.
        """
        # --- 1. Parse Image or Video Path from Prompt ---
        image_path = None
        video_path = None

        IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
        VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

        all_extensions = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
        # This regex pattern remains the same
        path_pattern = (
            r"([a-zA-Z0-9\./\-_]+\.(?:"
            + "|".join(ext.strip(".") for ext in all_extensions)
            + "))"
        )

        # Use re.findall() to get a LIST of all matching strings
        matches = re.findall(path_pattern, prompt, re.IGNORECASE)

        # ==================== CORRECTED LOGIC START ====================
        # Check if the list of matches is not empty
        if matches:
            # Get the LAST path from the list. This string IS our path.
            # NO .group() call is needed because `matches` is a list of strings.
            potential_path = matches[-1]
            print(
                f"Found potential paths: {matches}. Using the last one: {potential_path}"
            )

            if os.path.exists(potential_path):
                _, ext = os.path.splitext(potential_path)
                if ext.lower() in IMAGE_EXTENSIONS:
                    image_path = potential_path
                    print(f"Confirmed image: {image_path}")
                elif ext.lower() in VIDEO_EXTENSIONS:
                    video_path = potential_path
                    print(f"Confirmed video: {video_path}")
            else:
                print(
                    f"Warning: Last found media file does not exist at {potential_path}"
                )
        else:
            print("No media file path found in the prompt.")
        # ===================== CORRECTED LOGIC END =====================

        # --- 2. Construct Messages Structure ---
        messages = []

        user_content = []
        video_path = None
        if image_path:
            user_content.append({"type": "image", "image": image_path})
        elif video_path:
            user_content.append({"type": "video", "video": video_path})

        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        # --- 3. Prepare Inputs using Qwen's recommended approach ---
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"Formatted text input:\n{text}")

        image_inputs, video_inputs = process_vision_info(messages)
        print(f"Extracted image inputs: {image_inputs}")
        print(f"Extracted video inputs: {video_inputs}")

        inputs = self.processor(
            text=[text],
            # images=image_inputs,
            # videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        # --- 4. Generate Response ---
        max_new_tokens = kwargs.get(
            "max_new_tokens", 1024
        )  # Increased for potentially longer thoughts
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        do_sample = kwargs.get("do_sample", temperature > 0.0)

        print(
            f"Generating with max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}, do_sample={do_sample}"
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        # --- 5. Decode Response ---
        input_length = inputs.input_ids.shape[1]
        generated_ids_trimmed = outputs[:, input_length:]

        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        if stop is not None:
            for s in stop:
                if s in response:
                    response = response.split(s)[0]
                    break

        return response.strip()

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return self.model_name

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_path": self.model_path,
            "lora_adapter_path": self.lora_adapter_path,
        }
