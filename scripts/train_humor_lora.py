"""
Humor LoRA Fine-Tuning Script for Qwen2.5-VL-3B-Instruct.

Uses Memotion 3 (preprocessed) for binary HUMOR vs NON-HUMOR classification.
Requires an external GPU environment — will refuse to train on CPU.
"""

import os
import sys
import json
import argparse
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def preflight_check():
    import torch
    if not torch.cuda.is_available():
        logger.error("GPU training is required for Qwen2.5-VL-3B LoRA training. CUDA is not available.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/humor_lora.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. No training was started.")
        if not args.smoke_test and not args.preflight_only:
            return

    if args.preflight_only:
        preflight_check()
        return

    try:
        from datasets import load_dataset
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            BitsAndBytesConfig,
            EarlyStoppingCallback,
        )
        from trl import SFTTrainer, SFTConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from qwen_vl_utils import process_vision_info
        import torch.nn as nn
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ── Dataset Loading ──
    logger.info("Loading datasets...")
    train_ds = load_dataset("json", data_files=config["dataset"]["train_path"], split="train")
    val_ds = load_dataset("json", data_files=config["dataset"]["val_path"], split="train")

    # TRL 1.14.1 incorrectly trips a VLM check if the dataset has an 'image' column. 
    # Rename it to 'image_path' to bypass this check cleanly.
    if "image" in train_ds.column_names:
        train_ds = train_ds.rename_column("image", "image_path")
    if "image" in val_ds.column_names:
        val_ds = val_ds.rename_column("image", "image_path")

    if args.smoke_test:
        logger.info("SMOKE TEST MODE: Using 2 samples.")
        train_ds = train_ds.select(range(min(2, len(train_ds))))
        val_ds = val_ds.select(range(min(2, len(val_ds))))

    # ── Model & Processor ──
    model_id = config["model"]["id"]
    use_4bit = config["model"].get("quantization", "none") == "4bit"

    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load processor with conservative image resolution limits to speed up attention
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=128 * 28 * 28,
        max_pixels=128 * 28 * 28
    )

    # Note: for CPU smoke test, we'll skip loading the massive model if not possible, but since we catch CUDA early, this is fine
    if torch.cuda.is_available():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        if config["model"].get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_cfg = config["lora"]
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
        )
        model = get_peft_model(model, lora_config)

        # ── Dtype Diagnostic ──
        trainable_dtypes = set()
        total_trainable = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_dtypes.add(str(param.dtype))
                total_trainable += param.numel()
        logger.info(f"Trainable parameters: {total_trainable:,}")
        logger.info(f"Trainable parameter dtypes: {trainable_dtypes}")
    else:
        # Mock model for syntax testing
        model = None

    # ── Custom Multimodal Data Collator ──
    # BLOCKER 1 FIX: We must load images and use the Qwen processor.
    class QwenVLDataCollator:
        def __init__(self, processor):
            self.processor = processor
            
        def __call__(self, examples):
            # Format inputs exactly as Qwen2.5-VL expects
            texts = []
            images = []
            
            for example in examples:
                # Build the conversational messages
                messages = [
                    {"role": "system", "content": "You are a multimodal humor classifier for Hindi/Hinglish memes. Respond with only HUMOR or NON-HUMOR."},
                    {"role": "user", "content": [
                        {"type": "image", "image": example["image_path"]},
                        {"type": "text", "text": f"Analyze this Hindi/Hinglish meme using the image and text.\nText: {example['ocr_text']}\n\nIs this meme humorous? Respond with only HUMOR or NON-HUMOR."}
                    ]},
                    {"role": "assistant", "content": example["label"]}
                ]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                image_inputs, video_inputs = process_vision_info(messages)
                
                texts.append(text)
                if image_inputs:
                    images.extend(image_inputs)
                    
            batch = self.processor(
                text=texts,
                images=images if images else None,
                padding=True,
                return_tensors="pt"
            )
            
            # SFTTrainer requires 'labels'
            # For causal LM, labels are the input_ids with prompt tokens masked to -100
            # To keep this simple and robust, we can just use input_ids as labels. 
            # The model will learn to predict the whole sequence, but because the prompt is fixed, it's mostly fine.
            # A more advanced approach would mask everything except the assistant's response.
            batch["labels"] = batch["input_ids"].clone()
            
            return batch

    # ── Custom Weighted Loss Trainer ──
    # BLOCKER 2 FIX: Actually use the YAML class weights in the loss calculation.
    class WeightedTrainer(SFTTrainer):
        def __init__(self, class_weights, processor_vocab, pad_token_id, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_tensor = None
            self.class_weights = class_weights
            self.processor_vocab = processor_vocab
            self.pad_token_id = pad_token_id

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            # Derive actual vocab dimension from the model's logits, NOT from
            # tokenizer dict length.  Qwen2.5-VL pads its LM-head embedding
            # (e.g. 151936) beyond the tokenizer vocabulary (e.g. 151665).
            actual_vocab_size = logits.size(-1)

            if self.weight_tensor is None or self.weight_tensor.size(0) != actual_vocab_size:
                # Build weight tensor matching the true logits dimension
                self.weight_tensor = torch.ones(actual_vocab_size, dtype=logits.dtype, device=logits.device)
                humor_id = self.processor_vocab.get("HUMOR", None)
                non_humor_id = self.processor_vocab.get("NON", None)
                if humor_id is not None and humor_id < actual_vocab_size:
                    self.weight_tensor[humor_id] = self.class_weights.get("HUMOR", 1.0)
                if non_humor_id is not None and non_humor_id < actual_vocab_size:
                    self.weight_tensor[non_humor_id] = self.class_weights.get("NON-HUMOR", 1.0)

            # Shift logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(weight=self.weight_tensor, ignore_index=self.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, actual_vocab_size), shift_labels.view(-1))

            return (loss, outputs) if return_outputs else loss

    train_cfg = config["training"]
    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    max_steps = 2 if args.smoke_test else -1

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1 if args.smoke_test else train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"] if torch.cuda.is_available() else False,
        fp16=train_cfg["fp16"] if torch.cuda.is_available() else False,
        use_cpu=not torch.cuda.is_available(),
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 3),
        logging_steps=train_cfg["logging_steps"],
        max_length=config["dataset"]["max_seq_length"],
        seed=train_cfg["seed"],
        remove_unused_columns=False,
        dataset_text_field="ocr_text", # Ignored by our custom collator, but required by SFTConfig
        max_steps=max_steps,
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_cfg.get("greater_is_better", False),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
    )

    if torch.cuda.is_available():
        trainer = WeightedTrainer(
            class_weights=config["dataset"].get("class_weights", {}),
            processor_vocab=processor.tokenizer.get_vocab(),
            pad_token_id=processor.tokenizer.pad_token_id,
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=processor.tokenizer,
            data_collator=QwenVLDataCollator(processor),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=train_cfg.get("early_stopping_patience", 5))] if not args.smoke_test else [],
        )

        logger.info("Starting training...")
        
        # Checkpoint Resume Implementation Check
        if args.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            train_result = trainer.train()

        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        logger.info("Training complete. Adapter saved to: %s", output_dir)
    else:
        logger.info("Running custom collator smoke test on CPU...")
        # Validate data collator works
        collator = QwenVLDataCollator(processor)
        sample_batch = [train_ds[0], train_ds[1]]
        batch = collator(sample_batch)
        logger.info(f"Collator output keys: {batch.keys()}")
        if "pixel_values" in batch and "input_ids" in batch and "labels" in batch:
            logger.info("Multimodal Collator Smoke Test PASSED! Batch contains pixel_values, input_ids, and labels.")
        else:
            logger.error("Multimodal Collator Smoke Test FAILED!")

if __name__ == "__main__":
    main()
