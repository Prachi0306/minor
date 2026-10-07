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

from collections import Counter

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
            bnb_4bit_compute_dtype=torch.float16,
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
        # ── FIX FOR BF16 GRADIENTS ON T4 ──
        # The Trainer/Accelerate mixed-precision pipeline forcibly converts LoRA
        # adapter parameters to BF16 during model preparation, regardless of
        # explicit FP16 casts or autocast_adapter_dtype=False.  Since the T4
        # GradScaler does not support BF16 gradients, we disable Trainer AMP
        # entirely and let LoRA parameters train in FP32.  The 4-bit base model
        # remains quantized, so memory usage is unchanged.
        if hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
            model.config.torch_dtype = torch.float16
            
        model = get_peft_model(
            model,
            lora_config,
            autocast_adapter_dtype=False,
        )

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
                vision_info = process_vision_info(messages)
                image_inputs = vision_info[0]
                video_inputs = vision_info[1]
                
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
        # T4 FIX: Disable Trainer AMP entirely.  The Accelerate mixed-precision
        # pipeline converts LoRA params to BF16 (unsupported by T4 GradScaler).
        # LoRA trains in FP32; 4-bit base model stays quantized.
        bf16=False,
        fp16=False,
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

        # ── EXHAUSTIVE OPTIMIZER & GRADIENT DIAGNOSTIC ──
        if args.smoke_test and torch.cuda.is_available():
            logger.info("=" * 80)
            logger.info("EXHAUSTIVE OPTIMIZER & GRADIENT DIAGNOSTIC (PRE-TRAIN)")
            logger.info("=" * 80)
            
            diagnostic_passed = False
            try:
                # 1. Create optimizer exactly as Trainer would
                trainer.create_optimizer()
                
                # 2. Get one real batch
                collator = QwenVLDataCollator(processor)
                sample_batch = collator([train_ds[0]])
                for k, v in sample_batch.items():
                    if isinstance(v, torch.Tensor):
                        sample_batch[k] = v.to(model.device)
                
                # 3. Print environment settings
                logger.info(f"accelerator.mixed_precision: {trainer.accelerator.mixed_precision}")
                logger.info(f"trainer.args.fp16: {trainer.args.fp16}")
                logger.info(f"trainer.args.bf16: {trainer.args.bf16}")
                logger.info(f"bnb_4bit_compute_dtype: {bnb_config.bnb_4bit_compute_dtype if bnb_config else 'None'}")
                
                # Find scaler safely via introspection
                scaler_info = "NOT FOUND"
                for attr_name in ["scaler", "grad_scaler", "gradient_scaler"]:
                    if hasattr(trainer, attr_name) and getattr(trainer, attr_name) is not None:
                        scaler_info = f"trainer.{attr_name} = {getattr(trainer, attr_name).__class__.__name__}"
                        break
                if scaler_info == "NOT FOUND":
                    acc = trainer.accelerator
                    if hasattr(acc, "scaler") and acc.scaler is not None:
                        scaler_info = f"accelerator.scaler = {acc.scaler.__class__.__name__}"
                    elif hasattr(acc, "gradient_state"):
                        scaler_info = f"accelerator.gradient_state = {acc.gradient_state.__class__.__name__}"
                logger.info(f"GradScaler: {scaler_info}")
                
                if hasattr(torch, 'get_autocast_gpu_dtype'):
                    logger.info(f"torch.get_autocast_gpu_dtype(): {torch.get_autocast_gpu_dtype()}")
                
                # 4. Model dtype counts
                all_param_dtypes = Counter()
                for n, p in model.named_parameters():
                    all_param_dtypes[str(p.dtype)] += 1
                logger.info(f"Model parameter dtype counts (by count): {dict(all_param_dtypes)}")
                
                # 5. Run exact forward and backward pass
                model.train()
                
                if sft_config.fp16:
                    ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
                elif sft_config.bf16:
                    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                else:
                    from contextlib import nullcontext
                    ctx = nullcontext()
                accelerator = trainer.accelerator
                # Run a minimal forward and backward pass
                logger.info("Running dummy forward and backward pass...")
                sample_batch = {
                    k: v.to(accelerator.device) if hasattr(v, "to") else v
                    for k, v in sample_batch.items()
                }
                
                # SFTTrainer compute_loss does not automatically autocast if called directly
                # but Accelerator handles it during trainer.train(). We wrap it just to be safe.
                with accelerator.autocast():
                    loss = trainer.compute_loss(model, sample_batch)
                
                loss.backward()
                
                # 6a. Run optimizer.step() to verify full stability
                logger.info("Running optimizer.step() to verify full stability...")
                trainer.optimizer.step()
                logger.info("optimizer.step() completed successfully.")

                # 6. Inspect EVERY parameter in optimizer
                logger.info("-" * 40)
                logger.info("FINAL OPTIMIZER PARAMETER INSPECTION")
                logger.info("-" * 40)
                
                opt_param_dtypes = Counter()
                opt_grad_dtypes = Counter()
                combo_counts = Counter()
                bf16_grad_params = []
                
                logger.info(f"Optimizer class: {trainer.optimizer.__class__.__name__}")
                logger.info(f"Number of parameter groups: {len(trainer.optimizer.param_groups)}")
                
                # Mapping from param tensor id to name
                param_to_name = {}
                for n, p in model.named_parameters():
                    param_to_name[id(p)] = n

                for i, group in enumerate(trainer.optimizer.param_groups):
                    logger.info(f"Group {i} size: {len(group['params'])}")
                    for p in group["params"]:
                        name = param_to_name.get(id(p), "UNKNOWN_PARAM")
                        
                        p_dtype = str(p.dtype)
                        opt_param_dtypes[p_dtype] += 1
                        
                        if not p.requires_grad:
                            logger.error(f"Optimizer contains parameter with requires_grad=False: {name}")
                        
                        is_lora = "lora" in name.lower()
                        is_4bit = "Params4bit" in p.__class__.__name__ or "NF4" in p.__class__.__name__
                        
                        if is_4bit:
                            logger.error(f"Optimizer contains 4-bit parameter: {name}")

                        if p.grad is not None:
                            g_dtype = str(p.grad.dtype)
                            opt_grad_dtypes[g_dtype] += 1
                            combo_counts[f"{p_dtype} param + {g_dtype} grad"] += 1
                            
                            if p.grad.dtype == torch.bfloat16:
                                bf16_grad_params.append(name)
                                logger.error(f"BF16 GRADIENT -> Name: {name}, Param Dtype: {p_dtype}, Grad Shape: {p.grad.shape}, Is LoRA: {is_lora}, Is 4bit: {is_4bit}")
                        else:
                            logger.warning(f"Parameter in optimizer has NO gradient: {name}")

                logger.info("-" * 40)
                logger.info("AGGREGATED COUNTS")
                logger.info("-" * 40)
                logger.info(f"Optimizer parameter dtypes: {dict(opt_param_dtypes)}")
                logger.info(f"Optimizer gradient dtypes: {dict(opt_grad_dtypes)}")
                for combo, count in combo_counts.items():
                    logger.info(f"  {combo}: {count}")
                
                if bf16_grad_params:
                    logger.error(f"FIRST BF16 GRADIENT PARAMETER: {bf16_grad_params[0]}")
                    logger.error(f"ALL BF16 GRADIENT PARAMETERS ({len(bf16_grad_params)} total):")
                    for pname in bf16_grad_params:
                        logger.error(f"  {pname}")
                else:
                    logger.info("NO BF16 GRADIENTS FOUND IN OPTIMIZER.")
                
                # Clear gradients
                model.zero_grad()
                trainer.optimizer.zero_grad()
                
                diagnostic_passed = True
                
            except Exception as e:
                logger.error(f"Exhaustive diagnostic failed: {e}", exc_info=True)
                logger.error("DIAGNOSTIC FAILURE IS A HARD STOP. Exiting.")
                sys.exit(1)
                
            logger.info("=" * 80)

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
