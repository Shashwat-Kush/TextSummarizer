from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from datasets import load_dataset, load_from_disk
import torch
import os
from textSummarizer.logging import logger
from textSummarizer.entity import ModelTrainerConfig




class ModelTrainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.join(self.config.root_dir, 'pegasus-samsum-model')
        tokenizer_path = os.path.join(self.config.root_dir, 'tokenizer')
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            logger.info(f"✅ Found existing model at {model_path}. Skipping training!")
            return
        device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer,model = model_pegasus)

        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir = self.config.root_dir,
            per_device_train_batch_size = self.config.per_device_train_batch_size,
            per_device_eval_batch_size = self.config.per_device_eval_batch_size,
            num_train_epochs = self.config.num_train_epochs,
            warmup_steps = self.config.warmup_steps,
            weight_decay = self.config.weight_decay,
            logging_steps = self.config.logging_steps,
            eval_strategy = self.config.eval_strategy,
            eval_steps = self.config.eval_steps,
            save_steps = 1e6,
            gradient_accumulation_steps = self.config.gradient_accumulation_steps,
            use_cpu = True
        )

        trainer = Trainer(
            model = model_pegasus,args = trainer_args,
            tokenizer = tokenizer, data_collator = seq2seq_data_collator,
            train_dataset = dataset_samsum_pt['test'],
            eval_dataset=dataset_samsum_pt['validation']
        )

        trainer.train()

        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,'pegasus-samsum-model'))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,'tokenizer'))