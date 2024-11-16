# config.py
class TrainingConfig:
    def __init__(self):
        self.output_dir = "/content/drive/MyDrive/LLM_training_output_1Nov2024_V3"
        self.num_train_epochs = 1/70 
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 2e-4
        self.fp16 = True
        self.save_total_limit = 1
        self.logging_steps = 10
        self.save_steps = 500
        self.eval_steps = 500
        self.disable_tqdm = False
        self.eval_strategy = "epoch"
        self.save_strategy = "epoch"
        self.warmup_ratio = 0.1
        self.lr_scheduler_type = "cosine"
        self.evaluation_strategy = "epoch"
        self.push_to_hub = False
        self.report_to = "none"
        self.load_best_model_at_end = True
        self.remove_unused_columns = False
        self.resume_from_checkpoint=True
        self.logging_dir = "/content/"