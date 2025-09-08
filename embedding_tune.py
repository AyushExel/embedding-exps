import os
import logging
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers.training_args import BatchSamplers
from huggingface_hub import login as hf_login
import wandb

WANDB_DISABLED = False

def setup_env(cache_dir="/models"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    os.environ["WANDB_PROJECT"] = "xyneft"
    if not os.environ["WANDB_API_KEY"]:
        WANDB_DISABLED = True


def prepare_datasets(dataset_name="ayushexel/xyneft", sample_size=1080, test_size=250, seed=12):
    ds = load_dataset(dataset_name, split="train").select(range(sample_size))
    split = ds.train_test_split(test_size=test_size, seed=seed)
    return ds, split["train"], split["test"]

def mine_negatives(dataset, embed_model, num_negatives=1, batch_size=512, **kwargs):
    return mine_hard_negatives(
        dataset,
        embed_model,
        num_negatives=num_negatives,
        batch_size=batch_size,
        output_format="triplet",
        **kwargs
    )

def evaluate_baseline(model, ds_full, ds_eval, evaluator_name="baseline"):
    embed_model_cpu = SentenceTransformer(
        "sentence-transformers/static-retrieval-mrl-en-v1", device="cpu"
    )

    hard_eval = mine_negatives(
        ds_eval,
        embed_model=embed_model_cpu,
        corpus=list(ds_full["answer"]),
        include_positives=True
    )

    evaluator = TripletEvaluator(
        anchors=hard_eval["query"],
        positives=hard_eval["answer"],
        negatives=hard_eval["negative_1"],
        name=evaluator_name
    )

    logging.info(f"Evaluating baseline with {evaluator_name} evaluator")
    results = evaluator(model)
    logging.info(f"Baseline results: {results}")
    return results


def train_model(num_epochs=1, cache_dir="/models", learning_rate=3e-5, weight_decay=0.01, warmup_ratio=0.1):
    hf_login(token=os.environ.get("HF_TOKEN"))
    wandb.login(key=os.environ["WANDB_API_KEY"]) if not WANDB_DISABLED else None

    ds_full, ds_train, ds_eval = prepare_datasets()
    model_name = "google/embeddinggemma-300m"
    model = SentenceTransformer(model_name)

    # Baseline evaluation before fine-tuning
    evaluate_baseline(model, ds_full, ds_eval, evaluator_name="baseline_pre_training")

    embed_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")

    hard_train = concatenate_datasets([
        mine_negatives(ds_train.select(range(i, min(i + 200_000, len(ds_train)))), embed_model, margin=0, range_min=0, range_max=100, sampling_strategy="top")
        for i in range(0, len(ds_train), 200_000)
    ])

    hard_eval = mine_negatives(ds_eval, embed_model, corpus=list(ds_full["answer"]), include_positives=True)

    loss = MultipleNegativesRankingLoss(SentenceTransformer(model_name))
    evaluator = TripletEvaluator(
        anchors=hard_eval["query"],
        positives=hard_eval["answer"],
        negatives=hard_eval["negative_1"],
        name="ft-dev"
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(cache_dir, f"{model_name.split('/')[-1]}-{num_epochs}e"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        fp16=False,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        logging_steps=10,
        run_name=f"xynft-{model_name.split('/')[-1]}-{num_epochs}e",
        report_to="wandb"
    )

    trainer = SentenceTransformerTrainer(
        model=SentenceTransformer(model_name),
        args=args,
        train_dataset=hard_train,
        eval_dataset=ds_eval,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()
    evaluator(trainer.model)
    final_dir = os.path.join(cache_dir, f"{args.run_name}/model/final")
    trainer.model.save_pretrained(final_dir)
    trainer.model.push_to_hub(args.run_name)

def run_train_loop(start_epoch=2, end_epoch=11):
    for epoch in range(start_epoch, end_epoch):
        train_model(num_epochs=epoch)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    setup_env()
    run_train_loop()
