import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from interviews.ml_answer_classifier import (
    DATASET_PATH,
    MODEL_PATH,
    train_and_save_model,
)


class Command(BaseCommand):
    help = "Train the experimental answer-text classifier and report its held-out metrics."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dataset",
            default=str(DATASET_PATH),
            help="CSV training dataset path.",
        )
        parser.add_argument(
            "--output",
            default=str(MODEL_PATH),
            help="Trusted local joblib artifact output path.",
        )

    def handle(self, *args, **options):
        dataset_path = Path(options["dataset"])
        model_path = Path(options["output"])
        try:
            metrics, metadata = train_and_save_model(dataset_path, model_path)
        except (OSError, ValueError) as error:
            raise CommandError(f"Experimental classifier training failed: {error}") from error

        self.stdout.write(json.dumps({
            "model_path": str(model_path),
            "sklearn_version": metadata["sklearn_version"],
            "evaluation": metrics,
            "notice": metadata["data_notice"],
        }, indent=2))
