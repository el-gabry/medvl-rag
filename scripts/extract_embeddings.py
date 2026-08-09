"""CLI for extracting and saving image-report embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from medvl_rag.config import (
    load_config,
    parse_encoder_config,
)
from medvl_rag.data import ImageReportDataset
from medvl_rag.embedding_io import save_embedding_result
from medvl_rag.embeddings import extract_embeddings
from medvl_rag.encoders.factory import create_encoder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract image and report embeddings from a dataset "
            "manifest using a configured vision-language encoder."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the dataset manifest.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where extracted embeddings will be saved.",
    )

    return parser.parse_args()


def validate_paths(
    config_path: Path,
    manifest_path: Path,
) -> None:
    """Validate input paths before loading the pipeline."""

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_path}"
        )


def main() -> None:
    """Run end-to-end embedding extraction."""

    args = parse_args()

    validate_paths(
        args.config,
        args.manifest,
    )

    print("=" * 64)
    print("MedVL-RAG embedding extraction")
    print("=" * 64)

    print()
    print(f"Config:   {args.config}")
    print(f"Manifest: {args.manifest}")
    print(f"Output:   {args.output_dir}")

    print()
    print("Loading configuration...")

    raw_config = load_config(
        args.config
    )

    encoder_config = parse_encoder_config(
        raw_config
    )

    print(
        f"Encoder:    {encoder_config.name}"
    )
    print(
        f"Device:     {encoder_config.device}"
    )
    print(
        f"Batch size: {encoder_config.batch_size}"
    )

    if encoder_config.model_name is not None:
        print(
            f"Model:      {encoder_config.model_name}"
        )

    print()
    print("Loading dataset...")

    dataset = ImageReportDataset(
        args.manifest
    )

    dataset_size = len(dataset)

    if dataset_size < 1:
        raise ValueError(
            "Dataset manifest contains no samples."
        )

    print(
        f"Samples: {dataset_size}"
    )

    print()
    print("Creating encoder...")

    encoder = create_encoder(
        encoder_config
    )

    print(
        f"Resolved device: {encoder.device}"
    )

    print()
    print("Extracting embeddings...")

    result = extract_embeddings(
        encoder,
        dataset,
        batch_size=encoder_config.batch_size,
    )

    print()
    print("Saving embeddings...")

    output_path = save_embedding_result(
        result,
        args.output_dir,
    )

    print()
    print("=" * 64)
    print("Embedding extraction completed successfully")
    print("=" * 64)

    print(
        f"Samples:             {result.size}"
    )
    print(
        f"Embedding dimension: {result.embedding_dim}"
    )
    print(
        f"Image embeddings:    "
        f"{tuple(result.image_embeddings.shape)}"
    )
    print(
        f"Text embeddings:     "
        f"{tuple(result.text_embeddings.shape)}"
    )
    print(
        f"Saved to:            {output_path}"
    )


if __name__ == "__main__":
    main()