#!/usr/bin/env python3
"""CLI utility to enroll an individual into the face database."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.enrollment.service import EnrollmentService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enroll_cli")


def main():
    parser = argparse.ArgumentParser(
        description="Enroll an individual's face into the local biometric database."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Person name or identifier (e.g. 'Alice Smith')",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image containing the person's face",
    )
    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Bypass blur and resolution quality checks",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Specified image file does not exist: {image_path}", file=sys.stderr)
        sys.exit(1)

    service = EnrollmentService()
    result = service.enroll(
        name=args.name,
        image=image_path,
        allow_save_image=True,
        skip_quality_check=args.skip_qc,
    )

    if result.success:
        print("\n" + "=" * 50)
        print("          ENROLLMENT SUCCESSFUL")
        print("=" * 50)
        print(f"Identity Name:              {result.person_name}")
        print(f"Person ID:                  {result.person_id}")
        print(f"Embedding ID:               {result.embedding_id}")
        print(f"Total Stored Embeddings:    {result.total_embeddings_for_person}")
        if result.quality:
            print(
                f"Face Crop Size:             {result.quality.face_width}x{result.quality.face_height} px"
            )
            print(f"Sharpness Score:            {result.quality.blur_score:.1f}")
        print(f"Status:                     {result.message}")
        print("=" * 50 + "\n")
        sys.exit(0)
    else:
        print("\n" + "!" * 50)
        print("          ENROLLMENT FAILED")
        print("!" * 50)
        print(f"Identity: {result.person_name}")
        print(f"Reason:   {result.message}")
        print("!" * 50 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
