#!/usr/bin/env python3
"""CLI utility to identify faces in an image against the enrolled database."""

import argparse
import logging
import sys
from pathlib import Path
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import config
from app.recognition.recognizer import FaceRecognitionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("identify_cli")


def main():
    parser = argparse.ArgumentParser(
        description="Identify faces in an input image against enrolled identities."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to query image",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.match_threshold,
        help=f"Cosine similarity threshold for acceptance (default: {config.match_threshold})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save annotated visualization image",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Specified image file does not exist: {image_path}", file=sys.stderr)
        sys.exit(1)

    service = FaceRecognitionService()
    output = service.identify(image=image_path, threshold=args.threshold)

    print("\n" + "=" * 60)
    print("                IDENTIFICATION RESULTS")
    print("=" * 60)
    print(f"Query Image:       {image_path}")
    print(f"Threshold Applied: {args.threshold:.2f}")
    print(f"Faces Detected:    {output.num_faces_detected}")
    print(f"Matches:           {output.num_matches}")
    print(f"Unknowns:          {output.num_unknowns}")
    print("-" * 60)

    if output.num_faces_detected == 0:
        print("No faces detected in the provided image.")
    else:
        for res in output.results:
            status_indicator = "MATCH [ACCEPTED]" if res.accepted else "UNKNOWN [REJECTED]"
            print(f"Face #{res.face_index + 1}:")
            print(f"  Bounding Box:        {res.bbox}")
            print(f"  Detection Conf:      {res.detection_confidence:.2f}")
            print(f"  Predicted Identity:  {res.identity if res.identity else 'UNKNOWN'}")
            print(f"  Similarity Score:    {res.similarity:.4f}")
            print(f"  Decision Status:     {status_indicator}")
            if res.top_candidates:
                top_str = ", ".join(
                    [f"{c.person_name} ({c.similarity:.3f})" for c in res.top_candidates[:3]]
                )
                print(f"  Top Candidates:      {top_str}")
            print("-" * 60)

    # Save output visualization if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), output.annotated_image)
        print(f"Annotated visualization saved to: {out_path}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
