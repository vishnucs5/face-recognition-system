#!/usr/bin/env python3
"""Generates reproducible sample biometric data for demonstration, automated testing, and evaluation."""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.database import DatabaseManager
from app.enrollment.service import EnrollmentService


def create_sample_dataset(enroll_identities: bool = True):
    """Extract sample faces from standard benchmark and partition into known and unknown sets."""
    print("Loading benchmark face data...")
    dataset = fetch_olivetti_faces()
    images = dataset.images  # (400, 64, 64) in [0, 1]
    targets = dataset.target  # (400,)

    # Directory layout
    known_test_dir = config.test_dir / "known"
    unknown_test_dir = config.test_dir / "unknown"
    enrolled_ref_dir = config.project_root / "data" / "enrolled_samples"

    known_test_dir.mkdir(parents=True, exist_ok=True)
    unknown_test_dir.mkdir(parents=True, exist_ok=True)
    enrolled_ref_dir.mkdir(parents=True, exist_ok=True)

    identities_map = {
        0: "Alice",
        1: "Bob",
        2: "Charlie",
    }
    unknown_subjects = [10, 11, 12]

    # Preprocess helper: scale up 64x64 to 224x224 and convert to 3-channel BGR
    def format_face(img_float):
        img_uint8 = (img_float * 255).astype(np.uint8)
        img_resized = cv2.resize(img_uint8, (224, 224), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

    enroll_service = EnrollmentService() if enroll_identities else None

    print("\n1. Preparing Enrolled & Known Test Faces:")
    for subj_id, name in identities_map.items():
        indices = np.where(targets == subj_id)[0]
        person_test_dir = known_test_dir / name
        person_test_dir.mkdir(parents=True, exist_ok=True)
        person_ref_dir = enrolled_ref_dir / name
        person_ref_dir.mkdir(parents=True, exist_ok=True)

        # Image 0 & 1 for enrollment
        for i, idx in enumerate(indices[:2]):
            face_bgr = format_face(images[idx])
            ref_path = person_ref_dir / f"{name.lower()}_ref_{i+1}.jpg"
            cv2.imwrite(str(ref_path), face_bgr)

            if enroll_service:
                res = enroll_service.enroll(
                    name=name,
                    image=ref_path,
                    allow_save_image=False,
                    skip_quality_check=True,
                )
                print(f"  Enrolled {name} (sample #{i+1}): {res.message}")

        # Image 2, 3, 4 for known testing (disjoint from enrollment!)
        for i, idx in enumerate(indices[2:5]):
            face_bgr = format_face(images[idx])
            test_path = person_test_dir / f"{name.lower()}_test_{i+1}.jpg"
            cv2.imwrite(str(test_path), face_bgr)
            print(f"  Saved test image for {name}: {test_path.name}")

    print("\n2. Preparing Unknown Test Faces (Disjoint identities):")
    for subj_id in unknown_subjects:
        indices = np.where(targets == subj_id)[0]
        subj_dir = unknown_test_dir / f"person_subject_{subj_id}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(indices[:3]):
            face_bgr = format_face(images[idx])
            unk_path = subj_dir / f"unknown_subj_{subj_id}_{i+1}.jpg"
            cv2.imwrite(str(unk_path), face_bgr)
            print(f"  Saved unknown test image: {unk_path.name}")

    print("\n3. Creating Multi-Face Query Image for demonstration:")
    # Collage containing Alice, Bob, and Unknown Person
    alice_img = format_face(images[np.where(targets == 0)[0][5]])
    bob_img = format_face(images[np.where(targets == 1)[0][5]])
    unk_img = format_face(images[np.where(targets == 10)[0][5]])

    collage = np.hstack([alice_img, bob_img, unk_img])
    collage_path = config.test_dir / "multi_face_query.jpg"
    cv2.imwrite(str(collage_path), collage)
    print(f"  Multi-face image created: {collage_path}")

    print("\nSample dataset generation completed successfully.")


if __name__ == "__main__":
    create_sample_dataset(enroll_identities=True)
