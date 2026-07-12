from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def main() -> None:
    output = Path("data/demo")
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    findings = [
        ("normal", "No acute cardiopulmonary abnormality."),
        ("effusion", "Small right pleural effusion."),
        ("opacity", "Left lower lobe airspace opacity."),
        ("cardiomegaly", "Mild enlargement of the cardiac silhouette."),
        ("pneumothorax", "Small left apical pneumothorax."),
        ("edema", "Diffuse bilateral interstitial pulmonary edema."),
    ]

    rows = []
    for index, (label, report) in enumerate(findings):
        image = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((25, 20, 199, 204), outline="black", width=3)
        draw.text((40, 95), label, fill="black")
        image_path = images_dir / f"{index:03d}_{label}.png"
        image.save(image_path)

        rows.append(
            {
                "study_id": f"s{index:04d}",
                "patient_id": f"p{index:04d}",
                "image_path": str(image_path),
                "report": report,
                "split": "test",
                "pathology": label,
                "anatomy": "chest",
                "laterality": "",
                "severity": "",
                "temporal_status": "",
            }
        )

    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False)
    print(f"Created {output / 'manifest.csv'}")


if __name__ == "__main__":
    main()
