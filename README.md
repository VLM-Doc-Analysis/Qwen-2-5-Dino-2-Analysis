# Qwen 2.5 + DINO Drawing Analysis

This repository keeps notebook-based experiments for drawing view segmentation and dimension extraction.

## Utility module

`bbox_dimension_utils.py` is placed at the repository root, so notebooks and scripts in this repo can import it directly:

```python
import bbox_dimension_utils as bdu
```

## Example usage

### 1) Save detected view bounding boxes

```python
from pathlib import Path
import bbox_dimension_utils as bdu

payload = bdu.build_bbox_payload(
    pdf_path="sample.pdf",
    image_path="sample_page.png",
    page_shape=page_bgr.shape,
    views=views,
)

output_path = bdu.save_bbox_payload(payload, Path("bbox_exports"))
print(output_path)
```

### 2) Extract dimensions for each view

```python
from openai import OpenAI
import bbox_dimension_utils as bdu

client = OpenAI()
model_name = "gpt-4.1"

views_with_dimensions = bdu.extract_dimensions_for_views(
    page_bgr=page_bgr,
    views=views,
    client=client,
    model_name=model_name,
)

payload = bdu.build_dimension_payload(
    pdf_path="sample.pdf",
    image_path="sample_page.png",
    page_shape=page_bgr.shape,
    views_with_dimensions=views_with_dimensions,
    model_name=model_name,
)

output_path = bdu.save_dimension_payload(payload, Path("bbox_exports"))
print(output_path)
```

### 3) Compare two exported dimension JSON files

```python
import bbox_dimension_utils as bdu

report = bdu.compare_dimension_json_files(
    "bbox_exports/ref_view_dimensions.json",
    "bbox_exports/cand_view_dimensions.json",
)

print(report["summary"])
```

## Notebook usage

The notebooks can use the same root-level import:

```python
import bbox_dimension_utils as bdu
```

This matches the current usage in `bbox_seg.ipynb`.
