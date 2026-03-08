import base64
import json
import re
from pathlib import Path

import cv2


def clamp_box(box, width, height):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def shift_box(local_box, view_box):
    vx1, vy1, _, _ = view_box
    x1, y1, x2, y2 = local_box
    return [x1 + vx1, y1 + vy1, x2 + vx1, y2 + vy1]


def extract_json_array_text(raw_text):
    text = (raw_text or "").strip()
    if text.startswith("```json"):
        text = text[7:-3].strip()
    elif text.startswith("```"):
        text = text[3:-3].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def normalize_llm_string(value):
    value = str(value).strip()
    value = value.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    value = value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    value = value.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_relaxed_string_field(chunk, key, next_keys):
    quoted_next_keys = "|".join(re.escape(next_key) for next_key in next_keys)
    patterns = [
        rf'["\']{re.escape(key)}["\']\s*:\s*"(?P<value>.*?)(?="\s*(?:,\s*["\'](?:{quoted_next_keys})["\']\s*:|\s*[,}}]))',
        rf'["\']{re.escape(key)}["\']\s*:\s*\'(?P<value>.*?)(?=\'\s*(?:,\s*["\'](?:{quoted_next_keys})["\']\s*:|\s*[,}}]))',
        rf'["\']{re.escape(key)}["\']\s*:\s*(?P<value>[^,\n}}]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, chunk, re.DOTALL)
        if match:
            return normalize_llm_string(match.group("value"))
    return None


def extract_relaxed_box_field(chunk):
    match = re.search(r'["\']box["\']\s*:\s*\[(?P<value>[^\]]+)\]', chunk, re.DOTALL)
    if not match:
        return None
    numbers = re.findall(r"-?\d+(?:\.\d+)?", match.group("value"))
    if len(numbers) < 4:
        return None
    return [float(number) for number in numbers[:4]]


def parse_vision_json_relaxed(text):
    object_chunks = re.findall(r"\{.*?\}", text, re.DOTALL)
    items = []
    for chunk in object_chunks:
        text_value = extract_relaxed_string_field(chunk, "text", next_keys=["type", "box"])
        type_value = extract_relaxed_string_field(chunk, "type", next_keys=["text", "box"]) or "unknown"
        box_value = extract_relaxed_box_field(chunk)
        if not text_value or box_value is None:
            continue
        items.append(
            {
                "text": text_value,
                "type": type_value,
                "box": box_value,
            }
        )

    if items:
        return items

    raise ValueError(
        "Vision response JSON repair failed. "
        f"Preview: {normalize_llm_string(text)[:400]}"
    )


def parse_vision_json(raw_text):
    text = extract_json_array_text(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = parse_vision_json_relaxed(text)
    if not isinstance(data, list):
        raise ValueError("Vision response JSON is not an array.")
    return data


def canonicalize_dimension_text(text):
    normalized = (text or "").upper().strip()
    normalized = normalized.replace("Ø", "")
    normalized = normalized.replace("°", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("SØ", "S")
    return normalized


def normalize_view_records(views):
    normalized = []
    for idx, view in enumerate(views, start=1):
        view_type = view.get("view_type", "unknown_view")
        normalized.append(
            {
                "view_index": idx,
                "view_id": f"view_{idx:02d}",
                "view_type": view_type,
                "bbox": [int(v) for v in view.get("box", [0, 0, 1, 1])],
                "source": view.get("source", "unknown"),
                "layout_label": view.get("layout_label"),
                "score": view.get("score"),
            }
        )
    return normalized


def build_bbox_payload(pdf_path, image_path, page_shape, views):
    height, width = page_shape[:2]
    return {
        "pdf_path": str(pdf_path),
        "image_path": str(image_path),
        "page_size": {"width": int(width), "height": int(height)},
        "view_count": len(views),
        "views": normalize_view_records(views),
    }


def save_bbox_payload(payload, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(payload["pdf_path"]).stem
    out_path = output_dir / f"{stem}_view_bboxes.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def encode_image_bgr_base64(image_bgr, ext=".png"):
    ok, encoded = cv2.imencode(ext, image_bgr)
    if not ok:
        raise RuntimeError("이미지 인코딩에 실패했습니다.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def crop_from_view(page_bgr, view):
    x1, y1, x2, y2 = view["bbox"]
    crop = page_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise RuntimeError(f"Empty crop for view: {view['view_id']}")
    return crop


def infer_dimension_schema(text, item_type):
    raw = (text or "").strip()
    normalized = canonicalize_dimension_text(raw)
    lowered_type = (item_type or "unknown").strip().lower()

    schema = {
        "raw_text": raw,
        "normalized_text": normalized,
        "declared_type": item_type or "unknown",
        "category": "unknown",
        "nominal": None,
        "tolerance_plus": None,
        "tolerance_minus": None,
        "diameter": False,
        "radius": False,
        "angle_deg": None,
        "roughness_ra": None,
        "thread_designation": None,
    }

    compact = raw.replace(" ", "")
    upper = compact.upper()

    if "RA" in upper:
        schema["category"] = "roughness"
        match = re.search(r"RA\s*([0-9]+(?:\.[0-9]+)?)", upper)
        if match:
            schema["roughness_ra"] = float(match.group(1))
        return schema

    if upper.startswith("R"):
        schema["category"] = "radius"
        schema["radius"] = True
        number_match = re.search(r"R\s*([0-9]+(?:\.[0-9]+)?)", upper)
        if number_match:
            schema["nominal"] = float(number_match.group(1))
        return schema

    if "°" in raw or lowered_type == "angle":
        schema["category"] = "angle"
        angle_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*°", raw)
        if angle_match:
            schema["angle_deg"] = float(angle_match.group(1))
            schema["nominal"] = schema["angle_deg"]
        return schema

    if re.search(r"\bM[0-9]+(?:\.[0-9]+)?[Xx][0-9]+(?:\.[0-9]+)?", upper) or re.search(
        r"[0-9]+/[0-9]+\"?-[0-9]+", upper
    ):
        schema["category"] = "thread"
        schema["thread_designation"] = raw
        return schema

    if upper.startswith("C") and ("X" in upper or "×" in upper):
        schema["category"] = "chamfer"
    elif "Ø" in raw or "DIAMETER" in lowered_type:
        schema["category"] = "diameter"
        schema["diameter"] = True
    elif "TOLERANCE" in lowered_type:
        schema["category"] = "tolerance"
    else:
        schema["category"] = "linear"

    tol_match = re.search(
        r"(?P<nom>[0-9]+(?:\.[0-9]+)?)(?P<plus>[+-][0-9]+(?:\.[0-9]+)?)/(?P<minus>-?[0-9]+(?:\.[0-9]+)?)",
        upper,
    )
    if tol_match:
        schema["nominal"] = float(tol_match.group("nom"))
        schema["tolerance_plus"] = float(tol_match.group("plus"))
        schema["tolerance_minus"] = float(tol_match.group("minus"))
        return schema

    pm_match = re.search(r"(?P<nom>[0-9]+(?:\.[0-9]+)?)±(?P<tol>[0-9]+(?:\.[0-9]+)?)", upper)
    if pm_match:
        schema["nominal"] = float(pm_match.group("nom"))
        tol_value = float(pm_match.group("tol"))
        schema["tolerance_plus"] = tol_value
        schema["tolerance_minus"] = -tol_value
        return schema

    leading_number_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", upper)
    if leading_number_match:
        schema["nominal"] = float(leading_number_match.group(1))

    return schema


def request_dimensions_for_crop(client, model_name, crop_bgr):
    prompt = (
        "You are reading a cropped mechanical drawing view. "
        "Extract dimensions, tolerances, diameters, radii, threads, chamfers, angles, and roughness marks visible in this crop. "
        "Return JSON array only. "
        "Each element must follow exactly: "
        "[{\"text\":\"dimension text\",\"type\":\"classification\",\"box\":[x1,y1,x2,y2]}]. "
        "Coordinates must be pixel coordinates relative to this crop image."
    )
    base64_img = encode_image_bgr_base64(crop_bgr, ext=".png")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                ],
            }
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or "[]"


def extract_dimensions_for_views(page_bgr, views, client, model_name):
    height, width = page_bgr.shape[:2]
    result_views = []
    for view in normalize_view_records(views):
        crop = crop_from_view(page_bgr, view)
        raw_text = request_dimensions_for_crop(client, model_name, crop)
        try:
            items = parse_vision_json(raw_text)
        except Exception as exc:
            preview = normalize_llm_string(raw_text)[:600]
            raise RuntimeError(
                f"치수 JSON 파싱 실패: view_id={view['view_id']}, "
                f"view_type={view.get('view_type')}, bbox={view.get('bbox')}, "
                f"raw_preview={preview}"
            ) from exc
        crop_h, crop_w = crop.shape[:2]
        structured_items = []
        for item_index, item in enumerate(items, start=1):
            local_box = clamp_box(item.get("box", [0, 0, 1, 1]), crop_w, crop_h)
            global_box = clamp_box(shift_box(local_box, view["bbox"]), width, height)
            structured_items.append(
                {
                    "item_index": item_index,
                    "text": item.get("text", ""),
                    "type": item.get("type", "unknown"),
                    "box": local_box,
                    "global_box": global_box,
                    "parsed": infer_dimension_schema(item.get("text", ""), item.get("type", "unknown")),
                }
            )
        result_views.append(
            {
                **view,
                "dimension_count": len(structured_items),
                "dimensions": structured_items,
                "raw_response": raw_text,
            }
        )
    return result_views


def build_dimension_payload(pdf_path, image_path, page_shape, views_with_dimensions, model_name):
    height, width = page_shape[:2]
    return {
        "pdf_path": str(pdf_path),
        "image_path": str(image_path),
        "model_name": model_name,
        "page_size": {"width": int(width), "height": int(height)},
        "view_count": len(views_with_dimensions),
        "total_dimensions": sum(view["dimension_count"] for view in views_with_dimensions),
        "views": views_with_dimensions,
    }


def save_dimension_payload(payload, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(payload["pdf_path"]).stem
    out_path = output_dir / f"{stem}_view_dimensions.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _normalized_dimension_text(item):
    parsed = item.get("parsed", {}) if isinstance(item, dict) else {}
    return parsed.get("normalized_text") or canonicalize_dimension_text(item.get("text", ""))


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_close(a, b, abs_tol=0.15, rel_tol=0.01):
    if a is None or b is None:
        return False
    diff = abs(a - b)
    return diff <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1.0))


def is_dimension_like(item):
    parsed = item.get("parsed", {}) if isinstance(item, dict) else {}
    text = _normalized_dimension_text(item)
    if re.search(r"\d", text):
        return True
    if parsed.get("thread_designation"):
        return True
    if parsed.get("roughness_ra") is not None:
        return True
    if parsed.get("angle_deg") is not None:
        return True
    return False


def dimension_match_score(reference_item, candidate_item):
    ref_parsed = reference_item.get("parsed", {})
    cand_parsed = candidate_item.get("parsed", {})
    ref_text = _normalized_dimension_text(reference_item)
    cand_text = _normalized_dimension_text(candidate_item)

    if ref_text and cand_text and ref_text == cand_text:
        return 1.0, "exact_text"

    ref_thread = canonicalize_dimension_text(ref_parsed.get("thread_designation", ""))
    cand_thread = canonicalize_dimension_text(cand_parsed.get("thread_designation", ""))
    if ref_thread and cand_thread and ref_thread == cand_thread:
        return 0.97, "exact_thread"

    ref_rough = _float_or_none(ref_parsed.get("roughness_ra"))
    cand_rough = _float_or_none(cand_parsed.get("roughness_ra"))
    if ref_rough is not None and cand_rough is not None:
        if _float_close(ref_rough, cand_rough, abs_tol=0.05, rel_tol=0.05):
            return 0.96, "roughness_close"
        if _float_close(ref_rough, cand_rough, abs_tol=0.2, rel_tol=0.15):
            return 0.82, "roughness_near"

    ref_angle = _float_or_none(ref_parsed.get("angle_deg"))
    cand_angle = _float_or_none(cand_parsed.get("angle_deg"))
    if ref_angle is not None and cand_angle is not None:
        if _float_close(ref_angle, cand_angle, abs_tol=0.2, rel_tol=0.02):
            return 0.95, "angle_close"
        if _float_close(ref_angle, cand_angle, abs_tol=1.0, rel_tol=0.05):
            return 0.75, "angle_near"

    ref_nominal = _float_or_none(ref_parsed.get("nominal"))
    cand_nominal = _float_or_none(cand_parsed.get("nominal"))
    ref_plus = _float_or_none(ref_parsed.get("tolerance_plus"))
    cand_plus = _float_or_none(cand_parsed.get("tolerance_plus"))
    ref_minus = _float_or_none(ref_parsed.get("tolerance_minus"))
    cand_minus = _float_or_none(cand_parsed.get("tolerance_minus"))
    ref_category = (ref_parsed.get("category") or reference_item.get("type") or "unknown").lower()
    cand_category = (cand_parsed.get("category") or candidate_item.get("type") or "unknown").lower()

    if ref_nominal is not None and cand_nominal is not None and _float_close(ref_nominal, cand_nominal):
        score = 0.68
        reason = "nominal_close"
        if ref_category == cand_category:
            score += 0.12
            reason = "category_and_nominal_close"
        if bool(ref_parsed.get("diameter")) == bool(cand_parsed.get("diameter")) and bool(ref_parsed.get("diameter")):
            score += 0.06
        if bool(ref_parsed.get("radius")) == bool(cand_parsed.get("radius")) and bool(ref_parsed.get("radius")):
            score += 0.06
        plus_match = _float_close(ref_plus, cand_plus, abs_tol=0.05, rel_tol=0.05) if ref_plus is not None and cand_plus is not None else ref_plus == cand_plus
        minus_match = _float_close(ref_minus, cand_minus, abs_tol=0.05, rel_tol=0.05) if ref_minus is not None and cand_minus is not None else ref_minus == cand_minus
        if plus_match and minus_match and (ref_plus is not None or ref_minus is not None):
            score += 0.12
            reason = "nominal_tolerance_close"
        elif ref_plus is None and cand_plus is None and ref_minus is None and cand_minus is None:
            score += 0.05
        return min(score, 0.96), reason

    if ref_category == cand_category and ref_text and cand_text and (ref_text in cand_text or cand_text in ref_text):
        return 0.72, "text_contains"

    if ref_text and cand_text and len(ref_text) >= 3 and len(cand_text) >= 3:
        if ref_text[:4] == cand_text[:4]:
            return 0.58, "prefix_close"

    return 0.0, "no_match"


def classify_match_score(score):
    if score >= 0.95:
        return "exact"
    if score >= 0.80:
        return "strong"
    if score >= 0.60:
        return "partial"
    if score >= 0.45:
        return "weak"
    return "none"


def compare_dimension_lists(reference_items, candidate_items, match_threshold=0.60):
    ref_items = [item for item in reference_items if is_dimension_like(item)]
    cand_items = [item for item in candidate_items if is_dimension_like(item)]

    candidate_pairs = []
    for ref_index, ref_item in enumerate(ref_items):
        for cand_index, cand_item in enumerate(cand_items):
            score, reason = dimension_match_score(ref_item, cand_item)
            if score >= match_threshold:
                candidate_pairs.append((score, ref_index, cand_index, reason))
    candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_ref = set()
    used_cand = set()
    matches = []
    for score, ref_index, cand_index, reason in candidate_pairs:
        if ref_index in used_ref or cand_index in used_cand:
            continue
        used_ref.add(ref_index)
        used_cand.add(cand_index)
        matches.append(
            {
                "reference_index": ref_index,
                "candidate_index": cand_index,
                "score": round(float(score), 4),
                "classification": classify_match_score(score),
                "reason": reason,
                "reference_item": ref_items[ref_index],
                "candidate_item": cand_items[cand_index],
            }
        )

    unmatched_reference = [item for idx, item in enumerate(ref_items) if idx not in used_ref]
    unmatched_candidate = [item for idx, item in enumerate(cand_items) if idx not in used_cand]

    exact = sum(1 for match in matches if match["classification"] == "exact")
    strong = sum(1 for match in matches if match["classification"] == "strong")
    partial = sum(1 for match in matches if match["classification"] == "partial")
    weak = sum(1 for match in matches if match["classification"] == "weak")

    weighted_match = (
        exact * 1.00
        + strong * 0.85
        + partial * 0.65
        + weak * 0.40
    )
    denominator = max(len(ref_items), len(cand_items), 1)
    similarity = weighted_match / denominator

    return {
        "reference_count_raw": len(reference_items),
        "candidate_count_raw": len(candidate_items),
        "reference_count": len(ref_items),
        "candidate_count": len(cand_items),
        "match_count": len(matches),
        "exact_match_count": exact,
        "strong_match_count": strong,
        "partial_match_count": partial,
        "weak_match_count": weak,
        "unmatched_reference_count": len(unmatched_reference),
        "unmatched_candidate_count": len(unmatched_candidate),
        "similarity_score": round(float(similarity), 4),
        "matches": matches,
        "unmatched_reference": unmatched_reference,
        "unmatched_candidate": unmatched_candidate,
    }


def compare_dimension_payloads(reference_payload, candidate_payload):
    reference_views = {view["view_id"]: view for view in reference_payload.get("views", [])}
    candidate_views = {view["view_id"]: view for view in candidate_payload.get("views", [])}
    shared_view_ids = sorted(set(reference_views) | set(candidate_views))

    per_view = []
    weighted_sum = 0.0
    weight_total = 0.0
    total_ref = 0
    total_cand = 0
    total_exact = 0
    total_match = 0

    for view_id in shared_view_ids:
        ref_view = reference_views.get(view_id, {"view_id": view_id, "view_type": "missing", "dimensions": []})
        cand_view = candidate_views.get(view_id, {"view_id": view_id, "view_type": "missing", "dimensions": []})
        view_result = compare_dimension_lists(ref_view.get("dimensions", []), cand_view.get("dimensions", []))
        view_result.update(
            {
                "view_id": view_id,
                "reference_view_type": ref_view.get("view_type", "missing"),
                "candidate_view_type": cand_view.get("view_type", "missing"),
                "view_type_match": ref_view.get("view_type") == cand_view.get("view_type"),
            }
        )

        if view_result["reference_count"] == 0 and view_result["candidate_count"] == 0:
            status = "no_dimensions"
        elif (
            view_result["reference_count"] == view_result["candidate_count"]
            and view_result["reference_count"] > 0
            and view_result["exact_match_count"] == view_result["reference_count"]
            and view_result["view_type_match"]
        ):
            status = "identical"
        elif view_result["similarity_score"] >= 0.85:
            status = "high_similarity"
        elif view_result["similarity_score"] >= 0.60:
            status = "partial_similarity"
        else:
            status = "low_similarity"
        view_result["status"] = status

        weight = max(view_result["reference_count"], view_result["candidate_count"], 1)
        weighted_sum += view_result["similarity_score"] * weight
        weight_total += weight
        total_ref += view_result["reference_count"]
        total_cand += view_result["candidate_count"]
        total_exact += view_result["exact_match_count"]
        total_match += view_result["match_count"]
        per_view.append(view_result)

    overall_similarity = weighted_sum / max(weight_total, 1.0)
    if overall_similarity >= 0.90:
        overall_status = "very_high_match"
    elif overall_similarity >= 0.75:
        overall_status = "high_match"
    elif overall_similarity >= 0.55:
        overall_status = "partial_match"
    else:
        overall_status = "low_match"

    return {
        "reference_pdf": reference_payload.get("pdf_path"),
        "candidate_pdf": candidate_payload.get("pdf_path"),
        "summary": {
            "shared_view_count": len(shared_view_ids),
            "reference_total_dimensions": total_ref,
            "candidate_total_dimensions": total_cand,
            "total_exact_matches": total_exact,
            "total_matched_pairs": total_match,
            "overall_similarity": round(float(overall_similarity), 4),
            "overall_status": overall_status,
        },
        "per_view": per_view,
    }


def compare_dimension_json_files(reference_path, candidate_path):
    reference_payload = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    candidate_payload = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    return compare_dimension_payloads(reference_payload, candidate_payload)


def save_comparison_payload(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_stem = Path(report["reference_pdf"]).stem
    cand_stem = Path(report["candidate_pdf"]).stem
    out_path = output_dir / f"{ref_stem}__vs__{cand_stem}_view_dimension_comparison.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
