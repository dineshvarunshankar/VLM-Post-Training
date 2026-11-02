"""Audited generator source semantics, independent of the active generation runtime."""

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess

from scripts.train.shared.data import read_json, read_yaml
from scripts.train.shared.paths import REPO_ROOT

PROFILE_PATH = REPO_ROOT / "scripts/train/configs/cot/generator-profiles-2026-11-02.json"
TEACHER_PROMPTS = REPO_ROOT / "scripts/train/configs/cot/teacher-prompts/2026-11-02"
TRANSPORT_PROFILE = {"version":1,"codec":"libx264","crf":18,"preset":"medium","frames":"passthrough","resolution":"source","timestamps":"source"}


def current_implementation():
    return deepcopy(read_json(PROFILE_PATH)["current_implementation"])


def known_implementation(implementation):
    profiles = read_json(PROFILE_PATH)
    return implementation == profiles["current_implementation"] or any(implementation == item["implementation"] for item in profiles["profiles"].values())


# Exact pure compatibility predicate from the audited generator (source pinned in PROFILE_PATH).
def compatible_inputs(previous, current):
    """Accept the audited preflight/retry-only upgrade, not arbitrary code changes.

    Preserve accepted original-video outputs across the approved CRF 18 transport migration.
    Source, prompt, annotation, and validation changes still invalidate outputs.
    All other implementation files and every input/config hash must still match.
    """
    prior_schedulers = [{'generate.py': '36b21ca4b989bb8e222c50821f5eed54a69716e37d8eece66891997507acbfbf', 'request_runtime.py': '7205b8b4a202a2f3d035bcc589d404ce95cb3cfc9781ecbedc195cad16ce5b25'}, {
        'generate.py': '4fa8af4913131a8c6836d5e30942ff8f7ee9f7694d5c70c887b6aa1d462f5bbd',
        'request_runtime.py': '9fe12217d41ca0690610a1d29435d6e7dd414d17b58e3eb9a043a2db92d69383',
    }, {
        'generate.py': 'ab8117d3dfe41b6b47d3ab7b6645186452ea3be6d7a1e09d816bdfeca3f7a7a3',
        'request_runtime.py': 'e451a35848c4b2fcb0c7766ec0731bfc5ebf14aed1799e88ad100cfb1f6a7d3b',
    }, {
        'generate.py': 'ff05023aea6d274980de8020d725bcf130b0a816752308860d67a0ea6980bca6',
        'request_runtime.py': '7205b8b4a202a2f3d035bcc589d404ce95cb3cfc9781ecbedc195cad16ce5b25',
    }, {
        'generate.py': '39dbd2665ec0debe95574071e401b7a6d85f3924c9313f5b75d5aa2a59e10cec',
        'request_runtime.py': '7205b8b4a202a2f3d035bcc589d404ce95cb3cfc9781ecbedc195cad16ce5b25',
    }]
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    old_impl, new_impl = previous.get('implementation', {}), current.get('implementation', {})
    # These audited routing/eligibility fixes leave accepted request content and
    # validation unchanged. Keep every other implementation and input check.
    if old_impl.get('generate.py') in {
        '95bdbfb6970b8f2e21ec7b2208a97b4f206d52bf41f1af945e459c975dcf6b62',
        '70b23a2a44a0ed89e52bd79033fa2ce43f8f7cc18aa8ff1e0f50df9802b0e5fd',
    }:
        return ({k: v for k, v in old_impl.items() if k != 'generate.py'}
                == {k: v for k, v in new_impl.items() if k != 'generate.py'}
                and {k: v for k, v in previous.items() if k != 'implementation'}
                == {k: v for k, v in current.items() if k != 'implementation'})
    if not any(all(old_impl.get(name) == digest for name, digest in profile.items())
               for profile in prior_schedulers):
        return False
    excluded = {'generate.py', 'request_runtime.py', 'preflight.py', 'transport_media.py'}
    return (old_impl.keys() <= new_impl.keys()
            and {k: v for k, v in old_impl.items() if k not in excluded}
            == {k: v for k, v in new_impl.items() if k not in excluded}
            and {k: v for k, v in previous.items() if k != 'implementation'}
            == {k: v for k, v in current.items() if k not in ('implementation', 'video_transport')})


def reference_transcript(text):
    if text is not None and not isinstance(text,str):
        raise ValueError("Reference transcript must be text")
    return text if text and text.strip() else "Not provided."


def video_duration(video):
    result = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=duration","-of","json",str(video)],capture_output=True,text=True,check=True,timeout=120)
    try:
        duration = float(json.loads(result.stdout)["streams"][0]["duration"])
    except (KeyError,IndexError,TypeError,ValueError):
        raise ValueError("Prepared video has no usable duration") from None
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Prepared video has no usable duration")
    return duration


def assemble_inputs(folder, variant, category, prompt, *, modern, medical):
    """Rebuild source-only teacher assembly; never expose this payload to students."""
    meta = read_json(folder / "all_annotations/meta.json")
    video = folder / (folder.name + ".mp4")
    duration_fields = {}
    if modern and meta.get("duration_s") is None:
        meta["duration_s"] = video_duration(video)
        duration_fields = {"resolved_video_duration_s":meta["duration_s"],"duration_source":"prepared_video"}
    for key in ("duration_s","fps"):
        value = meta[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid source {key}")
    viewpoint = read_json(folder / "all_annotations/camera_motion.json")["view_type"]
    if not isinstance(viewpoint,str) or not viewpoint.strip():
        raise ValueError("Camera viewpoint is missing")
    transcript = (meta.get("scene_description") or {}).get("text")
    if modern:
        transcript = reference_transcript(transcript)
    metadata = {"modality":meta["modality"],"duration_s":meta["duration_s"],"time_of_day":meta.get("time_of_day")}
    values = {"metadata_json":json.dumps(metadata,ensure_ascii=False),"camera_viewpoint":viewpoint,"transcription":transcript}
    if variant == "native":
        counting = read_json(folder / "all_annotations/counting.json")
        counts = {key:counting[key] for key in ("person_count","subject_count","subject_type","other_people_count","robot_count")}
        if any(type(value) is not int or value < 0 for key,value in counts.items() if key!="subject_type"):
            raise ValueError("Invalid source counting annotation")
        values["counting_annotations_json"] = json.dumps({"scope":"recorded_clip_totals","scope_description":"Recorded totals for the clip, not counts at a specific timestamp or simultaneous visibility.","totals":counts},ensure_ascii=False)
    schema, prepared = prompt["response_schema"], None
    if category:
        additions, schema, prepared = medical.medical_inputs(folder,meta,viewpoint,category,variant,schema)
        if modern:
            labels = read_json(folder / "all_annotations/labels.json")
            catalog = read_yaml(medical.CATALOG)
            annotated = labels["categories"][catalog["categories"][category]["name"]]
            additions["category_transcription"] = reference_transcript(((annotated.get("narration") or {}).get("transcript") or {}).get("text"))
        values.update(additions)
    else:
        context, _ = medical.scene_medical_context(folder,meta,variant)
        if modern:
            for details in context.values():
                for cue in details["labels"]:
                    cue.pop("definition",None)
        values["medical_annotations_json"] = json.dumps(context,ensure_ascii=False)
    # Validate assembly against its exact versioned template as the generator does.
    prompt["user_prompt"].format_map(values)
    return values, schema, prepared, duration_fields
