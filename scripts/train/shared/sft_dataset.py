"""Build Arrow datasets without changing supervised targets or source records."""
import json

from datasets import Dataset


def build_sft_dataset(rows):
    # CoT provenance is a structured identity; subject checks use a digest.
    # Keep provenance in one string column and leave source rows unchanged.
    columns = dict.fromkeys(key for row in rows for key in row)
    records = []
    for row in rows:
        record = {key: row.get(key) for key in columns}
        if isinstance(record.get('target_identity'), dict):
            record['target_identity'] = json.dumps(record['target_identity'],
                sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
        records.append(record)
    return Dataset.from_list(records)
