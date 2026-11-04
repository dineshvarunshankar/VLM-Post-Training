"""Task input identity; source integrity is checked separately before reuse."""
import hashlib
import json

VERSION = 1
# These broad source records remain in generation.inputs for audit and race checks.
# Their task-relevant contents are reconstructed into task_input_sha256.
AUDIT_FIELDS = frozenset({
    'annotation_sha256', 'provenance_sha256', 'catalog_sha256',
    'assessment_prompts_sha256',
})


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def task_input_hash(prompt, values, schema, prepared):
    """Cover text, schema, exact attached frame indices and validation requirements.

    The video bytes, transport profile, settings, libraries and implementation
    are compared separately. Frame indices matter even when formatted times match.
    """
    return fingerprint({
        'task': prompt['name'], 'system': prompt['system_prompt'],
        'user': prompt['user_prompt'].format_map(values), 'schema': schema,
        'evidence_frames': (prepared or {}).get('evidence_frames', []),
        'expected_answer': (prepared or {}).get('expected_answer'),
        'requirements': (prepared or {}).get('requirements'),
        'evidence': (prepared or {}).get('evidence'),
    })


def same_task_inputs(previous, current):
    if (previous.get('task_identity_version') != VERSION
            or current.get('task_identity_version') != VERSION
            or not previous.get('task_input_sha256') or not current.get('task_input_sha256')):
        return False
    # Compare unknown fields too: newly added dependencies must fail closed.
    return ({k:v for k,v in previous.items() if k not in AUDIT_FIELDS}
            == {k:v for k,v in current.items() if k not in AUDIT_FIELDS})


LEGACY_IMPLEMENTATION = {'source_cache.py': '44554c141b47c56ccdb4a4a0c87fb722d6b772e12b98191f66821b11727be1ba', 'medical.py': '79a73834b7112ffbf069f136ebcbb6f706f7d784ac74bdfabe50501bb23be0ad', 'preflight.py': '5e0d0b8156a94e94c605122567f62fd9fc04e51b2bea9601259058bfb0f40c90', 'evidence_media.py': '3515ae5790379f73031aa6a3e1bf5458e281098349d32348f8b800099d0398e5', 'gemini_media.py': '85cd8afffff481fade48120f0c35174da02116e6a9acbfc358ab420155e85ef0', 'quality.py': '8458fbb401bebdfbd5d256fa780d73ad532b798f8a16b5e3b49c017e8049bd8d', 'generate.py': '2ceafdb30b2bdb98157ced412e35261e1a0ea159a2c8c889538e99295090b025', 'preview.py': 'f6c6f0b6cc68d4679a8f0f7ca3c91cddfc4fbeaff02671d2759fd11b0487e5fe', 'transport_media.py': 'c491890308ff0bfae94661996b4f4e9d02e8ceb99d7a8c5b07ef235f3595f0f4', 'request_runtime.py': '7205b8b4a202a2f3d035bcc589d404ce95cb3cfc9781ecbedc195cad16ce5b25'}
APPROVED_GENERATOR = 'cad3c3107ea1bee36a78dd9457e009e9a48950bb95a7158921cc7bec9679cba3'

def legacy_current_inputs(current):
    """Bridge this one audited upgrade; keep every legacy dependency check."""
    if current.get('task_identity_version') != VERSION:
        return None
    implementation = current.get('implementation', {})
    if implementation.get('generate.py') != APPROVED_GENERATOR:
        return None
    if ({k:v for k,v in implementation.items() if k not in ('generate.py','task_identity.py')}
            != {k:v for k,v in LEGACY_IMPLEMENTATION.items() if k != 'generate.py'}):
        return None
    result = {k:v for k,v in current.items()
              if k not in ('task_identity_version','task_input_sha256','video_fps_policy')}
    result['implementation'] = LEGACY_IMPLEMENTATION
    return result
