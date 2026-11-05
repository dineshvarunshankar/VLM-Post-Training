"""Validated runtime topology for replicated evaluation; never experiment metadata."""
import json
import os
from urllib.parse import urlsplit

ENVIRONMENT_VARIABLE = 'TRIAGE_EVAL_REPLICAS_FILE'


def load_replica_plan(path, world_size=None):
    with open(path) as source:
        plan = json.load(source)
    def require(condition, message):
        if not condition:
            raise ValueError('Invalid evaluation replica manifest: ' + message)
    require(isinstance(plan, dict), 'expected object')
    for key, expected in [('schema_version', 1), ('request_batch_size', 1), ('max_num_seqs', 1)]:
        require(type(plan.get(key)) is int and plan[key] == expected, key)
    world = plan.get('training_world_size')
    require(type(world) is int and world in (8, 16), 'training_world_size')
    require(world_size is None or world == world_size, 'training world size mismatch')
    replicas = plan.get('replicas')
    require(isinstance(replicas, list) and len(replicas) == 8, 'eight replicas required')
    tp = world // 8
    urls, ports = set(), set()
    for index, replica in enumerate(replicas):
        require(isinstance(replica, dict), 'replica object')
        for key in ('replica_id', 'node_rank', 'group_port', 'sender_rank'):
            require(type(replica.get(key)) is int, key)
        node, device = divmod(index * tp, 8)
        require(replica['replica_id'] == index and replica['node_rank'] == node, 'ordered topology')
        devices = replica.get('devices')
        require(isinstance(devices, list) and all(type(d) is int for d in devices)
                and devices == list(range(device, device + tp)), 'GPU coverage')
        sender = replica['sender_rank']
        require(sender // 8 == node and sender % 8 not in devices and 0 <= sender < world,
                'sender must be local and outside receiver GPUs')
        url = replica.get('url')
        require(isinstance(url, str), 'url')
        parsed = urlsplit(url)
        require(parsed.scheme in ('http', 'https') and parsed.hostname and not parsed.username
                and not parsed.password and parsed.path in ('', '/') and not parsed.query
                and not parsed.fragment, 'HTTP endpoint URL')
        normalized = url.rstrip('/')
        require(normalized not in urls, 'duplicate URL')
        urls.add(normalized)
        port = replica['group_port']
        require(1024 <= port <= 65535 and port not in ports, 'duplicate or invalid group port')
        ports.add(port)
    return plan


def replica_plan_from_environment(world_size=None):
    path = os.environ.get(ENVIRONMENT_VARIABLE)
    return load_replica_plan(path, world_size=world_size) if path is not None else None


def replica_for_rank(plan, rank):
    if type(rank) is not int or not 0 <= rank < plan['training_world_size']:
        raise ValueError('Invalid evaluation training rank')
    return plan['replicas'][rank // (plan['training_world_size'] // 8)]
