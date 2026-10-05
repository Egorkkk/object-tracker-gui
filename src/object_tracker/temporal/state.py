"""Project cache adapter. Raw records remain the source of truth."""
from dataclasses import asdict
import hashlib
import json
from .pose_filter import TemporalParameters, filter_poses
from .confidence import quality_from_record


def defaults():
    return dict(parameters=asdict(TemporalParameters()), preview='raw', filtered_poses={}, diagnostics={}, source_signature=None)


def signature(state):
    temporal = state.get('temporal')
    if not temporal:
        return None
    payload = [state['poses'], state['source']['fps'], temporal['parameters']]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def invalidate(state):
    temporal = state.get('temporal')
    if temporal and temporal.get('source_signature') != signature(state):
        temporal.update(filtered_poses={}, diagnostics={}, source_signature=None)


def ensure(state):
    temporal = state.get('temporal')
    if not temporal:
        return False
    key = signature(state)
    if temporal.get('source_signature') == key:
        return False
    parameters = TemporalParameters(**temporal['parameters'])
    poses, diagnostics = filter_poses({int(i): r['matrix'] for i, r in state['poses'].items()}, parameters,
                                     {int(i): quality_from_record(r) for i, r in state['poses'].items()}, state['source']['fps'])
    temporal.update(filtered_poses={str(i): m.tolist() for i, m in poses.items()},
                    diagnostics={str(i): d for i, d in diagnostics.items()}, source_signature=key)
    return True


def selected_entries(project, pose_source='raw'):
    if pose_source not in ('raw', 'filtered'):
        raise ValueError('Pose source must be raw or filtered')
    if pose_source == 'raw':
        return project.state['poses']
    temporal = project.state.get('temporal')
    if not temporal or not temporal['parameters']['enabled']:
        raise ValueError('Enable temporal smoothing before using filtered poses')
    ensure(project.state)
    return {i: dict(entry, matrix=temporal['filtered_poses'].get(i, entry['matrix']))
            for i, entry in project.state['poses'].items()}
