"""Legacy validation-only F1 selection; these are not ChefOST benchmark metrics."""
import numpy as np

def label_matrix(sequence, states, aliases=None):
    columns = {state: index for index, state in enumerate(states)}
    aliases = aliases or {}
    target = np.zeros((len(sequence), len(states)), dtype=np.float32)
    valid = np.zeros(len(sequence), dtype=bool)
    for frame, labels in enumerate(sequence):
        for label in labels:
            label = label.strip().lower()
            label = aliases.get(label, label)
            if label not in columns:
                raise ValueError(f'Label {label!r} is outside the scoring vocabulary')
            target[frame, columns[label]] = 1
        valid[frame] = bool(labels)
    return target, valid

def smooth_scores(scores, width):
    values = np.asarray(scores)
    if width < 1 or width % 2 == 0:
        raise ValueError('Smoothing width must be a positive odd integer')
    if values.ndim != 2 or not len(values):
        raise ValueError('Scores must be a nonempty time-by-state matrix')
    pad = width // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode='edge')
    return np.median(np.stack([padded[start:start+len(values)] for start in range(width)]), axis=0)

def f1_counts(tp,fp,fn):
    return np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(np.asarray(tp,dtype=float)),where=(2*tp+fp+fn)>0)

def annotation_matrix(meta,annotation,states,aliases):
    if annotation['fps']!=meta['fps'] or len(annotation['states'])>meta['raw_frame_count']:
        raise ValueError('Raw annotation clock mismatch')
    truth,valid=label_matrix(annotation['states'],states,aliases)
    # Published frame arrays start at zero. Coverage beyond their end is unknown.
    missing=meta['raw_frame_count']-len(truth)
    if missing:
        truth=np.pad(truth,((0,missing),(0,0)));valid=np.pad(valid,(0,missing),constant_values=False)
    return truth,valid

class ValidationCounts:
    """Collapse reference frames into prediction bins; preserve exact raw-frame micro-F1."""
    def __init__(self,meta,annotation,states,aliases,columns=None,ignore_states=()):
        truth,valid=annotation_matrix(meta,annotation,states,aliases)
        ignore=[states.index(s) for s in ignore_states]
        if ignore:valid &= ~truth[:,ignore].any(axis=1)
        if columns is not None:truth=truth[:,columns]
        anchors=meta['sample_anchors'];bins=np.repeat(np.arange(len(anchors)),np.diff(anchors+[meta['raw_frame_count']]))
        self.positives=np.zeros((len(anchors),truth.shape[1]),dtype=np.float64)
        np.add.at(self.positives,bins,truth*valid[:,None])
        self.known=np.bincount(bins,weights=valid,minlength=len(anchors))[:,None]
        self.support=self.positives.sum(axis=0)
    def measure(self,pred):
        tp=(pred*self.positives).sum(axis=0)
        fp=(pred*self.known).sum(axis=0)-tp
        fn=self.support-tp
        t,f,n=tp.sum(),fp.sum(),fn.sum()
        return {'micro_f1':float(2*t/max(2*t+f+n,1)), 'precision':float(t/max(t+f,1)),
                'macro_f1_gt_present':float(f1_counts(tp,fp,fn)[self.support>0].mean()) if (self.support>0).any() else 0}

def choose_policy(scores,counts):
    best=None
    for smoothing in (1,3,5):
        smoothed={key:smooth_scores(value,smoothing) for key,value in scores.items()}
        for threshold in np.linspace(.05,.95,19):
            rows=[counts[key].measure(value>=threshold) for key,value in smoothed.items() if counts[key].known.sum()>0]
            if not rows:raise ValueError('No validation frames')
            result={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            result.update(threshold=float(threshold),smoothing=smoothing,selection_unit='equal weight per object/video unit')
            if best is None or (result['micro_f1'],result['precision'])>(best['micro_f1'],best['precision']):best=result
    if best is None:raise ValueError('No validation frames')
    return best
