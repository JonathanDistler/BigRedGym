"""Recoverability-first reverse curriculum admission and advancement rules.

States are exact integrated snapshots. Morphology-specific states must be
validated independently; physical states are never copied between models.
"""
from dataclasses import dataclass, field
from experiments.recoverability.run import wilson


@dataclass
class ReverseCurriculum:
    horizon: float = .04
    maximum_horizon: float = .8
    increment: float = .04
    admitted: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def admit(self, key, snapshot, recovery, scale=1.0, predecessor=None):
        if not (recovery['success'] and recovery['hold']>=5-1e-8 and
                recovery['max_speed']<=1.001 and recovery['max_torque']<=1.00001 and
                recovery['max_joint']<.001 and not recovery['collision'] and
                recovery['first_failure'] is None and recovery['loaded_drift']<=.04):
            return False
        if predecessor is not None and predecessor not in self.admitted:
            raise ValueError('Reverse expansion requires a validated parent')
        if predecessor is not None and self.admitted[predecessor]['scale'] != scale:
            raise ValueError('A reverse edge cannot cross physical models')
        self.admitted[key]=dict(snapshot=snapshot,scale=scale,parent=predecessor,horizon=self.horizon)
        return True

    def admit_transition(self, key, start, recovery, transition, parent, scale=1.0):
        """Expand toward an earlier integrated start after validating its edge.

        `transition` is measured at every MuJoCo step of the motor trajectory;
        `recovery` continues its exact endpoint with the frozen controller.
        """
        if (transition['first_failure'] is not None or transition['collision'] or
                transition['loaded_slip'] > .02 or transition['loaded_drift'] > .04 or
                transition['max_speed'] > 1.001 or transition['max_torque'] > 1.00001 or
                transition['max_joint'] >= .001):
            return False
        return self.admit(key, start, recovery, scale, predecessor=parent)

    def advance(self, evaluations):
        # Held-out original-morphology evaluations only. Never pooled across scales.
        native=[r for r in evaluations if r['scale']==1.0]
        n=len(native); k=sum(r['success'] for r in native)
        safe=all(not r['collision'] and r['loaded_slip']<=.02 and
                 r['loaded_drift']<=.04 and r['max_speed']<=1.001 and
                 r['max_torque']<=1.00001 and r['max_joint']<.001 and
                 (not r['success'] or r['hold']>=5-1e-8) for r in native)
        passed=n>=20 and k/n>=.8 and wilson(k,n)[0]>=.6 and safe
        self.history.append(dict(n=n,k=k,passed=passed,horizon=self.horizon))
        if passed:self.horizon=min(self.maximum_horizon,self.horizon+self.increment)
        return passed

    def sample(self, rng, scale=1.0):
        keys=[k for k,v in self.admitted.items() if v['scale']==scale]
        if not keys:raise ValueError('No independently validated states for morphology')
        return self.admitted[keys[int(rng.integers(len(keys)))]]['snapshot']
