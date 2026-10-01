"""Evaluator-version trust requires measured agreement, not a self-declared judge label."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import math
from inference_control.util import digest


class EvaluatorTrustStore:
    def __init__(self, state=None):
        self.state = state
        self.records = state.trust if state else {}

    def calibrate(self, version: str, pairs: list[tuple[float,float]], *,
                  dataset_hash: str, human_version: str, minimum_agreement: float = .8,
                  min_samples: int = 30, ttl_seconds: int = 86400, at=None):
        at = at or datetime.now(timezone.utc)
        if not 0<=minimum_agreement<=1 or min_samples<1 or ttl_seconds<1:
            raise ValueError("invalid trust calibration bounds")
        if not pairs or not dataset_hash or not human_version:
            raise ValueError("independent judge/human agreement evidence required")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x,y in pairs):
            raise ValueError("agreement scores must be in [0,1]")
        agreement = sum(1-abs(x-y) for x,y in pairs)/len(pairs)
        lower = max(0, agreement-math.sqrt(math.log(20)/(2*len(pairs))))
        record = {"evaluator_version":version,"dataset_hash":dataset_hash,"human_version":human_version,
                  "sample_size":len(pairs),"agreement":agreement,"agreement_lower":lower,
                  "minimum_agreement":minimum_agreement,"trusted":len(pairs)>=min_samples and lower>=minimum_agreement,
                  "valid_until":(at+timedelta(seconds=ttl_seconds)).isoformat(),"calibrated_at":at.isoformat()}
        if self.state: self.state.write("evaluator_trust",record,key=f"evaluator:{digest(record)}")
        self.records[version] = record
        return record

    def trusted(self, version: str, at=None):
        at = at or datetime.now(timezone.utc)
        row = self.records.get(version)
        return bool(row and row["trusted"] and at < datetime.fromisoformat(row["valid_until"]))

    def invalidate(self, version: str, reason: str):
        if version not in self.records: return
        row = {**self.records[version],"trusted":False,"invalidated_reason":reason}
        if self.state: self.state.write("evaluator_trust",row,key=f"evaluator:{digest(row)}")
        self.records[version] = row
