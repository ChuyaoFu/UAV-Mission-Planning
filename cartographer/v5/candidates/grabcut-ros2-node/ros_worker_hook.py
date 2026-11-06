"""Nonpublishing reviewable ROS hook. No node/subscription/topic side effects.
Root wires exactly ONE /nav/depth_evidence publisher, replacing the old inference
backend while preserving its independent authoritative raw10Hz producer.
"""
import copy

def completed_to_message(adapter,bridge,result):
 if result is None or result.get('packet') is None:return None
 if result.get('original_header') is None:raise ValueError('original raw header required; never stamp with now')
 evidence=adapter.evidence_32fc2(result)
 msg=bridge.cv2_to_imgmsg(evidence,encoding='32FC2');msg.header=copy.deepcopy(result['original_header'])
 # Caller publishes once through the existing sole evidence owner and retains
 # consumer exact pose/header sync, raw dedup/freshness and whole-body checks.
 return msg
