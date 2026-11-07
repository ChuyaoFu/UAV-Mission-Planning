from pathlib import Path
import sys
base,changed=map(Path,sys.argv[1:]);checks={
'ov_msckf/src/core/VioManager.cpp':[('  if (!params.sensor_input_audit_path.empty()) input_audit = std::make_shared<SensorInputAudit>(params.sensor_input_audit_path);\n',''),('  if (input_audit) input_audit->imu(message);\n','')],
'ov_msckf/src/core/VioManager.h':[('#include "SensorInputAudit.h"\n',''),('if (input_audit) input_audit->camera(message); ',''),('if (input_audit) input_audit->motion(); ',''),('  std::shared_ptr<SensorInputAudit> input_audit;\n','')],
'ov_msckf/src/core/VioManagerHelper.cpp':[('      if (input_audit) input_audit->initialized(timestamp, state->_imu->value(), covariance);\n','')],
'ov_msckf/src/core/VioManagerOptions.h':[('  /// Optional read-only received sensor ledger. Empty default adds no logging.\n  std::string sensor_input_audit_path;\n',''),('      parser->parse_config("sensor_input_audit_path", sensor_input_audit_path, false);\n',''),('    PRINT_DEBUG("  - sensor input audit: %s\\n", sensor_input_audit_path.c_str());\n','')]}
for path,replacements in checks.items():
 s=(changed/path).read_text()
 for a,b in replacements:assert a in s,(path,a);s=s.replace(a,b,1)
 assert s==(base/path).read_text(),path
for p in base.rglob('*'):
 if p.is_file() and str(p.relative_to(base)) not in checks:assert p.read_bytes()==(changed/p.relative_to(base)).read_bytes(),str(p)
print('PASS all original source is byte-identical after removing explicit readonly hooks/options; no StateHelper/Propagator/update/feature/noise/gate/covariance edits; empty-string defaultoff')
