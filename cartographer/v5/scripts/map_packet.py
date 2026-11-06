"""Self-contained native map snapshots; never pair independent arrivals."""
import json,math
import numpy as np

PREFIX='map_metadata_v1:'


def metadata_from_knowledge(message):
    dims=getattr(getattr(message,'layout',None),'dim',[])
    label=dims[0].label if dims else ''
    if not isinstance(label,str) or not label.startswith(PREFIX) or len(label)>1024:
        raise ValueError('Knowledge lacks its own bounded native map metadata')
    meta=json.loads(label[len(PREFIX):])
    a=np.asarray(meta,float)
    if a.shape!=(9,) or not np.isfinite(a).all() or not .02<=a[3]<=.5:
        raise ValueError('Invalid atomic map geometry/time metadata')
    shape=a[4:7]
    if (shape!=np.floor(shape)).any() or (shape<=0).any() or (shape>2048).any():
        raise ValueError('Invalid bounded native map shape')
    count=math.prod(int(v) for v in shape)
    if count>4000000 or count!=len(message.data) or getattr(message.layout,'data_offset',0)!=0:
        raise ValueError('Knowledge payload differs from its own map geometry')
    return a.tolist()
