#!/usr/bin/env python3
"""Synthetic floor appearance for regression tests; never a venue replica."""
from pathlib import Path
import random
import struct
import zlib
root=Path(__file__).resolve().parents[1]
assets=root/'sim/assets';assets.mkdir(parents=True,exist_ok=True)
rng=random.Random(2468);w=1024
tiles=[[rng.randrange(130,211) for _ in range(128)] for _ in range(128)]
pixels=b''.join(b'\x00'+bytes(tiles[y//8][x//8] for x in range(w) for _ in range(3)) for y in range(w))
def chunk(t,d):return struct.pack('!I',len(d))+t+d+struct.pack('!I',zlib.crc32(t+d)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',w,w,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')
(assets/'regression-floor.png').write_bytes(png)
(assets/'regression-floor.sdf').write_text('''<sdf version="1.9"><model name="regression_floor_texture"><static>true</static><link name="appearance"><visual name="synthetic_floor"><pose>0 0 .0001 0 0 0</pose><geometry><plane><normal>0 0 1</normal><size>6.096 6.096</size></plane></geometry><material><ambient>.7 .7 .7 1</ambient><diffuse>.7 .7 .7 1</diffuse><pbr><metal><albedo_map>file:///work/sim/assets/regression-floor.png</albedo_map><roughness>.95</roughness><metalness>0</metalness></metal></pbr></material></visual></link></model></sdf>''')
print('Generated synthetic regression texture; original course unchanged.')
