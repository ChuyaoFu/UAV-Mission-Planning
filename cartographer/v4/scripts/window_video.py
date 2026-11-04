"""Actual X11 window recording. It never uses flight or evaluation data."""
import os,subprocess,time,json
from pathlib import Path
class WindowVideo:
 def __init__(self,out,label):
  self.stopped=False;self.result=None;self.out=Path(out);self.label=label;self.out.mkdir(exist_ok=True);self.log=open(self.out/'screen-recording.log','w');env=dict(os.environ,DISPLAY=':0',XAUTHORITY='/tmp/visual-nav-Xauthority');listing=subprocess.check_output(['wmctrl','-l'],env=env,text=True);ids={}
  for line in listing.splitlines():
   if 'Gazebo Sim' in line:ids['gazebo']=line.split()[0]
   if 'RViz' in line:ids['slam']=line.split()[0]
  if len(ids)!=2:raise RuntimeError('Both actual flight and SLAM windows must be present before recording')
  cmd=['ffmpeg','-nostdin','-nostats','-hide_banner','-loglevel','warning','-y']
  for key in ['gazebo','slam']:cmd+=['-thread_queue_size','32','-f','x11grab','-framerate','10','-draw_mouse','0','-window_id',ids[key],'-i',':0']
  cmd+=['-filter_complex','[0:v]scale=-2:900[a];[1:v]scale=-2:900[b];[a][b]hstack=inputs=2[v]','-map','[v]','-c:v','libx264','-preset','ultrafast','-crf','24','-threads','2','-pix_fmt','yuv420p',str(self.out/'flight-and-slam.mkv')];self.p=subprocess.Popen(cmd,env=env,stdout=self.log,stderr=subprocess.STDOUT);time.sleep(1.2)
  if self.p.poll() is not None:raise RuntimeError('Screen recording failed; see screen-recording.log')
  (self.out/'screen-source.json').write_text(json.dumps({'label':label,'source':'actual Gazebo and Cartographer RViz windows','window_ids':ids,'fps':10,'timebase':'wall clock, not accelerated','truth_overlay':False},indent=2))
 def stop(self):
  if self.stopped:return self.result
  self.stopped=True
  if self.p.poll() is None:self.p.send_signal(2)
  try:self.p.wait(timeout=20)
  except subprocess.TimeoutExpired:self.p.terminate();self.p.wait(timeout=5)
  self.log.close();subprocess.run(['ffmpeg','-nostdin','-hide_banner','-y','-i',str(self.out/'flight-and-slam.mkv'),'-c','copy','-movflags','+faststart',str(self.out/'flight-and-slam.mp4')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
  info=subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration,size','-of','json',str(self.out/'flight-and-slam.mp4')],text=True);data=json.loads(info);assert float(data['format']['duration'])>1;(self.out/'video-info.json').write_text(info);self.result=data;return data
if __name__=='__main__':
 import sys
 v=WindowVideo(sys.argv[1],'recording-test');time.sleep(3);print(v.stop())
