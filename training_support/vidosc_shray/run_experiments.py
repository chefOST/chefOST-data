"""Validation-dependent training runner; benchmark scoring is a separate central workflow."""
import subprocess,time,sys,hashlib
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training_support.vidosc_shray.bootstrap import configure
configure("broader")
import protocol
sys.path.insert(0,str(protocol.HERE))
from protocol import CONFIG,DATA,HERE,WORKSPACE,read,write
from features import feature_path

def wait_features(backbones,splits,config_path=CONFIG):
    cfg=read(config_path);paths=[feature_path(v,b) for s in splits for v in cfg[s] for b in backbones]
    while not all(p.exists() for p in paths):
        done=sum(p.exists() for p in paths)
        write(HERE/'progress.json',{'stage':'waiting for frozen features','backbones':backbones,'splits':splits,
                                  'completed_files':done,'expected_files':len(paths),'test_accuracy_available':False})
        print('Waiting for',backbones,splits,done,'/',len(paths),flush=True)
        time.sleep(30)

def run(arguments):
    subprocess.run([sys.executable,*map(str,arguments)],cwd=WORKSPACE,check=True)

def main():
    for mode,heldout,config_path in [('clip',False,CONFIG),('clip',True,CONFIG),
                                    ('clip',False,HERE/'protocol-manual.json'),('clip_slowfast',False,CONFIG)]:
        needed=['clip','slowfast'] if mode=='clip_slowfast' else ['clip']
        wait_features(needed,['train','validation'],config_path)
        name=mode+read(config_path).get('run_suffix','')+('-heldout-peeled' if heldout else '')
        out=HERE/'results'/name
        if (out/'model.pt').exists() and (out/'selection.json').exists():continue
        if out.exists():raise RuntimeError(f'Incomplete training directory requires inspection: {out}')
        write(HERE/'progress.json',{'stage':'training','run':name,'test_accuracy_available':False})
        run([Path(__file__).with_name('fit.py'),'--mode',mode,'--protocol',config_path,*(['--heldout'] if heldout else [])])

if __name__=='__main__':main()
