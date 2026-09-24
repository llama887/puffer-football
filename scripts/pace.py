import re, sys
ep=re.compile(r'Epoch\s+(\d+)\b'); up=re.compile(r'Uptime\s+([0-9hms ]+?)\s{2,}')
sps=re.compile(r'SPS\s+([0-9.]+)(K?)')
steps_re=re.compile(r'Steps\s+([0-9.]+)([MK]?)')
text=open(sys.argv[1],'rb').read().decode('utf8','ignore').replace('\x00','')
# The dashboard renders the CPU number on the panel line AFTER the label,
# so take the first percentage from the following line, not this one.
lines=text.splitlines()
cpus=[]
for i, line in enumerate(lines[:-1]):
    if 'CPU:' in line:
        m=re.search(r'([0-9.]+)%', lines[i+1])
        if m: cpus.append(float(m.group(1)))
pairs=[]; cur=None; spss=[]
for line in text.splitlines():
    m=sps.search(line)
    if m: spss.append(float(m.group(1))*(1000 if m.group(2)=='K' else 1))
    m=ep.search(line)
    if m: cur=int(m.group(1)); continue
    m=up.search(line)
    if m and cur is not None:
        s=sum(int(v)*{'h':3600,'m':60,'s':1}[u] for v,u in re.findall(r'(\d+)([hms])',m.group(1)))
        pairs.append((cur,s)); cur=None
if not pairs: sys.exit('no dashboard panels yet')
d={b:t2-t1 for (a,t1),(b,t2) in zip(pairs,pairs[1:]) if b==a+1}
fast=[v for v in d.values() if v<=30]; slow=[v for v in d.values() if v>30]
ep_n, up_s = pairs[-1]
steps=steps_re.findall(text)
if steps:
    v,u=steps[-1]
    total=float(v)*{'M':1e6,'K':1e3,'':1}[u]
    # steps divided by uptime is the number that decides when a run
    # finishes; the dashboard's own SPS is a recent window and reads
    # high because it excludes the promotion checks.
    print(f'STEPS/SEC     : {total/max(1,up_s):7.0f} true average   ({total/1e6:.1f}M steps)')
print(f'epoch {ep_n}   uptime {up_s/3600:.2f}h   overall {up_s/max(1,ep_n):.2f} s/epoch')
if fast: print(f'normal epochs : {sum(fast)/len(fast):5.2f} s/epoch   (n={len(fast)})   [old run: 4.84]')
if slow: print(f'slow epochs   : {sum(slow)/len(slow):5.0f} s each      (n={len(slow)})   [old run: 112]')
if cpus: print(f'CPU           : {sum(cpus[-20:])/len(cpus[-20:]):5.1f}% recent            [old run: 17.7]')
if spss: print(f'SPS           : {sum(spss[-20:])/len(spss[-20:]):5.0f} recent             [old run: 4600]')
if slow and fast:
    print(f'eval share    : {100*sum(slow)/(sum(slow)+sum(fast)):5.1f}%                   [old run: 53.6]')
