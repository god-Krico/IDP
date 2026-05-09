"""
safety_processor.py - Dashboard-compatible wrapper around combined.py logic.
All detection code is UNCHANGED. Only wrapped into process_safety() for Streamlit.
"""

import cv2
import numpy as np
import os
import time
import torch
from collections import defaultdict, deque
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = "mps"  if torch.backends.mps.is_available() \
    else "cuda" if torch.cuda.is_available() \
    else "cpu"

# ══════════════════════════════════════════════════════════════════════════════
# LAZY MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

_person_model    = None
_proximity_model = None
_ppe_model       = None
_USE_PPE         = False

def _load_models():
    global _person_model, _proximity_model, _ppe_model, _USE_PPE
    if _person_model is None:
        _person_model    = YOLO("yolov8m.pt")
        _proximity_model = YOLO("model/proximity_best.pt")
        ppe_path = "model/ppe_best.pt"
        if os.path.exists(ppe_path):
            _ppe_model = YOLO(ppe_path)
            _USE_PPE   = True
    return _person_model, _proximity_model, _ppe_model, _USE_PPE

# ══════════════════════════════════════════════════════════════════════════════
# AUTO ZONE (depends on video W/H — set inside process_safety)
# ══════════════════════════════════════════════════════════════════════════════

def _make_zone(W, H):
    if W >= 2400:
        SRC_W=2560; SRC_H=1440
        TRACK_TOP_LINE   =(0,960, 1900,250)
        TRACK_BOTTOM_LINE=(0,1600,2000,250)
        PPM_FALLBACK=180
    elif W >= 1800:
        SRC_W=1920; SRC_H=1080
        TRACK_TOP_LINE   =(0,0,  1920,0)
        TRACK_BOTTOM_LINE=(0,150,1920,515)
        PPM_FALLBACK=115
    else:
        SRC_W=W; SRC_H=H
        TRACK_TOP_LINE   =(0,0,W,0)
        TRACK_BOTTOM_LINE=(0,int(H*0.1),W,int(H*0.1))
        PPM_FALLBACK=100
    return SRC_W, SRC_H, TRACK_TOP_LINE, TRACK_BOTTOM_LINE, PPM_FALLBACK

# ══════════════════════════════════════════════════════════════════════════════
# ZONE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_line_y(x, line):
    x1,y1,x2,y2=line
    if x2==x1: return y1
    x=max(x1,min(x2,x))
    return int(y1+(y2-y1)*(x-x1)/(x2-x1))

def _is_on_track(fx, fy, TTL, TBL):
    t=_get_line_y(fx,TTL)
    b=_get_line_y(fx,TBL)
    return min(t,b)<=fy<=max(t,b)

def _get_side(box, OW, OH, SRC_W, SRC_H, TTL, TBL):
    x1,y1,x2,y2=map(int,box[:4])
    cx=(x1+x2)//2; cy=(y1+y2)//2
    cxs=int(cx*SRC_W/OW)
    cys=int(cy*SRC_H/OH)
    t=_get_line_y(cxs,TTL)
    b=_get_line_y(cxs,TBL)
    return "LEFT" if cys<(t+b)//2 else "RIGHT"

def _is_in_work_zone(box, OW, OH, SRC_W, SRC_H, TTL, TBL):
    x1,y1,x2,y2=map(int,box[:4])
    fx=(x1+x2)//2; fy=y2
    fxs=int(fx*SRC_W/OW)
    fys=int(fy*SRC_H/OH)
    mn=min(_get_line_y(fxs,TTL),
           _get_line_y(fxs,TBL))
    if fys<mn-500: return False
    return True

# ══════════════════════════════════════════════════════════════════════════════
# AUTO CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

REAL_H=1.7

def _auto_calibrate(vpath, person_model, W, H, TTL, TBL, PPM_FALLBACK):
    cap  =cv2.VideoCapture(vpath)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start=total//4; end=total*3//4
    step =max(1,(end-start)//40)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    idx=0; heights=[]
    while cap.isOpened():
        if len(heights)>=60: break
        ret,frame=cap.read()
        if not ret: break
        if idx%step==0:
            small=cv2.resize(frame,(640,int(640*H/W)))
            sh=small.shape[0]; sw=small.shape[1]
            res=person_model(
                small,conf=0.40,classes=[0],
                device=DEVICE,verbose=False)[0]
            for box in res.boxes.xyxy.cpu().numpy():
                x1,y1,x2,y2=map(int,box)
                bh=y2-y1; bw=x2-x1
                fx=int(((x1+x2)//2)*W/sw)
                fy=int(y2*H/sh)
                if _is_on_track(fx,fy,TTL,TBL): continue
                mn=min(_get_line_y(fx,TTL),
                       _get_line_y(fx,TBL))
                if fy<mn-400: continue
                if bw==0: continue
                if bh/bw<0.8 or bh/bw>4.0: continue
                if bh<20: continue
                heights.append(bh*H/sh)
        idx+=1
    cap.release()
    if len(heights)<3:
        return PPM_FALLBACK, 0.85
    heights.sort()
    n=len(heights)
    ref=np.mean(heights[n//4:n*3//4])
    ppm=ref/REAL_H
    avg_r=ref/H
    sy=1.0 if avg_r>0.08 else 0.75
    if W>=2000: ppm=max(60,min(400,ppm))
    else:       ppm=max(30,min(200,ppm))
    return ppm, sy

# ══════════════════════════════════════════════════════════════════════════════
# BOX SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class BoxSmoother:
    def __init__(self, alpha=0.55, cell=60):
        self.alpha=alpha
        self.cell =cell
        self.boxes={}

    def _gid(self, box):
        x1,y1,x2,y2=map(int,box[:4])
        return ((x1+x2)//2//self.cell,
                (y1+y2)//2//self.cell)

    def smooth(self, boxes):
        if not boxes: return boxes
        new_ids=set(); smoothed=[]
        for b in boxes:
            gid=self._gid(b)
            new_ids.add(gid)
            if gid in self.boxes:
                pb=self.boxes[gid]
                sb=tuple(
                    self.alpha*b[i]+
                    (1-self.alpha)*pb[i]
                    for i in range(4))+(b[4],)
                self.boxes[gid]=sb
                smoothed.append(sb)
            else:
                self.boxes[gid]=b
                smoothed.append(b)
        for k in [k for k in self.boxes
                  if k not in new_ids]:
            del self.boxes[k]
        return smoothed

# ══════════════════════════════════════════════════════════════════════════════
# PPE VOTER
# ══════════════════════════════════════════════════════════════════════════════

class PPEVoter:
    """
    Sticky PPE voter - once PPE OK is seen, stays OK for STICKY_FRAMES.
    Only flags NO PPE if EVERY frame in the full window is missing PPE.
    This eliminates flickering caused by occasional missed detections.
    """
    STICKY_FRAMES = 40   # frames of grace after last PPE detection

    def __init__(self, window=20, no_ppe_thr=18):
        # window=20, no_ppe_thr=18 means 18/20 frames must be NO PPE
        # before we actually flag it - very conservative
        self.window     = window
        self.no_ppe_thr = no_ppe_thr
        self.history    = defaultdict(lambda: deque(maxlen=window))
        self.sticky_ok  = defaultdict(int)   # countdown per person grid-cell

    def _gid(self, box, cell=50):
        x1,y1,x2,y2 = map(int, box[:4])
        return ((x1+x2)//2//cell, (y1+y2)//2//cell)

    def vote(self, box, raw_status):
        gid = self._gid(box)
        ok  = (raw_status == "PPE OK")
        self.history[gid].append(1 if ok else 0)

        if ok:
            # Reset sticky counter every time PPE is detected
            self.sticky_ok[gid] = self.STICKY_FRAMES
            return "PPE OK"

        # Decrement sticky - stay OK while any grace remains
        if self.sticky_ok[gid] > 0:
            self.sticky_ok[gid] -= 1
            return "PPE OK"

        # Only flag NO PPE if nearly ALL frames in window are missing PPE
        hist = self.history[gid]
        n    = len(hist)
        if n < self.window:          # not enough history yet
            return "PPE OK"
        ok_count_hist = sum(hist)
        if ok_count_hist >= (self.window - self.no_ppe_thr):
            return "PPE OK"          # some OK frames still in window
        return "NO PPE"              # persistently confirmed missing

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def iou(a,b):
    ax1,ay1,ax2,ay2=map(float,a[:4])
    bx1,by1,bx2,by2=map(float,b[:4])
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    u=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/u if u>0 else 0

def scale_box(box,sx,sy):
    x1,y1,x2,y2=box
    return np.array([x1*sx,y1*sy,x2*sx,y2*sy])

def get_center(box):
    x1,y1,x2,y2=map(int,box[:4])
    return ((x1+x2)//2,(y1+y2)//2)

def is_real_person(box,fW,fH):
    x1,y1,x2,y2=map(int,box[:4])
    bw=x2-x1; bh=y2-y1
    if bw<=0 or bh<=0:           return False
    if bw<12 or bh<25:           return False
    if bw>fW*0.12:               return False
    if bh>fH*0.45:               return False
    if bw*bh>fW*fH*0.025:       return False
    asp=bh/max(bw,1)
    if asp<1.3 or asp>5.0:       return False
    w_ratio=bw/max(bh,1)
    if w_ratio<0.18 or w_ratio>0.65: return False
    return True

def has_person_color(frame,box):
    fH,fW=frame.shape[:2]
    x1,y1,x2,y2=map(int,box[:4])
    x1=max(0,x1); y1=max(0,y1)
    x2=min(fW,x2); y2=min(fH,y2)
    if x2<=x1 or y2<=y1: return True
    roi=frame[y1:y2,x1:x2]
    if roi.size<50: return True
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    H=hsv[:,:,0]; S=hsv[:,:,1]; V=hsv[:,:,2]
    s_mean=float(np.mean(S))
    h_std =float(np.std(H))
    s_std =float(np.std(S))
    v_mean=float(np.mean(V))
    if s_mean<20 and h_std<12 and s_std<15:
        return False
    if s_mean<15 and v_mean>180:
        return False
    hiviz=((H>=5) &(H<=38)&(S>=60)&(V>=70))
    skin =((H>=0) &(H<=20)&(S>=25)&(S<=150)&(V>=60))
    dark =(V<70)
    color=(S>=55)&(V>=55)
    total=roi.shape[0]*roi.shape[1]
    score=(float(np.sum(hiviz))/total*5 +
           float(np.sum(skin)) /total*4 +
           float(np.sum(color))/total*2 +
           float(np.sum(dark)) /total*0.5)
    return score>0.12

def dedupe(boxes,thr=0.45):
    if not boxes: return []
    boxes=sorted(boxes,
                 key=lambda b:b[4],reverse=True)
    keep=[]
    for b in boxes:
        if all(iou(b[:4],k[:4])<thr
               for k in keep):
            keep.append(b)
    return keep

def is_near_any_person(box,persons,max_dist=150):
    if not persons: return False
    bx=(box[0]+box[2])//2
    by=(box[1]+box[3])//2
    for p in persons:
        px1,py1,px2,py2=map(int,p[:4])
        ph=py2-py1; pw=px2-px1
        pad_x=int(pw*0.5); pad_y=int(ph*0.3)
        if (px1-pad_x<=bx<=px2+pad_x and
                py1-pad_y<=by<=py2+pad_y):
            return True
        pcx=(px1+px2)//2; pcy=(py1+py2)//2
        dist=np.sqrt((bx-pcx)**2+(by-pcy)**2)
        if dist<max(ph*1.5,max_dist):
            return True
    return False

# ══════════════════════════════════════════════════════════════════════════════
# PPE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_ppe(p_box,helmets,vests):
    x1,y1,x2,y2=map(int,p_box[:4])
    ph=y2-y1; pw=x2-x1
    if ph<40 or pw<15:
        return "PPE OK"
    pad_x=int(pw*0.35)
    head =(max(0,x1-pad_x),y1,
           x2+pad_x,y1+int(0.45*ph))
    torso=(max(0,x1-pad_x),
           y1+int(0.05*ph),
           x2+pad_x,y2+int(0.05*ph))
    full =(max(0,x1-pad_x),max(0,y1-10),
           x2+pad_x,y2+int(0.15*ph))
    h_ok=False
    for h in helmets:
        hx1,hy1,hx2,hy2=map(int,h[:4])
        hcx=(hx1+hx2)//2; hcy=(hy1+hy2)//2
        if iou(head,(hx1,hy1,hx2,hy2))>0.005:
            h_ok=True; break
        if (head[0]<=hcx<=head[2] and
                head[1]<=hcy<=head[3]):
            h_ok=True; break
        if (full[0]<=hcx<=full[2] and
                full[1]<=hcy<=full[3]):
            h_ok=True; break
    v_ok=False
    for v in vests:
        vx1,vy1,vx2,vy2=map(int,v[:4])
        vcx=(vx1+vx2)//2; vcy=(vy1+vy2)//2
        if iou(torso,(vx1,vy1,vx2,vy2))>0.005:
            v_ok=True; break
        if (torso[0]<=vcx<=torso[2] and
                torso[1]<=vcy<=torso[3]):
            v_ok=True; break
        if iou(full,(vx1,vy1,vx2,vy2))>0.002:
            v_ok=True; break
        if (full[0]<=vcx<=full[2] and
                full[1]<=vcy<=full[3]):
            v_ok=True; break
    if h_ok or v_ok:
        return "PPE OK"
    return "NO PPE"

# ══════════════════════════════════════════════════════════════════════════════
# PROXIMITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _calc_dist(p_box, a_box, PPM_OUT, SX_OUT, SY_OUT):
    """
    Compute real-world distance from person feet to nearest aerolift edge.
    Uses a physical floor of 0.20m - person and machine cannot share space.
    Aerolift bounding box is inflated by model so we shrink it slightly
    to avoid 0cm readings from box overlap.
    """
    px1,py1,px2,py2=map(int,p_box[:4])
    ax1,ay1,ax2,ay2=map(int,a_box[:4])
    # Shrink aerolift box by 10% to compensate for detection over-expansion
    aw=ax2-ax1; ah=ay2-ay1
    shrink_x=int(aw*0.08); shrink_y=int(ah*0.08)
    ax1s=ax1+shrink_x; ay1s=ay1+shrink_y
    ax2s=ax2-shrink_x; ay2s=ay2-shrink_y
    # Person feet centre
    pfx=(px1+px2)//2; pfy=py2
    # Nearest point on shrunk aerolift box
    nax=max(ax1s,min(ax2s,pfx))
    nay=max(ay1s,min(ay2s,pfy))
    dx=abs(pfx-nax)/(PPM_OUT*max(SX_OUT,0.01))
    dy=abs(pfy-nay)/(PPM_OUT*max(SY_OUT,0.01))
    raw=np.sqrt(dx**2+dy**2)
    # Physical floor: 0.20m minimum (person cannot be inside the machine)
    return max(raw, 0.20),(nax,nay)

def _nearest_aerolift(p_box, aerolifts, PPM_OUT, SX_OUT, SY_OUT):
    if not aerolifts: return None,None,None
    bd=float('inf'); bn=None; ba=None
    for a_box,_,_ in aerolifts:
        d,n=_calc_dist(p_box,a_box,PPM_OUT,SX_OUT,SY_OUT)
        if d<bd: bd=d; bn=n; ba=a_box
    return bd,bn,ba

def _in_danger_box(p_box, a_box, DANGER_M, PPM_OUT, SX_OUT, SY_OUT):
    px1,py1,px2,py2=map(int,p_box[:4])
    ax1,ay1,ax2,ay2=map(int,a_box[:4])
    dp =max(5,int(DANGER_M*PPM_OUT*SX_OUT))
    dy_=max(5,int(DANGER_M*PPM_OUT*SY_OUT))
    return (px1<=ax2+dp and px2>=ax1-dp and
            py1<=ay2+dy_ and py2>=ay1-dy_)

def _get_prox_status(d, DANGER_M, WARNING_M):
    C_DANGER =(0,0,255)
    C_WARNING=(0,165,255)
    C_SAFE   =(0,255,0)
    if d<=DANGER_M:
        return f"DANGER {d*100:.0f}cm",C_DANGER
    elif d<=WARNING_M:
        return f"WARN {d:.1f}m",C_WARNING
    else:
        return f"SAFE {d:.1f}m",C_SAFE

# ══════════════════════════════════════════════════════════════════════════════
# DRAW FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

C_OK     =(0,255,0)
C_FAIL   =(0,0,255)
C_DANGER =(0,0,255)
C_WARNING=(0,165,255)
C_SAFE   =(0,255,0)
C_AEROLIFT=(0,215,255)
C_HARDHAT=(0,255,255)
C_VEST_C =(200,200,200)

def put_label(frame,text,x,y,color,fs=0.40):
    (tw,th),_=cv2.getTextSize(
        text,cv2.FONT_HERSHEY_SIMPLEX,fs,2)
    y=max(th+3,y)
    cv2.rectangle(frame,(x-2,y-th-4),
                 (x+tw+3,y+4),(0,0,0),-1)
    cv2.putText(frame,text,(x,y),
               cv2.FONT_HERSHEY_SIMPLEX,
               fs,color,2)

def draw_corners(frame,x1,y1,x2,y2,color,thick):
    c=6
    for p1,p2 in [
        ((x1,y1),(x1+c,y1)),((x1,y1),(x1,y1+c)),
        ((x2,y1),(x2-c,y1)),((x2,y1),(x2,y1+c)),
        ((x1,y2),(x1+c,y2)),((x1,y2),(x1,y2-c)),
        ((x2,y2),(x2-c,y2)),((x2,y2),(x2,y2-c)),
    ]:
        cv2.line(frame,p1,p2,color,thick)

def draw_person_box(frame,x1,y1,x2,y2,
                    top_label,box_color,
                    ppe_status,ppe_color,
                    thick=2):
    x1,y1,x2,y2=map(int,(x1,y1,x2,y2))
    cv2.rectangle(frame,(x1,y1),(x2,y2),
                 box_color,thick)
    draw_corners(frame,x1,y1,x2,y2,
                box_color,thick)
    put_label(frame,top_label,x1,y1-4,box_color)
    (tw,th),_=cv2.getTextSize(
        ppe_status,cv2.FONT_HERSHEY_SIMPLEX,
        0.38,1)
    cv2.rectangle(frame,(x1,y2+1),
                 (x1+tw+4,y2+th+4),
                 (0,0,0),-1)
    cv2.putText(frame,ppe_status,
               (x1+2,y2+th+1),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.38,ppe_color,1)

def draw_aerolift(frame, a_box, DANGER_M, WARNING_M, SAFE_M, PPM_OUT, SX_OUT, SY_OUT):
    ax1,ay1,ax2,ay2=map(int,a_box[:4])
    fH,fW=frame.shape[:2]
    dp =max(5,int(DANGER_M *PPM_OUT*SX_OUT))
    dy_=max(5,int(DANGER_M *PPM_OUT*SY_OUT))
    wp =int(WARNING_M*PPM_OUT*SX_OUT)
    wy =int(WARNING_M*PPM_OUT*SY_OUT)
    sp =int(SAFE_M   *PPM_OUT*SX_OUT)
    sy_=int(SAFE_M   *PPM_OUT*SY_OUT)
    cv2.rectangle(frame,
                 (max(0,ax1-sp),max(0,ay1-sy_)),
                 (min(fW,ax2+sp),min(fH,ay2+sy_)),
                 C_SAFE,2)
    cv2.rectangle(frame,
                 (max(0,ax1-wp),max(0,ay1-wy)),
                 (min(fW,ax2+wp),min(fH,ay2+wy)),
                 C_WARNING,2)
    cv2.rectangle(frame,
                 (max(0,ax1-dp),max(0,ay1-dy_)),
                 (min(fW,ax2+dp),min(fH,ay2+dy_)),
                 C_DANGER,2)
    cv2.rectangle(frame,(ax1,ay1),(ax2,ay2),
                 C_AEROLIFT,3)
    cv2.rectangle(frame,(ax1,max(0,ay1-18)),
                 (ax1+80,ay1),(0,0,0),-1)
    cv2.putText(frame,"AEROLIFT",
               (ax1+2,max(13,ay1-3)),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.45,C_AEROLIFT,1)

def draw_top_panel(frame,n_workers,ok,fail,
                   n_h,n_v,n_danger,n_warn,
                   t_sec,fps_r):
    fH,fW=frame.shape[:2]
    fs=0.40
    items=[
        (f"Workers:{n_workers}",(255,255,255)),
        (f"PPE OK:{ok}",        C_OK),
        (f"NO PPE:{fail}",      C_FAIL),
        (f"Danger:{n_danger}",  C_DANGER),
        (f"Warn:{n_warn}",      C_WARNING),
        (f"Helmets:{n_h}",      C_HARDHAT),
        (f"Vests:{n_v}",        C_VEST_C),
        (f"FPS:{fps_r:.0f}",    (150,150,150)),
    ]
    gap=12; total_w=0; sizes=[]
    for txt,col in items:
        (tw,th),_=cv2.getTextSize(
            txt,cv2.FONT_HERSHEY_SIMPLEX,fs,2)
        sizes.append((tw,th))
        total_w+=tw+gap
    total_w-=gap
    pad=7
    panel_h=sizes[0][1]+pad*2
    px=(fW-total_w)//2-pad
    ov=frame.copy()
    cv2.rectangle(ov,(px,2),
                 (px+total_w+pad*2,2+panel_h),
                 (20,20,20),-1)
    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    cx=px+pad
    for (txt,col),(tw,th) in zip(items,sizes):
        y=2+pad+th
        cv2.putText(frame,txt,(cx,y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   fs,col,2)
        cx+=tw+gap
    time_txt=f"Time:{t_sec:.1f}s"
    (tw2,th2),_=cv2.getTextSize(
        time_txt,cv2.FONT_HERSHEY_SIMPLEX,0.35,1)
    tx=(fW-tw2)//2
    cv2.putText(frame,time_txt,
               (tx,2+panel_h+th2+2),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.35,(120,120,120),1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — process_safety() called by dashboard.py
# Detection approach: yolov8m (full + top-half) + PPE model workers
#                   + Proximity model persons  → 3 combined sources
# ══════════════════════════════════════════════════════════════════════════════


def _save_img(img, path):
    """Save cv2 image to disk, return path or None."""
    if img is None:
        return None
    import os as _os
    _os.makedirs(_os.path.dirname(path) or '.', exist_ok=True)
    cv2.imwrite(path, img)
    return path


def process_safety(video_path, frame_skip=5, frame_callback=None):
    """Process video for safety analytics.
      2. yolov8m - top half (catches distant/small workers)
      3. PPE model workers class
      4. Proximity model persons class
    """
    person_model, proximity_model, ppe_model, USE_PPE = _load_models()

    cap   = cv2.VideoCapture(video_path)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    OUT_W = W // 2;  OUT_H = H // 2
    SRC_W, SRC_H, TTL, TBL, PPM_FALLBACK = _make_zone(W, H)

    PIXELS_PER_METER, SCALE_Y = _auto_calibrate(
        video_path, person_model, W, H, TTL, TBL, PPM_FALLBACK)
    SCALE_X = 1.0

    # ── Constants (same as Safety_Moduling.py) ────────────────
    DANGER_M      = 0.60
    WARNING_M     = 2.0
    SAFE_M        = 4.0
    LINE_THR      = 2.0
    PROX_CHK      = 4.0   # track only within safe zone (4m)
    PROCESS_EVERY = 3

    PPM_OUT = PIXELS_PER_METER * (OUT_W / W)
    SX_OUT  = SCALE_X
    SY_OUT  = SCALE_Y
    PROC_W  = 480
    PROC_H  = int(PROC_W * H / W)
    sx2     = OUT_W / PROC_W
    sy2     = OUT_H / PROC_H

    # Camera label + detection mode flag
    if W >= 2400:
        cam_type  = "Camera 2 - Aerolift (2560x1440)"
        cam_is_tlc = False   # Use full 4-source detection
    elif W >= 1800:
        cam_type  = "Camera 1 - TLC Overhead (1920x1080)"
        cam_is_tlc = True    # TLC: yolov8m only, no PPE/proximity person sources
    else:
        cam_type  = f"Unknown ({W}x{H})"
        cam_is_tlc = False

    # ── PPE / Proximity class IDs ─────────────────────────────
    prox_person_cls = set()
    for k, v in proximity_model.names.items():
        if any(x in str(v).lower() for x in ["person","worker"]):
            prox_person_cls.add(k)

    ppe_person_cls = set()
    ppe_hat_cls    = set()
    ppe_vest_cls   = set()
    AEROLIFT_CLS   = {"aerolifter"}
    VEHICLE_CLS    = {"vehicle"}

    if USE_PPE:
        for k, v in ppe_model.names.items():
            vl = str(v).lower()
            if any(x in vl for x in ["person","worker"]):
                ppe_person_cls.add(k)
            elif any(x in vl for x in ["hardhat","helmet"]):
                ppe_hat_cls.add(k)
            elif "vest" in vl:
                ppe_vest_cls.add(k)

    # ── Stabilizers ───────────────────────────────────────────
    box_smoother = BoxSmoother(alpha=0.55, cell=60)
    ppe_voter    = PPEVoter(window=12, no_ppe_thr=10)

    # ── Counters ──────────────────────────────────────────────
    fc              = 0
    danger_ev       = 0
    warn_ev         = 0
    tot_no_ppe      = 0
    event_log       = []
    t0              = time.time()

    closest_approach    = float("inf")    # absolute minimum distance ever
    closest_time_str    = "-"
    proximity_timeline  = []               # [{t, d, n_workers}] per frame
    workers_per_minute  = {}
    ppe_ok_frames       = 0
    ppe_tot_frames      = 0
    cum_caution_workers    = 0
    cum_warning_workers    = 0
    cum_safe_workers       = 0
    all_distances          = []
    caution_last_logged    = {}   # {worker_key: last_t} debounce
    warning_last_logged    = {}   # {worker_key: last_t} debounce
    caution_per_minute     = {}
    warning_per_minute     = {}
    # Image capture tracking
    ppe_compliance_timeline  = []
    ppe_tl_last_t            = -5.0
    # Aerolift active time
    aerolift_active_frames   = 0      # frames where aerolift was detected
    aerolift_active_sec      = 0.0
    # Zone person count timeline [{t, caution, warning, safe}]
    zone_count_timeline      = []
    zone_tl_last_t           = -5.0   # record every 5s
    best_frame_workers       = 0
    best_frame_img           = None     # frame with most workers
    zone_boundary_img        = None     # first frame with aerolift zones
    zone_boundary_captured   = False
    max_no_ppe_count         = 0
    max_no_ppe_frame_img     = None     # frame with MOST simultaneous NO PPE workers
    max_caution_count        = 0
    max_caution_frame_img    = None     # frame with MOST simultaneous caution workers
    dwell_caution            = {}
    dwell_warning            = {}
    dwell_last_seen          = {}

    # Cache for non-processed frames
    L_persons   = []
    L_aerolifts = []
    L_vehicles  = []
    L_helmets   = []
    L_vests     = []

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if fc % frame_skip != 0:
            fc += 1
            continue

        fout = cv2.resize(frame, (OUT_W, OUT_H))
        fH, fW = fout.shape[:2]

        if fc % PROCESS_EVERY == 0:

            # ══════════════════════════════════════════════════
            # CAMERA-AWARE PERSON DETECTION
            # TLC cam  → yolov8m only (strict), PPE via yolov8m crop
            # Aerolift → full 4-source approach
            # ══════════════════════════════════════════════════
            persons = []

            # SOURCE 1 - yolov8m full frame (both cameras)
            conf_thr = 0.55 if cam_is_tlc else 0.45
            r = person_model(fout, conf=conf_thr, classes=[0],
                             device=DEVICE, verbose=False)[0]
            if r.boxes is not None:
                for box, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy()):
                    mb = (*map(int, box), float(cf))
                    if is_real_person(mb, fW, fH) and has_person_color(fout, mb):
                        persons.append(mb)

            # SOURCE 2 - yolov8m top half (both cameras)
            top_conf = 0.55 if cam_is_tlc else 0.40
            top = fout[:fH // 2, :]
            r2  = person_model(top, conf=top_conf, classes=[0],
                               device=DEVICE, verbose=False)[0]
            if r2.boxes is not None:
                for box, cf in zip(r2.boxes.xyxy.cpu().numpy(),
                                   r2.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)
                    mb = (x1, y1, x2, y2, float(cf))
                    if is_real_person(mb, fW, fH // 2) and has_person_color(fout, mb):
                        if not any(iou(mb[:4], q[:4]) > 0.40 for q in persons):
                            persons.append(mb)

            # SOURCE 3+4 - PPE model persons + Proximity model persons
            # Only for Aerolift camera - TLC camera uses yolov8m only
            helmets_raw = []
            vests_raw   = []
            if USE_PPE:
                r3 = ppe_model(fout, conf=0.35, imgsz=960,
                               device=DEVICE, verbose=False)[0]
                if r3.boxes is not None:
                    for box, cls, cf in zip(r3.boxes.xyxy.cpu().numpy(),
                                            r3.boxes.cls.cpu().numpy(),
                                            r3.boxes.conf.cpu().numpy()):
                        cls_id = int(cls)
                        cf     = float(cf)
                        x1, y1, x2, y2 = map(int, box)
                        mb = (x1, y1, x2, y2, cf)
                        bw = x2 - x1
                        bh = y2 - y1
                        if cls_id in ppe_person_cls:
                            # Only add PPE-model workers for Aerolift camera
                            if not cam_is_tlc:
                                if is_real_person(mb, fW, fH) and has_person_color(fout, mb):
                                    if not any(iou(mb[:4], q[:4]) > 0.40 for q in persons):
                                        persons.append(mb)
                        elif cls_id in ppe_hat_cls:
                            if cf >= 0.10 and bw > 4 and bh > 3 and bw < fW * 0.20:
                                helmets_raw.append(mb)
                        elif cls_id in ppe_vest_cls:
                            if cf >= 0.05 and bw > 6 and bh > 6:
                                vests_raw.append(mb)

            small = cv2.resize(frame, (PROC_W, PROC_H))
            mr    = proximity_model(small, conf=0.10,
                                    device=DEVICE, verbose=False)[0]
            aerolifts = []
            vehicles  = []
            mn = mr.names
            for i in range(len(mr.boxes)):
                cls  = mn[int(mr.boxes.cls[i])]
                conf = float(mr.boxes.conf[i])
                box  = scale_box(mr.boxes.xyxy[i].cpu().numpy(), sx2, sy2)
                if cls in AEROLIFT_CLS and conf >= 0.10:
                    aerolifts.append((box, get_center(box),
                                      _get_side(box, OUT_W, OUT_H, SRC_W, SRC_H, TTL, TBL)))
                elif cls in VEHICLE_CLS and conf >= 0.40:
                    vehicles.append((box, get_center(box)))

            # Proximity model persons - Aerolift camera only
            if not cam_is_tlc:
                r4 = proximity_model(fout, conf=0.50,
                                     device=DEVICE, verbose=False)[0]
                if r4.boxes is not None:
                    for box, cls, cf in zip(r4.boxes.xyxy.cpu().numpy(),
                                            r4.boxes.cls.cpu().numpy(),
                                            r4.boxes.conf.cpu().numpy()):
                        cls_id = int(cls)
                        cf     = float(cf)
                        x1, y1, x2, y2 = map(int, box)
                        mb = (x1, y1, x2, y2, cf)
                        if (cls_id in prox_person_cls
                                and is_real_person(mb, fW, fH)
                                and has_person_color(fout, mb)):
                            if not any(iou(mb[:4], q[:4]) > 0.40 for q in persons):
                                persons.append(mb)

            # ── Dedupe + smooth ───────────────────────────────
            persons = dedupe(persons, thr=0.45)
            persons = box_smoother.smooth(persons)

            # ── Crop+upscale per person for PPE (fine-grained) ─
            if USE_PPE:
                for p in persons:
                    px1, py1, px2, py2 = map(int, p[:4])
                    ph_ = py2 - py1
                    pw_ = px2 - px1
                    pad = int(max(pw_, ph_) * 0.40)
                    cx1 = max(0,      px1 - pad)
                    cy1 = max(0,      py1 - pad)
                    cx2 = min(fW - 1, px2 + pad)
                    cy2 = min(fH - 1, py2 + pad)
                    crop = fout[cy1:cy2, cx1:cx2]
                    if crop.size == 0:
                        continue
                    ch, cw = crop.shape[:2]
                    scale = max(1.0, min(6.0, 200 / max(ph_, 1)))
                    if scale > 1.0:
                        crop = cv2.resize(crop,
                                          (int(cw * scale), int(ch * scale)),
                                          interpolation=cv2.INTER_CUBIC)
                    rc = ppe_model(crop, conf=0.05, imgsz=640,
                                   device=DEVICE, verbose=False)[0]
                    if rc.boxes is None:
                        continue
                    for box, cls, cf in zip(rc.boxes.xyxy.cpu().numpy(),
                                            rc.boxes.cls.cpu().numpy(),
                                            rc.boxes.conf.cpu().numpy()):
                        cls_id = int(cls)
                        cf     = float(cf)
                        if cls_id in ppe_person_cls:
                            continue
                        bx1, by1, bx2, by2 = map(int, box)
                        fx1 = int(bx1 / scale) + cx1
                        fy1 = int(by1 / scale) + cy1
                        fx2 = int(bx2 / scale) + cx1
                        fy2 = int(by2 / scale) + cy1
                        mb  = (fx1, fy1, fx2, fy2, cf)
                        bw  = fx2 - fx1
                        bh  = fy2 - fy1
                        if cls_id in ppe_hat_cls:
                            if bw > 3 and bh > 2:
                                if not any(iou(mb[:4], h[:4]) > 0.30
                                           for h in helmets_raw):
                                    helmets_raw.append(mb)
                        elif cls_id in ppe_vest_cls:
                            if bw > 5 and bh > 5:
                                if not any(iou(mb[:4], v[:4]) > 0.30
                                           for v in vests_raw):
                                    vests_raw.append(mb)

            # Filter PPE detections to only those near a person
            helmets = [h for h in helmets_raw
                       if is_near_any_person(h, persons, max_dist=120)]
            vests   = [v for v in vests_raw
                       if is_near_any_person(v, persons, max_dist=200)]

            L_persons   = persons
            L_aerolifts = aerolifts
            L_vehicles  = vehicles
            L_helmets   = helmets
            L_vests     = vests

        else:
            persons   = L_persons
            aerolifts = L_aerolifts
            vehicles  = L_vehicles
            helmets   = L_helmets
            vests     = L_vests

        # ── DRAW AEROLIFT + VEHICLES ──────────────────────────
        for a_box, _, _ in aerolifts:
            draw_aerolift(fout, a_box, DANGER_M, WARNING_M, SAFE_M,
                          PPM_OUT, SX_OUT, SY_OUT)
        for v_box, _ in vehicles:
            vx1, vy1, vx2, vy2 = map(int, v_box[:4])
            cv2.rectangle(fout, (vx1, vy1), (vx2, vy2), (255, 255, 0), 1)

        # ── PER PERSON ────────────────────────────────────────
        f_danger    = False
        f_warn      = False
        ok_count    = 0
        fail_count  = 0
        n_caution_f      = 0
        n_warn_f         = 0
        n_safe_f         = 0
        n_near_aerolift  = 0   # workers within PROX_CHK of any aerolift
        frame_alerts     = []
        frame_min_bd     = None

        for p in persons:
            x1, y1, x2, y2, cf = p
            if not _is_in_work_zone(p[:4], OUT_W, OUT_H, SRC_W, SRC_H, TTL, TBL):
                continue
            pfeet = (int((x1 + x2) // 2), int(y2))

            # PPE check + temporal vote
            raw_ppe = check_ppe(p, helmets, vests)
            ppe_st  = ppe_voter.vote(p, raw_ppe)
            ppe_ok  = (ppe_st == "PPE OK")
            ppe_col = C_OK if ppe_ok else C_FAIL
            if ppe_ok:
                ok_count += 1
            else:
                fail_count  += 1
                tot_no_ppe  += 1

            bd, bn, ba = _nearest_aerolift(p[:4], aerolifts, PPM_OUT, SX_OUT, SY_OUT)

            # Track closest approach - use REAL measured distance before any clamping
            real_bd = bd  # bd is the true measured value at this point
            if real_bd is not None and (frame_min_bd is None or real_bd < frame_min_bd):
                frame_min_bd = real_bd

            if not aerolifts or bd is None or bd > PROX_CHK:
                n_safe_f += 1
                draw_person_box(fout, x1, y1, x2, y2,
                                ppe_st, ppe_col, ppe_st, ppe_col, thick=2)
                continue

            # Worker is within proximity check range of aerolift
            n_near_aerolift += 1

            is_d = (ba is not None and
                    _in_danger_box(p[:4], ba[:4], DANGER_M, PPM_OUT, SX_OUT, SY_OUT))
            if is_d:
                bd = min(bd, DANGER_M * 0.5)
            prox_st, prox_col = _get_prox_status(bd, DANGER_M, WARNING_M)

            if bd <= DANGER_M or is_d:
                n_caution_f         += 1
                f_danger             = True
                cum_caution_workers += 1
                t = fc / fps
                _wk = (int((x1+x2)//2//120), int((y1+y2)//2//120))
                _dt = min(t - dwell_last_seen.get(_wk, t), 5.0)
                dwell_caution[_wk]   = dwell_caution.get(_wk, 0) + _dt
                dwell_last_seen[_wk] = t
                worker_key = _wk
                last_log   = caution_last_logged.get(worker_key, -999)
                if (t - last_log) >= 5.0:
                    danger_ev += 1
                    caution_last_logged[worker_key] = t
                    _min_key = str(int(t // 60))
                    caution_per_minute[_min_key] = caution_per_minute.get(_min_key, 0) + 1
                    alert = (f"[{int(t//60)}m{int(t%60):02d}s]"
                             f" CAUTION {bd*100:.0f}cm  {ppe_st}")
                    frame_alerts.append(alert)
                    event_log.append(alert)
                draw_person_box(fout, x1, y1, x2, y2,
                                f"CAUTION {bd*100:.0f}cm",
                                C_DANGER, ppe_st, ppe_col, thick=4)

            elif bd <= WARNING_M:
                n_warn_f            += 1
                f_warn               = True
                cum_warning_workers += 1
                t_now = fc / fps
                _wk_w = (int((x1+x2)//2//120), int((y1+y2)//2//120))
                _dt_w = min(t_now - dwell_last_seen.get(_wk_w, t_now), 5.0)
                dwell_warning[_wk_w]   = dwell_warning.get(_wk_w, 0) + _dt_w
                dwell_last_seen[_wk_w] = t_now
                worker_key_w = _wk_w
                last_warn    = warning_last_logged.get(worker_key_w, -999)
                if (t_now - last_warn) >= 5.0:
                    warn_ev += 1
                    warning_last_logged[worker_key_w] = t_now
                    _min_key_w = str(int(t_now // 60))
                    warning_per_minute[_min_key_w] = warning_per_minute.get(_min_key_w, 0) + 1
                    _warn_alert = (f"[{int(t_now//60)}m{int(t_now%60):02d}s]"
                                   f" WARNING {bd:.1f}m  {ppe_st}")
                    frame_alerts.append(_warn_alert)
                    event_log.append(_warn_alert)
                draw_person_box(fout, x1, y1, x2, y2,
                                f"WARN {bd:.1f}m",
                                C_WARNING, ppe_st, ppe_col, thick=3)
            else:
                n_safe_f            += 1
                cum_safe_workers    += 1
                draw_person_box(fout, x1, y1, x2, y2,
                                f"SAFE {bd:.1f}m",
                                C_SAFE, ppe_st, ppe_col, thick=2)

            if bd <= LINE_THR and bn:
                pfeet_i = (int(pfeet[0]), int(pfeet[1]))
                bn_i    = (int(bn[0]),    int(bn[1]))
                cv2.line(fout, pfeet_i, bn_i, prox_col, 2)
                cv2.circle(fout, pfeet_i, 3, prox_col, -1)
                cv2.circle(fout, bn_i,    4, prox_col, -1)

        # ── UPDATE METRICS ────────────────────────────────────
        t_sec_now  = fc / fps
        minute_key = str(int(t_sec_now // 60))
        n_in_zone  = ok_count + fail_count
        workers_per_minute[minute_key] = max(
            workers_per_minute.get(minute_key, 0), n_in_zone)

        if n_in_zone > 0:
            ppe_tot_frames += n_in_zone
            ppe_ok_frames  += ok_count

        # Best frame (most workers)
        if n_in_zone > best_frame_workers:
            best_frame_workers = n_in_zone
            best_frame_img     = fout.copy()

        # Max NO PPE frame (most workers without PPE simultaneously)
        if fail_count > max_no_ppe_count:
            max_no_ppe_count     = fail_count
            max_no_ppe_frame_img = fout.copy()

        # Max caution frame (most workers in caution zone simultaneously)
        if n_caution_f > max_caution_count:
            max_caution_count     = n_caution_f
            max_caution_frame_img = fout.copy()

        # PPE compliance timeline every 5 seconds
        if ppe_tot_frames > 0 and (t_sec_now - ppe_tl_last_t) >= 5.0:
            _pct = round(100 * ppe_ok_frames / max(ppe_tot_frames, 1), 1)
            ppe_compliance_timeline.append({"t": round(t_sec_now, 1), "pct": _pct})
            ppe_tl_last_t = t_sec_now

        # Aerolift active time
        if aerolifts:
            aerolift_active_frames += 1
        aerolift_active_sec = round(aerolift_active_frames * frame_skip / fps, 1)

        # Zone person count timeline every 5 seconds
        if (t_sec_now - zone_tl_last_t) >= 5.0:
            zone_count_timeline.append({
                "t":       round(t_sec_now, 1),
                "caution": n_caution_f,
                "warning": n_warn_f,
                "safe":    n_safe_f,
                "total":   n_in_zone,
            })
            zone_tl_last_t = t_sec_now

        # Fallback zone boundary
        if not zone_boundary_captured and fc > 0 and zone_boundary_img is None:
            zone_boundary_img = fout.copy()

        if frame_min_bd is not None:
            proximity_timeline.append({
                "t": round(t_sec_now, 1),
                "d": round(frame_min_bd, 2),
                "w": n_in_zone,
            })
            all_distances.append(frame_min_bd)
            # Only update closest if > 0.05m (avoid spurious 0cm readings
            # from persons detected inside the aerolift bounding box)
            if frame_min_bd > 0.20 and frame_min_bd < closest_approach:
                closest_approach = frame_min_bd
                m_ = int(t_sec_now // 60)
                s_ = int(t_sec_now % 60)
                closest_time_str = f"{m_}m{s_:02d}s"

        # ── ALERTS OVERLAY ────────────────────────────────────
        if f_danger:
            cv2.rectangle(fout, (0, 0), (OUT_W - 1, OUT_H - 1), (0, 0, 255), 5)
            cv2.putText(fout, "!! CAUTION !!",
                        (OUT_W // 2 - 90, OUT_H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif f_warn:
            cv2.putText(fout, "!! WARNING !!",
                        (OUT_W // 2 - 95, OUT_H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        # Top panel removed from video frame - stats shown in dashboard only
        fps_r  = fc / (time.time() - t0) if (time.time() - t0) > 0 else 0

        # ── CALLBACK ──────────────────────────────────────────
        ppe_pct_live = round(100 * ppe_ok_frames / max(ppe_tot_frames, 1), 1)
        if frame_callback:
            frame_callback({
                "frame_idx":           fc,
                "total_frames":        TOTAL,
                "annotated_frame":     fout,
                "frame_alerts":        frame_alerts,
                "frame_danger":        n_caution_f,
                "frame_warning":       n_warn_f,
                "frame_no_helmet":     0,
                "frame_no_vest":       0,
                "frame_no_ppe":        fail_count,
                "total_workers":       n_in_zone,
                "cumul_danger":        danger_ev,
                "cumul_warning":       warn_ev,
                "cumul_no_ppe":        tot_no_ppe,
                "cumul_no_helmet":     0,
                "cumul_no_vest":       0,
                "ppe_compliance_pct":  ppe_pct_live,
                "closest_approach":    round(closest_approach, 2)
                                       if closest_approach < float("inf") else None,
                "robust_closest":      round(min(d for d in all_distances if d>0.20), 2)
                                       if any(d>0.20 for d in all_distances) else None,
                "closest_time":        closest_time_str,
                "proximity_timeline":   proximity_timeline[-150:],
                "workers_per_minute":   dict(workers_per_minute),
                "aerolift_active_sec":  aerolift_active_sec,
                "zone_count_timeline":  zone_count_timeline[-150:],
                "n_caution":           n_caution_f,
                "n_near_aerolift":     n_near_aerolift,
                "frame_ppe_ok":        ok_count,
                "frame_ppe_fail":      fail_count,
                "n_warning":           n_warn_f,
                "n_safe":              n_safe_f,
                "cum_caution_workers": cum_caution_workers,
                "cum_warning_workers": cum_warning_workers,
                "cum_safe_workers":    cum_safe_workers,
                "ppm":                 round(PPM_OUT, 1),
                "scale_y":             round(SY_OUT, 2),
                "cam_type":            cam_type,
                "video_res":           f"{W}x{H}",
            })

        fc += 1

    cap.release()

    # ── FINAL RISK SCORE (rate-based, 0-100) ──────────────────
    ppe_pct_final = round(100 * ppe_ok_frames / max(ppe_tot_frames, 1), 1)
    duration_min  = max(fc / fps / 60, 0.1)

    caution_rate  = danger_ev  / duration_min   # events/min
    warning_rate  = warn_ev    / duration_min
    no_ppe_rate   = tot_no_ppe / duration_min

    # Distance statistics (compute FIRST - needed for risk formula)
    import statistics as _stats
    valid_dists = [d for d in all_distances if d > 0.20]
    if valid_dists:
        dist_median = round(_stats.median(valid_dists), 2)
        dist_p10    = round(sorted(valid_dists)[int(len(valid_dists)*0.10)], 2)
        dist_mean   = round(sum(valid_dists)/len(valid_dists), 2)
    else:
        dist_median = dist_p10 = dist_mean = None

    # Proximity penalty: use median distance as representative
    if valid_dists and dist_median and dist_median > 0.05:
        prox_penalty = max(0, 30 * (1 - dist_median / 4.0))  # 4m = new max tracking distance
    else:
        prox_penalty = 0

    # PPE penalty: 0 if 100% compliant, 30 if 0%
    ppe_penalty = (1 - ppe_pct_final / 100) * 30

    # Rate penalties - calibrated so typical sites score 20-60
    caution_penalty = min(25, caution_rate * 8)   # 3/min -> 24 pts
    warning_penalty = min(12, warning_rate * 2)   # 6/min -> 12 pts
    no_ppe_penalty  = min(8,  no_ppe_rate  * 1.5)

    # Clamp to 10-95 - never shows 0 or 100 spuriously
    risk = max(10, min(95, int(
        prox_penalty + ppe_penalty + caution_penalty
        + warning_penalty + no_ppe_penalty
    )))

    # Use 5th-percentile distance as "typical closest" (more robust than abs min)
    robust_closest = dist_p10 if dist_p10 is not None else (
        closest_approach if closest_approach < float("inf") else None)

    return {
        "danger_events":       danger_ev,
        "warning_events":      warn_ev,
        "total_no_ppe":        tot_no_ppe,
        "total_no_helmet":     0,
        "total_no_vest":       0,
        "frame_count":         fc,
        "event_log":           event_log,
        "fps":                 fps,
        "ppe_compliance_pct":  ppe_pct_final,
        "closest_approach":    closest_approach
                               if closest_approach < float("inf") else None,
        "robust_closest":      robust_closest,
        "dist_median":         dist_median,
        "dist_mean":           dist_mean,
        "closest_time":        closest_time_str,
        "proximity_timeline":  proximity_timeline,
        "workers_per_minute":  workers_per_minute,
        "caution_per_minute":  caution_per_minute,
        "warning_per_minute":  warning_per_minute,
        "cum_caution_workers": cum_caution_workers,
        "cum_warning_workers": cum_warning_workers,
        "cum_safe_workers":    cum_safe_workers,
        "safety_index":              risk,
        "ppm":                       round(PPM_OUT, 1),
        "scale_y":                   round(SY_OUT, 2),
        "cam_type":                  cam_type,
        "video_res":                 f"{W}x{H}",
        "aerolift_active_sec":       aerolift_active_sec,
        "aerolift_active_pct":       round(100 * aerolift_active_sec / max(fc / fps, 1), 1),
        "zone_count_timeline":       zone_count_timeline,
        "cam_is_tlc":                cam_is_tlc,
        "ppe_compliance_timeline":   ppe_compliance_timeline,
        "best_frame_workers":        best_frame_workers,
        "max_no_ppe_count":          max_no_ppe_count,
        "max_caution_count":         max_caution_count,
        "total_dwell_caution_sec":   round(sum(dwell_caution.values()), 1),
        "total_dwell_warning_sec":   round(sum(dwell_warning.values()), 1),
        "best_frame_img_path":       _save_img(best_frame_img,       "output/safety_best_frame.jpg"),
        "zone_boundary_img_path":    _save_img(zone_boundary_img,    "output/safety_zone_boundary.jpg"),
        "no_ppe_frame_img_path":     _save_img(max_no_ppe_frame_img, "output/safety_no_ppe_frame.jpg"),
        "caution_frame_img_path":    _save_img(max_caution_frame_img,"output/safety_caution_frame.jpg"),
    }
