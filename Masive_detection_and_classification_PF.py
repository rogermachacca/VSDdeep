#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:08:59 2023

@author: rmachacca
"""
# ============================================================ Load Utils
from VSDdeep_Utils import SeqSelfAttention, FeedForward, LayerNormalization
from obspy.clients.fdsn import Client
#from obspy.clients.earthworm import Client
from obspy import UTCDateTime, Stream
from matplotlib import mlab
import numpy as np
import time, math, csv, os
import tensorflow as tf
tf.random.set_seed(1234)
# ========================================================  Def. New functions
class Create_config:
    pass

def _nearest_pow_2(x):
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b

def norm_3Cdata(data):
    m,n = data.shape
    for i in range(n):
        minA =  data[:,i].min()
        data[:,i] = data[:,i] - minA
        data[:,i] = data[:,i]/data[:,i].max()
    return data

def _normalize(data, mode = 'max'):  
    'Normalize waveforms in each batch'
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data              

    elif mode == 'std':               
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data

def yc_patch_inv(X1,n1,n2,l1,l2,o1,o2):
    tmp1=np.mod(n1-l1,o1)
    tmp2=np.mod(n2-l2,o2)
    if (tmp1!=0) and (tmp2!=0):
        A     = np.zeros((n1+o1-tmp1,n2+o2-tmp2))
        mask  = np.zeros((n1+o1-tmp1,n2+o2-tmp2)) 

    if (tmp1!=0) and (tmp2==0): 
        A   = np.zeros((n1+o1-tmp1,n2))
        mask= np.zeros((n1+o1-tmp1,n2))

    if (tmp1==0) and (tmp2!=0):
        A    = np.zeros((n1,n2+o2-tmp2))   
        mask = np.zeros((n1,n2+o2-tmp2))   

    if (tmp1==0) and (tmp2==0):
        A    = np.zeros((n1,n2))
        mask = np.zeros((n1,n2))

    N1,N2= np.shape(A)
    ids=0
    for i1 in range(0,N1-l1+1,o1):
        for i2 in range(0,N2-l2+1,o2):
            A[i1:i1+l1,i2:i2+l2]=A[i1:i1+l1,i2:i2+l2]+np.reshape(X1[:,ids],(l1,l2))
            mask[i1:i1+l1,i2:i2+l2]=mask[i1:i1+l1,i2:i2+l2]+ np.ones((l1,l2))
            ids=ids+1
    A=A/mask;  
    A=A[0:n1,0:n2]
    return A

def yc_patch(A,l1,l2,o1,o2):
    n1,n2=np.shape(A);
    tmp=np.mod(n1-l1,o1)
    if tmp!=0:
#        print(np.shape(A), o1-tmp, n2)
        A=np.concatenate([A,np.zeros((o1-tmp,n2))],axis=0)

    tmp=np.mod(n2-l2,o2);
    if tmp!=0:
        A=np.concatenate([A,np.zeros((A.shape[0],o2-tmp))],axis=-1); 

    N1,N2 = np.shape(A)
    X=[]
    for i1 in range (0,N1-l1+1, o1):
        for i2 in range (0,N2-l2+1,o2):
            tmp=np.reshape(A[i1:i1+l1,i2:i2+l2],(l1*l2,1));
            X.append(tmp);  
    X = np.array(X)
    return X[:,:,0]

def get_spectrogram(data,samp_rate,fmax, mat, mult=6, wlen=2, per_lap=0.8):
    npts = len(data)
    if not wlen:
        wlen = samp_rate / 100.
    nfft = int(_nearest_pow_2(wlen * samp_rate))
    if nfft > npts:
        nfft = int(_nearest_pow_2(npts / 8.0))
    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                            pad_to=mult, noverlap=nlap)
    specgram = 20 * np.log10(specgram[1:, :])
    specgram = np.flipud(specgram)
    m,n = specgram.shape

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0
    extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
              freq[0] - halfbin_freq, freq[int(m/2)] + halfbin_freq)
    Nspect = specgram[int(m/2):,:]
    mm,nn = Nspect.shape
    mat = mat * Nspect.min()
    mat[:,:nn] = Nspect
    return (mat,extent)

def build_array(Data):
    tt1 = Data.sort(['starttime'])[0].stats.starttime
    tt2 = Data.sort(['endtime'], reverse=True)[0].stats.endtime
    Data.trim(tt1, tt2, pad=True, nearest_sample=True, fill_value=0)

    trE = Data.select(component='E')[0].data
    trN = Data.select(component='N')[0].data
    trZ = Data.select(component='Z')[0].data
    m,n = max(len(trE), len(trN), len(trZ)), 3

    data = np.zeros([m,n])
    data[:len(trE),0] = trE
    data[:len(trN),1] = trN
    data[:len(trZ),2] = trZ
    data = norm_3Cdata(data)
    return data

def VSDdeep_Continous_Windows_Overlapping_EvenTime(Data,ch0,ch1,ch2,model,n1,n2,w1,w2,s1z,s2z,cfg):
    norm_mode = 'std'
    astart = Data[0].stats.starttime
    winlen = int(cfg.wind * cfg.fs)
    X = np.zeros((int(ch0.shape[0]), winlen, 3))
    print( 'data are split in :' + str(ch0.shape[0]))
    for kq in range(0, int(ch0.shape[0])):
        X[kq,:,0] = ch0[kq]
        X[kq,:,1] = ch1[kq]
        X[kq,:,2] = ch2[kq]
    X = _normalize(X, norm_mode)
    lab = model(X)
    lab = lab[:,:,0]
    lab = np.transpose(lab)
    labf = yc_patch_inv(lab,n1,n2,w1,w2,s1z,s2z)
    return labf, astart

def Patching(Data,w1,w2,s1z,s2z,cfg):
    tr = Data.select(component='E')[0]
    ach0 = np.reshape(tr.data, ((tr.data).shape[0],1))
    tr = Data.select(component='N')[0]
    ach1 = np.reshape(tr.data, ((tr.data).shape[0],1))
    tr = Data.select(component='Z')[0]
    ach2 = np.reshape(tr.data, ((tr.data).shape[0],1))
    ch0 = yc_patch(ach0,w1,w2,s1z,s2z)
    ch1 = yc_patch(ach1,w1,w2,s1z,s2z)
    ch2 = yc_patch(ach2,w1,w2,s1z,s2z)
    return ch0,ch1,ch2

def detect_eve(Data, VSD_model, VSC_3Cmodel, wrtrD_3C, wrtrC_3C, cfg):
    w1 = int(cfg.wind * cfg.fs)
    w2 = 1
    s1z = int(cfg.wind_steep * cfg.fs)
    s2z = 1
    ch0, ch1, ch2 = Patching(Data,w1,w2,s1z,s2z,cfg)
    n1 = len(Data[0].data)
    n2 = 1
    labf, _t0_ = VSDdeep_Continous_Windows_Overlapping_EvenTime(Data,ch0,ch1,ch2,VSD_model,n1,n2,w1,w2,s1z,s2z,cfg)
    thre = cfg.VSDdeep_thsld
    y = np.where(labf>thre,1,0)
    yy = np.copy(y)
    for i in range(1, len(yy)):
        yy[i] = y[i] - y[i-1]
    if y[-1] == 1:
        yy[-1] = -1
    Onsets = list(np.where(yy == 1)[0])
    Offsets = list(np.where(yy == -1)[0])
    if len(Onsets) > 0:   # ---------------------------- Revisa si hay eventos detectados
        for i in range(len(Onsets)):
            _t1, _t2 = _t0_ + Onsets[i]/cfg.fs, _t0_ + Offsets[i]/cfg.fs
            durEVE = (_t2 - _t1)
            #print("classifying .., ", _t1, durEVE)
            # ---------------------------------------------------------- Write detections
            wrtrD_3C.writerow([str(_t1)[:22],"%.2f" % (durEVE)]) # Add All Triggers
            # ========================================== Run Clasifications
            Cst = Data.copy().trim(_t1, _t2)
            for windowed_st in Cst.slide(window_length=60.0, step=60.0, include_partial_windows=True):
                windowed_st = chk_st(windowed_st, cfg.cpns)
                if len(windowed_st[0].data) >= cfg.minLastEve_for_class*cfg.fs: # Check min duration
                    Ct0 = windowed_st[2].stats.starttime   # use Z
                    EventLasting = windowed_st[2].times()[-1]          # use Z
                    # --------------------------------- Build spectrogram for VSCdeep
                    data = build_array(windowed_st)
                    m, n = cfg.stft_dim
                    specgram0 = np.ones([1, m, n, 3])
                    mat_zero = np.ones([m, n])
                    for cha in cfg.cpns_ix:
                        wv = data[:,cha]
                        wv = wv - wv.mean()
                        specgram, extent = get_spectrogram(wv, cfg.fs, cfg.fmax1, mat_zero)
                        specgram0[0, :, :, cha] = specgram
                    
                    prediction3C = VSC_3Cmodel.predict(specgram0)             # ------ Use 3C
                    prediction3C = list(prediction3C[0]*100)
                    ix_max_3C = np.argmax(np.array(prediction3C))
                    wrtrC_3C.writerow([cfg.commands[ix_max_3C], str(Ct0)[:22], "%.2f" % (EventLasting), ["%.2f" % ix for ix in prediction3C]])
                    
    else:
        print('No events detected in :', _t0_)
        
def chk_st(st, ocha):
    cha=[]
    for tr in st:
        cha.append(tr.id[-1])
    empty_cha = list(set(ocha).difference(cha))
    if len(empty_cha) > 0:
        for i in empty_cha:
            Ctr = st.copy().select(component=cha[0])[0]
            Ctr.stats.component = i
            st.append(Ctr)
    return st

def get_waveforms(client, t1, t2, cfg):
    net, sta, loc, cpn = cfg.info.split(".")
    try:
        st = client.get_waveforms(net, sta, loc, cpn, t1, t2)
    except:
        st = Stream()
    if len(st) > 0:
        #st.detrend('spline', order=3, dspline=1000)
        st.detrend("linear")
        st.detrend('demean')
        for tr in st:
            if tr.stats.sampling_rate != cfg.fs:
                tr.resample(cfg.fs)
        st.merge(method=1, fill_value='interpolate')
        st.filter('bandpass', freqmin=cfg.fmin1, freqmax=cfg.fmax1, corners=cfg.corners, zerophase=True)
        st.sort(['starttime'])
        st.trim(t1,t2,pad=True, nearest_sample=True,fill_value=0)
    return st
    
# ================================================================ Load Models
hm_models="./models/"
# --------------------------------------------- Load pre-trained model VSDdeep
VSD_model = tf.keras.models.load_model(hm_models + 'VSDdeep_PF.h5',
                                       custom_objects={'SeqSelfAttention':SeqSelfAttention,
                                       'FeedForward':FeedForward,
                                       'LayerNormalization':LayerNormalization})
# --------------------------------------------- Load pre-trained model VSDdeep
VSC_3C_model = tf.keras.models.load_model(hm_models + 'VSCdeep_PF.h5')
# ================================================================ Def. Config
cfg = Create_config()
# ---------------------- Fill the fields of the record
cfg.name = 'Roger Machacca'
cfg.web_service="IPGP"
cfg.web_port=None
cfg.info = "PF.BOR.00.EH?"  # we use ? for [E, N, Z]
cfg.cpns = ['E', 'N', 'Z']
cfg.cpns_ix = [0, 1, 2]
cfg.fs = 100.0    # Frequency sample
# -------------------------------------- conf. for Detection
cfg.fmin1 = 1.0
cfg.fmax1 = 25.0
cfg.corners = 4.0
# -------------------------------------- conf. for Classification CNN
cfg.stft_dim = [256, 111]
cfg.commands = ['VT', 'TR', 'TC', 'LP', 'RF', 'NO', 'RU']
# -------------------------------------- conf. 
cfg.wind = 60.0   # Window to analize [s]
cfg.wind_steep = 20.0   # Steep to overlap [s]
cfg.hm_out = './output_DC/'
cfg.t_ini = UTCDateTime("2019-02-01T00:00:00.00")
cfg.t_fin = UTCDateTime("2019-03-01T00:00:00.00")
cfg.t_delta = 60*60*1  # last value corresponde to hours
# -------------------------------------- Output control for Classificatiton with VSCdeep
cfg.VSDdeep_thsld = 0.5 
cfg.minLastEve_for_class = 5   # it is the min duration to be classified considering the FFT

# =============================================================== Start Script
#start_time = time.time()
t1 = cfg.t_ini
t2 = cfg.t_fin
client = Client(cfg.web_service)
#client = Client(cfg.web_service, cfg.web_port) # for EW
hd_3C = None
while t1 < t2:
    print('================= t :', t1)
    # -------------------------------------------------------------- Creating Output Files
    flOUT_class_3C = cfg.hm_out+"VSDCdeep_"+cfg.info[:-1]+'Z_'+(t1+10).strftime('%Y_Classifications.csv')
    flOUT_trigs_3C = flOUT_class_3C.rsplit("_",1)[0] + '_Detections.csv' # New line
    if os.path.exists(flOUT_class_3C) == False:
        hd_3C = ['EveType','starttime', 'duration', 'Prob']
        hdD = ['starttime', 'duration']
    myfileC_3C = open(flOUT_class_3C,'a')
    myfileD_3C = open(flOUT_trigs_3C,'a')
    wrtrC_3C = csv.writer(myfileC_3C, delimiter=',', quotechar='"')
    wrtrD_3C = csv.writer(myfileD_3C, delimiter=',', quotechar='"')
    if hd_3C != None: # This condition add header to new data
        wrtrC_3C.writerow(hd_3C)
        wrtrD_3C.writerow(hdD)
        hd_3C = None
        hdD = None      # for detection
    # ----------------------------------------------------------------- Get Waveforms
    st = get_waveforms(client, t1, t1 + cfg.t_delta, cfg)
    if len(st) > 0:
        st = chk_st(st, cfg.cpns)
        detect_eve(st, VSD_model, VSC_3C_model, wrtrD_3C, wrtrC_3C, cfg)
    else:
        print('No data for:', t1)
    t1 = t1 + cfg.t_delta
    myfileC_3C.close() # Close 3C Classification csv file
    myfileD_3C.close() # Close triggers csv file
    
# print("Done! ---- %.5f seconds ---" % (time.time() - start_time))
