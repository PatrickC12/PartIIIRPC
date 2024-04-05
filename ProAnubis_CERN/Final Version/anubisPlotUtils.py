import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as image
import mplhep as hep
hep.style.use([hep.style.ATLAS])
import sys
from PIL import Image

# def importFromROOTFile(filename):
#     inputRoot = ROOT.TFile(filename)
#     eventTree = inputRoot.Get("Events")    
#     data = [[] for tdc in range(5)]
#     for event in range(eventTree.GetEntries()):
#         eventTree.GetEvent(event)
#         tdc = eventTree.TDC
#         thisEvent = []
#         hits = eventTree.Words
#         for hit in hits:
#             thisEvent.append(hit)
#         data[tdc].append(thisEvent)
#         #eventTime = eventTree.TimeStamp.AsString()
#     endTime = inputRoot.Get("endTime").AsString()
#     return data, endTime

def countChannels(events):
    #Expects events from one TDC, counts how many hits each channel has within the event list
    chanCounts = [0 for x in range(128)]
    for event in events:
        for word in event:
            try:
                chanCounts[(word>>24)&0x7f]=chanCounts[(word>>24)&0x7f]+1
            except:
                print(word>>24)
    return chanCounts

def heatFromROOTFile(dataFile, time=1800, name="proANUBISMonitorHeatMap"):
    #Plots heat maps from triggered data, showing the hit rate in each rpc channel. 2D plots designed to replicate RPC layout and channel counting direction.
    thisData, endTime = importFromROOTFile(dataFile)
    thisHitData = {}
    addresses = ['ee00','ee01','ee02','ee03','ee04']
    for tdc in range(5):
        thisHitData[addresses[tdc]] = countChannels(thisData[tdc])
    makeHitMaps(thisHitData,endTime, name,False,unit='hz',time=time)
    
def plotPhi(array, name, zrange = [0.01,200], unit='khz', time=60.):
    fig, ax = plt.subplots(1, figsize=(16, 8), dpi=100)
    #lab = hep.atlas.label(com=False,data=True, label="Internal")
    #lab[2].set_text("")
    im = image.imread('/eos/user/m/mireveri/anubis/ANUBISLogo.png')
    phichannels = [x-0.5 for x in range(65)]
    phiHist = ((np.array([array])/time).transpose(),np.array(phichannels),np.array([0,1]))
    thisHist = hep.hist2dplot(phiHist,norm=colors.LogNorm(zrange[0],zrange[1]))
    thisHist.cbar.set_label('Event Rate ('+unit+')', rotation=270, y=0.4)
    plt.xlabel("Channel")
    plt.ylabel(" ")
    plt.title(name)
    fig.tight_layout()
    ax.get_yaxis().set_visible(False)
    ax.imshow(im, aspect='auto', extent=(1, 3, .88, .93), zorder=1)
    plt.savefig('/eos/user/m/mireveri/anubis/'+name.strip(" ")+".png")
    plt.close()
    return '/eos/user/m/mireveri/anubis/'+name.strip(" ")+".png"

def plotEta(array, name, zrange = [0.01,200], unit='khz', time=60.):
    fig, ax = plt.subplots(1, figsize=(16, 8), dpi=100)
    #lab = hep.atlas.label(com=False,data=True, label="Internal")
    #lab[2].set_text("")
    im = image.imread('/eos/user/m/mireveri/anubis/ANUBISLogo.png')
    etachannels = [x-0.5 for x in range(33)]
    etaHist = (np.array([array])/time,np.array([0,1]),np.array(etachannels))
    thisHist = hep.hist2dplot(etaHist,norm=colors.LogNorm(zrange[0],zrange[1]))
    thisHist.cbar.set_label('Event Rate ('+unit+')', rotation=270, y=0.4)
    plt.ylim(31.5,-0.5)
    plt.ylabel("Channel")
    plt.xlabel(" ")
    plt.title(name)
    fig.tight_layout()
    ax.get_xaxis().set_visible(False)
    ax.imshow(im, aspect='auto', extent=(0.03, 0.06, 3.2, 1.6), zorder=1)
    plt.savefig('/eos/user/m/mireveri/anubis/'+name.strip(" ")+".png")
    plt.close()
    return '/eos/user/m/mireveri/anubis/'+name.strip(" ")+".png"

def combinePlots(plots,imname):
    images = [Image.open(x) for x in plots]
    widths, heights = zip(*(i.size for i in images))

    total_width = int(2*widths[0])
    if(len(plots)|2>0):
        max_height = int((sum(heights)+heights[0])/2)
    else:
        max_height = int(sum(heights)/2)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    y_offset = 0
    even = True
    for im in images:
        if even:
            new_im.paste(im, (x_offset,y_offset))
            x_offset += im.size[0]
            even = False
        else:
            new_im.paste(im,(x_offset,y_offset))
            x_offset = 0
            y_offset += im.size[1]
            even = True

    new_im.save('/eos/user/m/mireveri/anubis/data/monitor/'+imname.strip(" ")+'.pdf')
    
def makeHitMaps(filenames, plotTitle, imname, useJson=True, unit='khz', time=60.):
    if(useJson):
        hitData = {}
    
        addresses = ['ee00','ee01','ee02','ee03','ee04']
        for fname in filenames:
            thisFile = open(fname)
            jsonData = json.load(thisFile)
            for addr in addresses:
                try:
                    hitData[addr]=jsonData['Summary']['TDCs'][addr]['nHits']
                except KeyError:
                    continue
        for addr in addresses:
            if addr not in hitData.keys():
                hitData[addr] = [0 for x in range(128)]
            else:
                for idx, hit in enumerate(hitData[addr]):
                    hitData[addr][idx]= hit/1000. #Divide by 1000 to convert to khz
    else:
        hitData = filenames
    tripEtaLow = hitData['ee00'][0:32]
    tripPhiLow = hitData['ee00'][32:96]
    tripEtaMid = hitData['ee00'][96:128]
    tripPhiMid = hitData['ee01'][0:64]
    tripEtaTop = hitData['ee01'][64:96]
    tripPhiTop = hitData['ee01'][96:128]+hitData['ee02'][0:32]
    singEta = hitData['ee02'][32:64]
    singPhi = hitData['ee02'][64:128]
    doubEtaLow = hitData['ee03'][0:32]
    doubPhiLow = hitData['ee03'][32:96]
    doubEtaTop = hitData['ee03'][96:128]
    doubPhiTop = hitData['ee04'][0:64]
    imageArr = []
    imageArr.append(plotPhi(tripPhiLow,"Phi Triplet Low "+plotTitle, unit=unit, time=time))
    imageArr.append(plotEta(tripEtaLow,"Eta Triplet Low "+plotTitle, unit=unit, time=time))
    imageArr.append(plotPhi(tripPhiMid,"Phi Triplet Mid "+plotTitle, unit=unit, time=time))
    imageArr.append(plotEta(tripEtaMid,"Eta Triplet Mid "+plotTitle, unit=unit, time=time))
    imageArr.append(plotPhi(tripPhiTop,"Phi Triplet Top "+plotTitle, unit=unit, time=time))
    imageArr.append(plotEta(tripEtaTop,"Eta Triplet Top "+plotTitle, unit=unit, time=time))
    imageArr.append(plotPhi(singPhi,"Phi Singlet "+plotTitle, unit=unit, time=time))
    imageArr.append(plotEta(singEta,"Eta Singlet "+plotTitle, unit=unit, time=time))
    imageArr.append(plotPhi(doubPhiLow,"Phi Doublet Low "+plotTitle, unit=unit, time=time))
    imageArr.append(plotEta(doubEtaLow,"Eta Doublet Low "+plotTitle, unit=unit, time=time))
    imageArr.append(plotPhi(doubPhiTop,"Phi Doublet Top "+plotTitle, unit=unit, time=time))
    imageArr.append(plotEta(doubEtaTop,"Eta Doublet Top "+plotTitle, unit=unit, time=time))
    combinePlots(imageArr,imname)
    for im in imageArr:
        os.remove(im)