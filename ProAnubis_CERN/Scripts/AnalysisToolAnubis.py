from PIL import Image
import h5py
import anubisPlotUtils as anPlot
import json
import numpy as np
import os
import hist as hi
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'GTK3Agg', etc.
import mplhep as hep
hep.style.use([hep.style.ATLAS])
import sys

def importFromTextFile(filename):
    inputText = open(filename)
    thisEvent = []
    data = [[] for tdc in range(5)]
    tdc=0
    for line in inputText:
        if "Header" in line:
            thisEvent = []
            tdc = int(line.split(" ")[1].strip(","))
        elif "Data" in line:
            thisEvent.append(int("0x"+line.split(" ")[2].strip("."),0))
        elif "EOB" in line:
            data[tdc].append(thisEvent)
    return data

def importFromHDF5File(filename):
    inputHDF5 = h5py.File(filename)
    thisEvent = []
    data = [[] for tdc in range(5)]
    tdc=0
    for event in inputHDF5['data']:
        tdc = event[0]-60928
        thisEvent = []
        for hit in event[2]:
            thisEvent.append(hit)
        data[tdc].append(thisEvent)
    return data

def importDatafile(filename):
    if "txt" in filename.split(".")[-1]:
        return importFromTextFile(filename)
    elif "h5" in filename.split(".")[-1]:
        return importFromHDF5File(filename)
    else:
        print("File type not recognized. Expect .txt or .h5 input file.")
        
def convertDataToHDF5(data, outfileName):
    npformattedData = []
    dt = np.dtype([('tdc', np.int_, 32), ('time', np.int_, 32),('data', np.ndarray)])
    for event in range(len(data[0])):
        for tdc in range(5):
            thisArr = np.array(data[tdc][event],dtype=np.uint32)
            thisPoint = (int(60928+tdc),int(0),thisArr)
            npformattedData.append(thisPoint)
    h5formattedData = np.array(npformattedData,dtype=dt)
    hf = h5py.File(outfileName+'.h5', 'w')
    #Read an input HDF5 file to get the right dtype. Does not work just trying to create an identical one - not sure what's missing
    preMadeDtype = h5py.File(current_directory +'\ProAnubis_CERN\ProAnubisData\60sRun_24_3_4.h5','r')['data'].dtype
    dset = hf.create_dataset('data',(len(h5formattedData),),dtype=preMadeDtype)
    for idx in range(len(h5formattedData)):
        dset[idx] = h5formattedData[idx]
    hf.close()
    
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

def getEventTimes(events):
    eventTimes = []
    for event in events:
        for word in event:
            eventTimes.append(word&0xfffff)
    return eventTimes

def makeSingleLayer(data, name):
    #Heatmap plot of one RPC layer. Takes already-split heat map, used by event display
    fig, ax = plt.subplots(1, figsize=(16, 8), dpi=100)
    channels= [x-0.5 for x in range(len(data)+1)]
    if(len(data)==32):
        histArr = (np.array([data]),np.array([0,1]),np.array(channels))
    else:
        histArr = ((np.array([data])).transpose(),np.array(channels),np.array([0,1]))
    thisHist = hep.hist2dplot(histArr,norm=colors.LogNorm(0.1,2))
    thisHist.cbar.remove()
    if(len(data)==32):
        plt.ylim(len(data)-0.5,-0.5)
    plt.ylabel(" ")
    plt.xlabel(" ")
    #plt.title(name)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.savefig(current_directory+"\ProAnubis_CERN\Figures"+name+".png")
    return current_directory+"\ProAnubis_CERN\Figures"+name+".png"

def stackAndConvert(images, name="testDisplay"):
    #PIL hacking to distort images and put them together to make a primitive replication of the detector
    img = Image.open(images[0])
    total_width = 3*img.size[0]
    max_height = int(4*img.size[1])
    new_im = Image.new('RGB', (total_width, max_height))
    newData = new_im.load()
    x_offset = 0
    y_offset = 6*int(max_height/8.)
    for y in range(max_height):
        for x in range(total_width):
            #newData forms the background of the image, need to set it to all-white to start. Probably some better way to do this?
            newData[x, y] = (255, 255, 255)
    for idx, image in enumerate(images):
        img = Image.open(image)
        img = img.convert("RGBA")
        temp_im = Image.new('RGBA', (3*img.size[0], img.size[1]))
        temp_im.paste(img, (int(img.size[0]/2.),0))
        temp_im = temp_im.transform(temp_im.size, Image.AFFINE, (0.5, 1., 0, 0, 1, 0))
        pixdata = temp_im.load()
        width, height = temp_im.size
        for y in range(height):
            for x in range(width):
                if pixdata[x, y] == (255, 255, 255, 255):
                    #Manually make any white pixel transparent so that they can stack together nicely.
                    pixdata[x, y] = (255, 255, 255, 0)
        new_im.paste(temp_im, (0, y_offset), temp_im)
        y_offset = y_offset-int(max_height/28.)
        if idx == 5 or count==7:
            #Counts from the bottom up, want bigger gaps between the different chambers
            y_offset = y_offset-5*int(max_height/28.)                   
    new_im.save(current_directory+"\ProAnubis_CERN\Figures"+name+".png"+name.strip(" ")+".png", "PNG")


def makeEventDisplay(eventData,name):
    #Expects a single event, divided as [tdc0,tdc2,...,tdc4]
    countOne = countChannels([eventData[0]])
    countTwo = countChannels([eventData[1]])
    countThree = countChannels([eventData[2]])
    countFour = countChannels([eventData[3]])
    countFive = countChannels([eventData[4]])
    singEventPlots = []
    singEventPlots.append(makeSingleLayer(countOne[0:32],"Eta Triplet Low, Three Coincidences Required"))
    singEventPlots.append(makeSingleLayer(countOne[32:96],"Phi Triplet Low, Three Coincidences Required"))
    singEventPlots.append(makeSingleLayer(countOne[96:128],"Eta Triplet Mid, Three Coincidences Required"))
    singEventPlots.append(makeSingleLayer(countTwo[0:64],"Phi Triplet Mid, Three Coincidences Required"))
    singEventPlots.append(makeSingleLayer(countTwo[64:96],"Eta Triplet Top, Three Coincidences Required"))
    singEventPlots.append(makeSingleLayer(countTwo[96:128]+countThree[0:32],"Phi Triplet Top, Three Coincidences Required"))
    singEventPlots.append(makeSingleLayer(countThree[32:64],"Eta Singlet, Three Coincidences Required"))
    singEventPlots.append(makeSingleLayer(countThree[64:128],"Phi Singlet, Three Coincidences Required"))
    singEventPlots.append(makeSingleLayer(countFour[0:32],"Eta Doublet Low, Three Coincidences Required"))
    singEventPlots.append(makeSingleLayer(countFour[32:96],"Phi Doublet Low, Three Coincidences Required"))
    singEventPlots.append(makeSingleLayer(countFour[96:128],"Eta Doublet Top, Three Coincidences Required"))
    singEventPlots.append(makeSingleLayer(countFive[0:64],"Phi Doublet Top, Three Coincidences Required"))
    stackAndConvert(singEventPlots,name)
    for plot in singEventPlots:
        #Remove all the temporary plots. There's probably a better way to move pil images around than making and deleting .png files.
        os.remove(plot)
        
def GetEvent(eventData, num):
    return [eventData[0][num],eventData[1][num],eventData[2][num],eventData[3][num],eventData[4][num]] 

def heatFromFile(dataFile, time=240, name="HeatMap"):
    #Plots heat maps from triggered data, showing the hit rate in each rpc channel. 2D plots designed to replicate RPC layout and channel counting direction.
    thisData = importDatafile(dataFile)
    thisHitData = {}
    addresses = ['ee00','ee01','ee02','ee03','ee04']
    for tdc in range(5):
        thisHitData[addresses[tdc]] = countChannels(thisData[tdc])
    anPlot.makeHitMaps(thisHitData,name,False,unit='hz',time=time)
    
def plotHitCounts(histogram, name):
    #Plot the number of hits per event to determine the mean hits within any given TDC, as well as see the long tails from correlated noise
    fig, ax = plt.subplots(1, figsize=(6, 4), dpi=100)
    lab = hep.atlas.label(com=False,data=True, label="Internal")
    lab[2].set_text(" ")
    hep.histplot(np.histogram(histogram,bins=[x-0.5 for x in range(140)]), label='TDC 0')
    plt.xlabel('TDC Hits')
    plt.ylabel('Events')
    plt.title(name)
    plt.xlim([-0.4,40.5])
    #plt.ylim([1,3000])
    plt.yscale('log')
    plt.savefig(name.strip(" ")+"chanCounts.png")
    return name.strip(" ")+"chanCounts.png"

def plotEventTimes(inputData, name):
    fig, ax = plt.subplots(1, figsize=(6, 4), dpi=100)
    lab = hep.atlas.label(com=False,data=True, label="Internal")
    lab[2].set_text(" ")
    hep.histplot(np.histogram(inputData,bins=[x-0.5 for x in range(1000)]), label='TDC 0')
    plt.xlabel('TDC Hit Times')
    plt.ylabel('Hits')
    plt.title(name)
    #plt.xlim([-0.4,40.5])
    #plt.ylim([1,3000])
    plt.yscale('log')
    plt.savefig(name.strip(" ")+"hitTimes.png")
    return name.strip(" ")+"hitTimes.png"

def divideHitCountsByRPC(data):
    #Divides the number of hits in each channel into individual RPCs
    etaHits = [[],[],[],[],[],[]]
    phiHits = [[],[],[],[],[],[]]
    for event in range(0,len(data[0])):
        tdcCounts = [countChannels([data[tdc][event]]) for tdc in range(5)]
        etaHits[0].append(tdcCounts[0][0:32])
        phiHits[0].append(tdcCounts[0][32:96])
        etaHits[1].append(tdcCounts[0][96:128])
        phiHits[1].append(tdcCounts[1][0:64])
        etaHits[2].append(tdcCounts[1][64:96])
        phiHits[2].append(tdcCounts[1][96:128]+tdcCounts[2][0:32])
        etaHits[3].append(tdcCounts[2][32:64])
        phiHits[3].append(tdcCounts[2][64:128])
        etaHits[4].append(tdcCounts[3][0:32])
        phiHits[4].append(tdcCounts[3][32:96])
        etaHits[5].append(tdcCounts[3][96:128])
        phiHits[5].append(tdcCounts[4][0:64])
    return etaHits,phiHits

def divideEventsByRPC(data):
    #Divides the data into individual RPCs, preserving the initial words within each event
    splitEvents = []
    for event in range(len(data[0])):
        thisEvent = [[] for rpc in range(12)]
        for tdc in range(5):
            for word in data[tdc][event]:
                chan = word>>24
                if(tdc==0):
                    if(chan<32):
                        thisEvent[0].append(word)
                    elif(chan<96):
                        thisEvent[1].append(word)
                    else:
                        thisEvent[2].append(word)
                elif(tdc==1):
                    if(chan<64):
                        thisEvent[3].append(word)
                    elif(chan<96):
                        thisEvent[4].append(word)
                    else:
                        thisEvent[5].append(word)
                elif(tdc==2):
                    if(chan<32):
                        thisEvent[6].append(word)
                    elif(chan<64):
                        thisEvent[7].append(word)
                    else:
                        thisEvent[8].append(word)   
                elif(tdc==3):
                    if(chan<32):
                        thisEvent[8].append(word)
                    elif(chan<64):
                        thisEvent[9].append(word)
                    else:
                        thisEvent[10].append(word)                
                elif(tdc==4):
                    thisEvent[11].append(word)
        splitEvents.append(thisEvent)
    return splitEvents

def countCoincidences(primRpc,SecRpc, winSize=5, minHit=3, maxHit=10):
    #Count events where at least minHit channels have an RPC hit within winSize of each channel, while the primary and secondary RPC has more than minHit and less than maxHit to filter noise events
    #Uses the pre-made hit counts instead of the event words directly
    coincArray = [0 for channel in range(len(primRpc[0]))]
    for idx, event in enumerate(primRpc):
        for channel in range(len(event)):
            nHits = 0
            for itr in range(channel-int(winSize/2.), channel+int(winSize/2.)+1):
                if itr>=0 and itr<len(event):
                    if event[channel]>0:
                        nHits = nHits+1
            if nHits>=minHit and sum(event)<=maxHit:
                if sum(SecRpc[idx])>=minHit and sum(SecRpc[idx])<=maxHit:
                    coincArray[channel]=coincArray[channel]+1
    return coincArray

