from PIL import Image
import h5py
import anubisPlotUtils as anPlot
import json
import numpy as np
import os
import hist as hi
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations
import matplotlib.colors as colors
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'GTK3Agg', etc.
import mplhep as hep
hep.style.use([hep.style.ATLAS])
import sys
import AnalysisToolAnubis as AT
import seaborn as sns
import pandas as pd
import tkinter as tk
from tkinter import ttk
from pandasgui import show
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import convolve2d
from scipy.ndimage import label, find_objects
from scipy.spatial.distance import pdist, squareform
from matplotlib.backends.backend_pdf import PdfPages
import ipywidgets as widgets
from IPython.display import display
import sys
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator



mapping = {
    0: {
        (0, 31): ('rpc1', 'eta'),
        (32, 63): ('rpc1', 'phi1'),
        (64, 95): ('rpc1', 'phi2'),
        (96, 127): ('rpc2', 'eta'),
    },
    1: {
        (0, 31): ('rpc2', 'phi1'),
        (32, 63): ('rpc2', 'phi2'),
        (64, 95): ('rpc3', 'eta'),
        (96, 127): ('rpc3', 'phi1'),
    },
    2: {
        (0, 31): ('rpc3', 'phi2'),
        (32, 63): ('rpc4', 'eta'),
        (64, 95): ('rpc4', 'phi1'),
        (96, 127): ('rpc4', 'phi2'),
    },
    3: {
        (0, 31): ('rpc5', 'eta'),
        (32, 63): ('rpc5', 'phi1'),
        (64, 95): ('rpc5', 'phi2'),
        (96, 127): ('rpc6', 'eta'),
    },
    4: {
        (0, 31): ('rpc6', 'phi1'),
        (32, 63): ('rpc6', 'phi2'),
    },
}

def divideHitCountsByRPC_Timed(data):
    #Divides the number of hits in each channel into individual RPCs
    etaHits = [[],[],[],[],[],[]]
    phiHits = [[],[],[],[],[],[]]
    for event in range(0,len(data[0])):
        tdcCounts = [countChannels_Timed([data[tdc][event]]) for tdc in range(5)]
        etaHits[0].append(tdcCounts[0][0:32]) #Triplet Eta Low
        phiHits[0].append(tdcCounts[0][32:96]) #Triplet Phi low
        etaHits[1].append(tdcCounts[0][96:128]) #Triplet Eta Mid
        phiHits[1].append(tdcCounts[1][0:64]) #Triplet Phi Mid
        etaHits[2].append(tdcCounts[1][64:96]) #Triplet Eta Top
        phiHits[2].append(tdcCounts[1][96:128]+tdcCounts[2][0:32]) #Triplet Phi Top
        etaHits[3].append(tdcCounts[2][32:64])#Singlet Eta
        phiHits[3].append(tdcCounts[2][64:128])#Singlet Phi
        etaHits[4].append(tdcCounts[3][0:32])#Double Eta low
        phiHits[4].append(tdcCounts[3][32:96])#Double Phi Low
        etaHits[5].append(tdcCounts[3][96:128])#Doublet Eta top
        phiHits[5].append(tdcCounts[4][0:64])#Doublet Phi top
    return etaHits,phiHits

def countChannels_Timed(events):
    #Expects events from one TDC, counts how many hits each channel has within the event list
    chanCounts = [[] for x in range(128)]
    for event in events:
        for word in event:

            #get time of event
            time = word&0xfffff

            try:
                #Append time of event to specific channel triggered to get event.
                chanCounts[(word>>24)&0x7f].append(time)
            except:
                print(word>>24)
    return chanCounts

def assign_to_coordinates(tdc, channel, mapping):
    for channel_range, (rpc, side) in mapping.get(tdc, {}).items():
        if channel_range[0] <= channel <= channel_range[1]:
            offset = channel - channel_range[0]
            if side == 'eta':
                return [rpc, 'eta', [0, offset - 31]]
            elif side in 'phi1':
                return [rpc, 'phi', [offset, 0]]
            elif side in 'phi2':
                return[rpc, 'phi', [offset + 32, 0]]
            
    return ['Null', 'Null', [tdc, channel]]  # If no match found
def remake_data(Data, mapping):
    full_data = []
    for tdc in range(0, len(Data)):
        for events in range(0, len(Data[tdc])): 
            for hits in Data[tdc][events]:
                channel = (hits>>24)&0x7f
                coord = assign_to_coordinates(tdc, channel, mapping)
                event_time = hits&0xfffff
                
            
                
                full_data.append({
                    'tdc': tdc,
                    'event number': events,
                    'channel':channel,
                    'rpc number': coord[0],
                    'strip direction': coord[1],
                    'hit coordinate': coord[2],
                    'time': event_time
                })
    df = pd.DataFrame(full_data)
        
    return df
    
    

def initialize_cluster(hit=None):
    """Initialize a new cluster with optional starting hit."""
    cluster = {
        'event_number': hit['event number'] if hit is not None else None,
        'rpc_number': hit['rpc number'] if hit is not None else None,
        'strip_direction': hit['strip direction'] if hit is not None else None,
        'locations': [hit['hit coordinate']] if hit is not None else [],
        'times': [hit['time']] if hit is not None else [],
        'start_time': hit['time'] if hit is not None else None,
        'end_time': hit['time'] if hit is not None else None,
        'size': 0 
    }
    return cluster

def calculate_cluster_metrics_better(df, time_window, min_hits=1, max_hits=10):
    clusters = []

    for direction in ['eta', 'phi']:
        strip_direction_df = df[df['strip direction'] == direction]
        current_cluster = initialize_cluster()
        for index, row in strip_direction_df.iterrows():
            hit = {
                'event number': row['event number'],
                'rpc number': row['rpc number'],
                'strip direction': row['strip direction'],
                'hit coordinate': row['hit coordinate'],
                'time': row['time']
            }

            coordinate_index = 0 if direction == 'phi' else 1
            hit_coordinate = hit['hit coordinate'][coordinate_index]

            if not current_cluster['locations']:
                current_cluster = initialize_cluster(hit)
            else:
                last_hit_coordinate = current_cluster['locations'][-1][coordinate_index]
                if abs(hit_coordinate - last_hit_coordinate) != 1 or hit['time'] - current_cluster['end_time'] > time_window:
                    if len(current_cluster['locations']) >= min_hits:
                        current_cluster['size'] = len(current_cluster['locations'])
                        clusters.append(current_cluster)
                    current_cluster = initialize_cluster(hit)
                else:
                    current_cluster['locations'].append(hit['hit coordinate'])
                    current_cluster['times'].append(hit['time'])
                    current_cluster['end_time'] = hit['time']

        if len(current_cluster['locations']) >= min_hits:
            current_cluster['size'] = len(current_cluster['locations'])
            clusters.append(current_cluster)

    return pd.DataFrame(clusters)

def remove_noisy_strips(row, noisy_strips):

    rpc_number = row['rpc_number']
    strip_direction = row['strip_direction']
    locations = row['locations']

    if rpc_number in noisy_strips:
        if strip_direction in noisy_strips[rpc_number]:
            index = 0 if strip_direction == 'phi' else 1
            for loc in locations:
                if loc[index] in noisy_strips[rpc_number][strip_direction]:
                    return False 
    return True 


def dataframe_to_thisData(df):
    max_tdc = df['tdc'].max()
    max_event_number = df['event number'].max()
    
    reconstructed_data = [[[] for _ in range(max_event_number + 1)] for _ in range(max_tdc + 1)]

    for index, row in df.iterrows():
        tdc = row['tdc']
        event_number = row['event number']
        channel = row['channel']
        time = row['time']
        hits = (channel << 24) | time
        reconstructed_data[tdc][event_number].append(hits)

    return reconstructed_data

def analyze_clusters(df, window_size=5):
    all_clusters = []
    PHI_POINTS = 64
    ETA_POINTS = 32

    for event_number, event_group in df.groupby('event number'):
        for rpc, rpc_group in event_group.groupby('rpc number'):
            for direction, direction_group in rpc_group.groupby('strip direction'):
                direction_group = direction_group.copy()
                direction_group['sort_key'] = direction_group['hit coordinate'].apply(lambda coord: coord[0] if direction == 'phi' else coord[1])
                sorted_direction_group = direction_group.sort_values(by=['sort_key', 'time'])

                clusters = []
                current_cluster = []
                last_hit_time = -1

                for _, hit in sorted_direction_group.iterrows():
                    hit_point = hit['sort_key']
                    hit_time = hit['time']

                    # Start a new cluster or continue the current one
                    if not current_cluster or (
                        (hit_point - current_cluster[-1][0] <= 1 or
                        (direction == 'phi' and abs(hit_point - current_cluster[-1][0]) == PHI_POINTS - 1) or
                        (direction == 'eta' and abs(hit_point - current_cluster[-1][0]) == ETA_POINTS - 1))
                        and hit_time - last_hit_time <= window_size
                    ):
                        current_cluster.append((hit_point, hit_time))
                    else:
                        if current_cluster:
                            clusters.append(process_cluster(current_cluster))
                        current_cluster = [(hit_point, hit_time)]

                    last_hit_time = hit_time

                # Process the last cluster
                if current_cluster:
                    clusters.append(process_cluster(current_cluster))

                # Flatten the clusters into the all_clusters list
                for cluster in clusters:
                    location = (cluster['start'] + cluster['end']) / 2
                    size = cluster['size']
                    event_time = cluster['event_time']
                    all_clusters.append([event_number, rpc, direction, location, size, event_time])

    cluster_df = pd.DataFrame(all_clusters, columns=['Event Number', 'RPC Number', 'Direction', 'Cluster Location', 'Cluster Size', 'Cluster Event Time'])
    return cluster_df

def process_cluster(cluster_points):
    sorted_points = sorted(cluster_points, key=lambda x: x[0])
    start = sorted_points[0][0]
    end = sorted_points[-1][0]
    size = len(sorted_points)
    # Calculate the average event time for the cluster
    event_time = sum(point[1] for point in sorted_points) / size
    return {'start': start, 'end': end, 'size': size, 'event_time': event_time}

def calculate_metric_for_combo(combo, rpc_separation):

    rpc_heights = {'rpc1': 1}  
    for (rpc1, rpc2), separation in rpc_separation.items():
        if rpc1 in rpc_heights and rpc2 not in rpc_heights:
            rpc_heights[rpc2] = rpc_heights[rpc1] + separation
        elif rpc2 in rpc_heights and rpc1 not in rpc_heights:
            rpc_heights[rpc1] = rpc_heights[rpc2] - separation

    locations = np.array([c['location'] for c in combo])
    uncertainties = np.array([c['uncertainty'] for c in combo])
    heights = np.array([rpc_heights[c['rpc']] for c in combo])  

    try:
        slope, intercept = np.polyfit(locations, heights, 1)
        predicted = slope * locations + intercept
        residuals = heights - predicted
        chi2 = np.sum(residuals ** 2 / heights)

    except np.linalg.LinAlgError:
        slope = 0
        intercept = 0
        chi2 = np.inf
    combined_uncertainty = np.sum(uncertainties)

    return slope, intercept, chi2, combined_uncertainty, combo

def analyze_inter_rpc_hit_with_timing_adjusted(df):
    paths = []
    total_events = df['Event Number'].nunique()

    adjusted_muon_speed_cm_ns = 28 

    rpc_separation = {
        ('rpc1', 'rpc2'): 0.1, # Touching
        ('rpc2', 'rpc3'): 0.1, # Touching
        ('rpc3', 'rpc4'): 60,
        ('rpc4', 'rpc5'): 60,
        ('rpc5', 'rpc6'): 0.1  # Touching
    }
    

    for event_number, event_group in df.groupby('Event Number'):
        for direction in ['eta', 'phi']:
            direction_group = event_group[event_group['Direction'] == direction]
            all_clusters = []
            
            unique_rpcs = direction_group['RPC Number'].unique()
            for rpc in unique_rpcs:
                rpc_group = direction_group[direction_group['RPC Number'] == rpc]
                for _, row in rpc_group.iterrows():
                    location_scaling = 3.125 if direction == 'eta' else 2.5
                    cluster_size_scaled = max(row['Cluster Size'] * location_scaling, location_scaling) / 2
                    all_clusters.append({
                        'rpc': rpc,
                        # 'position': No longer used directly in this simplified approach
                        'location': row['Cluster Location'] * location_scaling,
                        'event_time': row['Cluster Event Time'],
                        'uncertainty': cluster_size_scaled,
                        'original_location': row['Cluster Location']
                    })

            combination_metrics = []
            for combo in combinations(all_clusters, 3):
                if len({c['rpc'] for c in combo}) < len(combo):
                    continue
                time_diffs_are_valid = True
                for i in range(len(combo)):
                    for j in range(i+1, len(combo)):
                        rpc_pair = (combo[i]['rpc'], combo[j]['rpc'])
                        if rpc_pair not in rpc_separation and rpc_pair[::-1] not in rpc_separation:
                            continue  
                        separation_distance = rpc_separation.get(rpc_pair) or rpc_separation.get(rpc_pair[::-1])
                    
                        if separation_distance is None:

                            continue  


                        time_diff = abs(combo[i]['event_time'] - combo[j]['event_time'])
                        expected_time_diff = separation_distance / adjusted_muon_speed_cm_ns

                        intrinsic_timing_uncertainty = 5  
                        if separation_distance == 0.1:  
                            if time_diff > intrinsic_timing_uncertainty:
                                time_diffs_are_valid = False
                                break
                        else:  
                            uncertainty_margin = 5  
                            if not (time_diff <= expected_time_diff + uncertainty_margin):
                                time_diffs_are_valid = False
                                break
                    if not time_diffs_are_valid:
                        break

                if not time_diffs_are_valid:
                    continue
                if time_diffs_are_valid: 
                    metric = calculate_metric_for_combo(combo,rpc_separation) 
                    combination_metrics.append(metric) 
                

            if combination_metrics:
                best_combinations = sorted(combination_metrics, key=lambda x: x[2]) 
                selected_combination = best_combinations[0]

                paths.append({
                    'Event Number': event_number,  
                    'Direction': direction,  
                    'Slope': selected_combination[0], 
                    'Intercept': selected_combination[1], 
                    'Used Coordinates': [(c['rpc'], c['original_location'], c['uncertainty']) for c in selected_combination[-1]],
                    'chi2': selected_combination[2],
                    'Combined Uncertainty': selected_combination[3]
                })
    path_df = pd.DataFrame(paths)
    return path_df

def plot_rpc_histograms_with_offsets(dataframe, output_file):
    peak_event_times_dict = {}

    for rpc_number, group in dataframe.groupby('RPC Number'):
        event_times = group['Cluster Event Time'].dropna()
        if not event_times.empty:
            kde = gaussian_kde(event_times)
            kde_x = np.linspace(event_times.min(), event_times.max(), 500)
            kde_y = kde(kde_x)
            peak_idx = np.argmax(kde_y)
            peak_event_time = kde_x[peak_idx]
            peak_event_times_dict[rpc_number] = peak_event_time

    peak_event_times_series = pd.Series(peak_event_times_dict)

    with PdfPages(output_file) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        peak_event_times_series.plot(kind='bar', color='lightblue', edgecolor='black', ax=ax)
        print(peak_event_times_series)
        average_offset = peak_event_times_series.mean()
        ax.axhline(y=average_offset, color='r', linestyle='--', label=f'Average Offset: {average_offset:.2f}')

        ax.set_title('Peak Event Times by RPC with Average Offset')
        ax.set_xlabel('RPC Number')
        ax.set_ylabel('Event Time')
        ax.legend()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        

def plot_rpc_histograms(df, pdf_path):
    with PdfPages(pdf_path) as pdf:
        for rpc in df['rpc_number'].unique():
            for direction in ['eta', 'phi']:
                dataset = df[(df['strip_direction'] == direction) & (df['rpc_number'] == rpc)].reset_index(drop=True)
                all_cluster_midpoints = []
                all_cluster_sizes = []

                for _, row in dataset.iterrows():
                    locs = [loc[0 if direction == 'phi' else 1] for loc in row['locations']]
                    if locs: 
                        midpoint = (min(locs) + max(locs)) / 2
                        all_cluster_midpoints.append(midpoint)
                    sizes = row['size']
                    all_cluster_sizes.append(sizes)

                midpoints_series = pd.Series(all_cluster_midpoints)
                sizes_series = pd.Series(all_cluster_sizes)

                midpoint_frequencies = midpoints_series.value_counts().sort_index()
                size_frequencies = sizes_series.value_counts().sort_index()

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle(f'RPC {rpc} - {direction.upper()} Clusters')

                if not midpoint_frequencies.empty:
                    axs[0].bar(midpoint_frequencies.index, midpoint_frequencies.values, color='skyblue', edgecolor='black')
                axs[0].set_title('Cluster Midpoints')
                axs[0].set_xlabel('Cluster Midpoint Location')
                axs[0].set_ylabel('Number of Occurrences')
                axs[0].set_xlim(0, 64 if direction == 'phi' else -32) 

                if not size_frequencies.empty:
                    axs[1].bar(size_frequencies.index, size_frequencies.values, color='salmon', edgecolor='black')
                axs[1].set_title('Cluster Sizes')
                axs[1].set_xlabel('Cluster Size')
                axs[1].set_ylabel('Number of Occurrences')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close()
                
def plot_rpc_histograms_by_cluster_size(df, pdf_path):
    with PdfPages(pdf_path) as pdf:
        for rpc in df['rpc_number'].unique():
            for direction in ['eta', 'phi']:
                dataset = df[(df['strip_direction'] == direction) & (df['rpc_number'] == rpc)].reset_index(drop=True)
                
                plots_per_page = 4
                total_cluster_sizes = 8
                pages = (total_cluster_sizes + plots_per_page - 1) // plots_per_page
                
                for page in range(pages):
                    fig, axs = plt.subplots(2, 2, figsize=(12, 12))  
                    fig.suptitle(f'RPC {rpc} - {direction.upper()} Clusters by Size')

                    for i in range(plots_per_page):
                        cluster_size = page * plots_per_page + i + 1
                        if cluster_size > total_cluster_sizes:
                            break 

                        subset = dataset[dataset['size'] == cluster_size]
                        all_cluster_midpoints = []

                        for _, row in subset.iterrows():
                            locs = [loc[0 if direction == 'phi' else 1] for loc in row['locations']]
                            if locs: 
                                midpoint = (min(locs) + max(locs)) / 2
                                all_cluster_midpoints.append(midpoint)

                        midpoints_series = pd.Series(all_cluster_midpoints)
                        midpoint_frequencies = midpoints_series.value_counts().sort_index()

                        ax = axs[i // 2, i % 2] 
                        if not midpoint_frequencies.empty:
                            ax.bar(midpoint_frequencies.index, midpoint_frequencies.values, color='skyblue', edgecolor='black')
                        ax.set_title(f'Cluster Size {cluster_size}')
                        ax.set_xlabel('Cluster Midpoint Location')
                        ax.set_ylabel('Number of Occurrences')
                        ax.set_xlim(0, 64 if direction == 'phi' else -32)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    pdf.savefig(fig)
                    plt.close(fig)
                    
def plot_time_differences_and_event_times(df, pdf_path):
    with PdfPages(pdf_path) as pdf:
        for rpc in df['rpc_number'].unique():
            for direction in ['eta', 'phi']:
                filtered_data = df[(df['rpc_number'] == rpc) & (df['strip_direction'] == direction)]

                grouped_by_event = filtered_data.groupby('event_number')
                
                time_differences = []  

                for event_number, event_group in grouped_by_event:
                    event_group_sorted = event_group.sort_values(by='start_time')

                    diffs = event_group_sorted['start_time'].diff().dropna()

                    diffs = diffs[diffs > 0]
                    
                    time_differences.extend(diffs)

                fig, ax = plt.subplots(figsize=(16, 12))
                ax.hist(time_differences, bins=100, color='lightblue', edgecolor='black', log=True)
                ax.set_title(f'Time Differences Between Consecutive Clusters\nRPC: {rpc}, Direction: {direction}')
                ax.set_xlabel('Time Difference (ns)')
                ax.set_ylabel('Number_of_occurance')
                ax.set_xlim(0, 125)  

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

def calculate_metric_for_combo(combo, rpc_heights):
    locations = np.array([c['location'] for c in combo])
    heights = np.array([rpc_heights[c['rpc']] for c in combo])

    try:
        coeffs, cov = np.polyfit(locations, heights, 1, cov=True)
        slope, intercept = coeffs
        slope_error, intercept_error = np.sqrt(np.diag(cov))
        predicted = slope * locations + intercept
        residuals = heights - predicted
        RSS = np.sum(residuals ** 2)
    except np.linalg.LinAlgError:
        slope = np.inf
        intercept = np.inf
        slope_error, intercept_error = np.inf, np.inf
        RSS = np.inf
    
    # combined_uncertainty = np.sum(uncertainties)
    return slope, intercept, slope_error, intercept_error,RSS, combo

def analyze_inter_rpc_hit_with_timing_adjusted(df):
    rpc_time_offsets = {
    ('rpc0', 'eta'): (7.94, 12.48),
    ('rpc0', 'phi'): (-2.38, 13.69),
    ('rpc1', 'eta'): (8.36, 12.22),
    ('rpc1', 'phi'): (-3.79, 13.25),
    ('rpc2', 'eta'): (8.84, 12.56),
    ('rpc2', 'phi'): (-4.35, 13.57),
    ('rpc3', 'eta'): (6.86, 12.41),
    ('rpc3', 'phi'): (-4.3, 13.96),
    ('rpc4', 'eta'): (2.7, 12.37),
    ('rpc4', 'phi'): (-7.89, 13.41),
    ('rpc5', 'eta'): (2.82, 13.05),
    ('rpc5', 'phi'): (9.15, 14.14),
}
    paths = []

    adjusted_muon_speed_cm_ns = 28

    rpc_heights = {
        'rpc0': 0, 
        'rpc1': 0.5, 
        'rpc2': 1.0, 
        'rpc3': 61.5, 
        'rpc4': 121.5, 
        'rpc5': 122.0
    }

    for event_number, event_group in df.groupby('event_number'):
        for direction in ['eta', 'phi']:
            direction_group = event_group[event_group['strip_direction'] == direction]
            all_clusters = []

            unique_rpcs = direction_group['rpc_number'].unique()
            for rpc in unique_rpcs:
                rpc_group = direction_group[direction_group['rpc_number'] == rpc]
                for _, row in rpc_group.iterrows():
                    location_scaling = 3.09375 if direction == 'eta' else 2.8125
                    strip_locations = np.array(row['locations'])
                    non_zero_locations = strip_locations[strip_locations != 0]
                    if non_zero_locations.size > 0:
                        strip_location = non_zero_locations[0] 
                    else:
                        continue 

                    location = strip_location * location_scaling
                    event_time = np.mean(row['times']) - rpc_time_offsets[(rpc, direction)][0]
                    cluster_size_scaled = max(row['size'] * location_scaling, location_scaling) / 2
                    all_clusters.append({
                        'rpc': rpc,
                        'location': location,
                        'event_time': event_time,
                        'uncertainty': cluster_size_scaled,
                        'original_location': strip_location
                    })

            combination_metrics = []
            valid_combinations = [] 
            for n in range(3, 6):
                for combo in combinations(all_clusters, n):
                    metric = calculate_metric_for_combo(combo, rpc_heights)
                    if metric[4] != np.inf:
                        combination_metrics.append(metric)
                        
            # Move filtering logic outside the loop so it's not reset each time
            for combo_metric in combination_metrics:
                combo = combo_metric[-1]
                if len({c['rpc'] for c in combo}) < len(combo):
                    continue

                time_diffs_are_valid = True
                for i in range(len(combo)):
                    for j in range(i + 1, len(combo)):
                        error_window = rpc_time_offsets[(combo[i]['rpc'], direction)][1] + rpc_time_offsets[(combo[j]['rpc'], direction)][1]
                        
                        # Use direct height differences
                        height_diff = abs(rpc_heights[combo[i]['rpc']] - rpc_heights[combo[j]['rpc']])
                        
                        time_diff = abs(combo[i]['event_time'] - combo[j]['event_time'])
                        expected_time_diff = height_diff / adjusted_muon_speed_cm_ns

                        uncertainty_margin = 5
                        # Use the expected time difference with the error window and uncertainty margin for validation
                        if not (time_diff <= expected_time_diff + error_window + uncertainty_margin):
                            time_diffs_are_valid = False
                            break
                    if not time_diffs_are_valid:
                        break

                if time_diffs_are_valid:
                    valid_combinations.append(combo_metric)

            for valid_combination in valid_combinations:
                paths.append({
                    'Event Number': event_number,
                    'Direction': direction,
                    'Slope': valid_combination[0],
                    'Intercept': valid_combination[1],
                    'Slope_error': valid_combination[2],
                    'Intercept_error': valid_combination[3],
                    'Used Coordinates': [(c['rpc'], c['original_location'], c['event_time']) for c in valid_combination[-1]],
                    'RSS': valid_combination[4],
                })

    path_df = pd.DataFrame(paths)
    return path_df