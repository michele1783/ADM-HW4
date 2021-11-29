import numpy as np      
import matplotlib.pyplot as plt 
import scipy.io.wavfile 
import subprocess
import librosa
import librosa.display
import IPython.display as ipd

from pathlib import Path, PurePath   
from tqdm.notebook import tqdm

def convert_mp3_to_wav(audio:str) -> str:  
    """Convert an input MP3 audio track into a WAV file.
    Args:
        audio (str): An input audio track.
    Returns:
        [str]: WAV filename.
    """
    if audio[-3:] == "mp3":
        wav_audio = audio[:-3] + "wav"
        if not Path(wav_audio).exists():
                subprocess.check_output(f"ffmpeg -i {audio} {wav_audio}", shell=True)
        return wav_audio
    
    return audio

def plot_spectrogram_and_peaks(track:np.ndarray, sr:int, peaks:np.ndarray, onset_env:np.ndarray) -> None:
    """Plots the spectrogram and peaks 
    Args:
        track (np.ndarray): A track.
        sr (int): Aampling rate.
        peaks (np.ndarray): Indices of peaks in the track.
        onset_env (np.ndarray): Vector containing the onset strength envelope.
    """
    times = librosa.frames_to_time(np.arange(len(onset_env)),
                            sr=sr, hop_length=HOP_SIZE)

    plt.figure()
    ax = plt.subplot(2, 1, 2)
    D = librosa.stft(track)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                            y_axis='log', x_axis='time')
    plt.subplot(2, 1, 1, sharex=ax)
    plt.plot(times, onset_env, alpha=0.8, label='Onset strength')
    plt.vlines(times[peaks], 0,
            onset_env.max(), color='r', alpha=0.8,
            label='Selected peaks')
    plt.legend(frameon=True, framealpha=0.8)
    plt.axis('tight')
    plt.tight_layout()
    plt.show()


def load_audio_peaks(audio, offset, duration, hop_size):
    """Load the tracks and peaks of an audio.
    Args:
        audio (string, int, pathlib.Path or file-like object): [description]
        offset (float): start reading after this time (in seconds)
        duration (float): only load up to this much audio (in seconds)
        hop_size (int): the hop_length
    Returns:
        tuple: Returns the audio time series (track) and sampling rate (sr), a vector containing the onset strength envelope
        (onset_env), and the indices of peaks in track (peaks).
    """
    try:
        track, sr = librosa.load(audio, offset=offset, duration=duration)
        onset_env = librosa.onset.onset_strength(track, sr=sr, hop_length=hop_size)
        peaks = librosa.util.peak_pick(onset_env, 10, 10, 10, 10, 0.5, 0.5)
    except Error as e:
        print('An error occurred processing ', str(audio))
        print(e)

    return track, sr, onset_env, peaks


# We use this function to clean our dataset, in particular to fill the NaN values.
def fill_nan(df):
    for i in df.columns:
        if df[i].isnull().any() == True:
            # Check for columns with NaN values
            if is_numeric_dtype(df[i]) == True:
                # If is a numeric column fillna with the mean of the column
                df[i] = df[i].fillna(df[i].mean())
            elif is_string_dtype(df[i]) == True:
                # If is a string column fill na with an empty string
                df[i] = df[i].fillna("")
    
    print('All NaN filled!')  
    
# We want to consider only non-object types of variable to perform PCA.
def remove_object_dtype(df):
    new_df = df.select_dtypes(exclude = 'object') 
    return new_df

# We leverage the StandardScaler module from sklearn.preprocessing
def scale_f(df):
    scaler = preprocessing.StandardScaler()
    # We don't want to consider 'track_id' - the first col - because we need it for the merge
    new_df = pd.DataFrame(scaler.fit_transform(df[df.columns[1:]].values), columns = df.columns[1:])
    # Add track_id
    final_df = pd.concat([df[df.columns[:1]], new_df], axis = 1)
    
    return final_df

def K_Means(K, data):
    '''
    Input:
    K = number of clusters (integer)
    data = dataframe
    '''
    # Creating an array with the values of the dataframe
    array = np.array(data).reshape(data.shape[0], data.shape[1])
    
    n = array.shape[0]     # Number of rows
    m = array.shape[1]     # Number of columns
    
    # Picking randomly the first centroids
    centroids = array[np.random.choice(n, size = K, replace = False)]
    # Initialize the array of the previous centroids at each step of the algo
    prev_centroids = np.zeros((n,K)) 
    
    iterations = 0
    # Until the centroids don't change or the iterations are maximum n (the number of observation)
    while iterations != 3 or np.array_equal(centroids, prev_centroids) == False:
        # Saving the previous values of the cluster for the next while loop
        prev_centroids = centroids    
        euc_dis = np.zeros((n,K))      # Initialize the euclidean distance array
        clus = defaultdict(list)       # Collecting the clusters in a dictionary    
        clusters = []                  #list where I put the number of cluster for each element 
        
        for i in range(n):
            for j in range(K):
            # Computing the euclidean distance from each point to each centroid
                euc_dis[i][j] += linalg.norm(array[i]-centroids[j])
            
            # List containing the cluster to which each observation belongs to
            # Find the minimum distance between each observations and the clusters
            #I have to put [0][0] to extract the index where the obeservation has the min distance
            # We add '1' to get clusters that start from 1
            clusters.append(np.where(euc_dis[i] == min(euc_dis[i]))[0][0]+1)
            
            # Dictionary that maps each cluster to the observations that belong to it
            clus[clusters[i]].append(i)
        
        for k in range(K):
            for j in range(m):
                values = []
                for i in clus[k+1]: # Clusters start from value '1'
                    # Taking the values of the observation belonging to the i-th cluster
                    values.append(array[i][j])
                    
                # Computing the mean for each cluster and taking it as new centroid
                centroids[k][j] = np.mean(values)
        
        iterations += 1  # Pass at the next iteration
        
    return clusters, euc_dis, clus

def elbow(data, k):
    '''
    Input:
    data = dataframe
    k = max number of clusters (integer)
    '''
    cost = [] 
    
    for i in range(2,k): # The number of clusters analyzed goes from 2 to k
        # Run the implemented algorithm
        euc_dis, dictionary = K_Means(i, data)
        
        # Append the cost of the K-Means algorithm for each number of clusters
        #since we have to summation in the formula seen during lesson for the cost
        #it's faster to take all the minima distances from the nearest cluster for each obeservation
        #and sum all together
        cost.append(sum(euc_dis.min(axis=1)**2))
    
    x = list(range(2,k)) # Set the x-axis for the plot
    for i in range(1,len(cost)):
         # I want to compute the differences between contiguous cost 
         #to retrieve the optimal key 
         # I will not pick neither the minimum or the maximum difference
        diff = cost[i]-cost[i-1]
    # Plot the Elbow method
    plt.plot(x, cost, color = "orchid")
    plt.title("Elbow method")
    plt.xlabel("K")
    plt.ylabel("Cost")
    plt.show()
    
    return diff

# Create new random reference set
def gap_stat(data, k):
    
    randomReference = np.random.random_sample(size=data.shape)
    cost = []
    cost_r = []
    gaps = [] 
    for i in range(2,k):
        euc_dis_r, dictionary_r = K_Means(i, randomReference)
        euc_dis, dictionary = K_Means(i, data)
        cost_ = sum(euc_dis.min(axis=1)**2)
        cost.append(cost_)
        cost_r_ = sum(euc_dis_r.min(axis=1)**2)
        cost_r.append(cost_r_)
        gap = np.log(np.mean(cost_r)) - np.log(cost_)
        gaps.append(gap)
    plt.plot(list(range(2,k)), gaps, linestyle='--', marker='o', color='r')
    plt.title("Gap statistics")
    plt.xlabel("K")
    plt.ylabel("Gap Stistics")
    plt.show()
    maxx = max(gaps)
    return(gaps.index(maxx) + 2)

    


        
