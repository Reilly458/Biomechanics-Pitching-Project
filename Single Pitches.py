import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from matplotlib.backends.backend_pdf import PdfPages

pdf_name = "Sophie_single_pitch.pdf"

# Import csv data 
data = pd.read_csv("C:/Users/Jennifer/Documents/EPHE 341/Project/Full Data Sets/Project times - Sophie.csv")

size_wrist = 68504       #Number of arm cells with data
size_chest = 70320       #Number of chest cells with data
raw_wrist = data['Wrist IMU (Gyroscope) z'].iloc[:size_wrist]
raw_chest = data['Chest IMU (linear acceleration) z'].iloc[:size_chest]
arm_length = data['Arm Length [m]'].iloc[0]
force = data['Parallel Force (N)'].iloc[:5266]
force_time = data['Force Plate (Parallel force) Time (s)'].iloc[:5266]
wrist_samples = data['Wrist IMU (Gyroscope) timestamp'].iloc[:size_wrist]
speed = data['Fastball Speed [mph]'].iloc[:7]
speed = speed.astype(float)
change = data['Change up Speed [mph]'].iloc[2:7]
change = change.astype(float)
stride = data['Fastball Stride [in]'].iloc[:7]
stride = stride.astype(float)
c_stride = data['Change up Stride [in]'].iloc[2:7]
C_stride = c_stride.astype(float)
peak_vel = []

sample_rate_FP = 500     #Enter your FP sample rate here, default is 500
sample_rate_IMU = 208    #Enter your IMU sammple rate here, defauly is 208
cutoff_freq = 4
chest_cutoff_freq = 3

def plots(x, y, title, xaxis, yaxis, label_1):
    fig, ax = plt.subplots()
    ax.plot(x, y, label= label_1 )
    ax.set_title(title)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    fig.set_size_inches(15,10, forward=True)
    ax.legend(bbox_to_anchor = (1.13, 0.55), loc='center right')
    fig.set_dpi = 300 
    pdf.savefig(fig)
    plt.show()
    
def plot_comp(x1, y1, title, xaxis, yaxis, label_1, x2, y2, label_2):
    fig, ax = plt.subplots()
    ax.plot(x1, y1, label= label_1 )
    ax.plot(x2, y2, label= label_2, color = 'red' )
    ax.set_title(title)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    fig.set_size_inches(15,10, forward=True)
    ax.legend(bbox_to_anchor = (1.13, 0.55), loc='center right')
    fig.set_dpi = 300 
    pdf.savefig(fig)
    plt.show()

# Define low-pass filter
def highpass_filter(data, cutoff_freq, sample_rate_IMU, order=4):
    nyquist = 0.5 * sample_rate_IMU
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)

def lowpass_filter(data, cutoff_freq, sample_rate_FP, order=4):
    nyquist = 0.5 * sample_rate_FP
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

#Filter wrist data with highpass filter
wrist_filtered = highpass_filter(raw_wrist, cutoff_freq, sample_rate_IMU)

# Negate Filtered Wrist IMU z
dataNeg = wrist_filtered*-1 

''' Calculate tangential velocity by multiplying negated Wrist IMU x 
    by arm length in m/s'''
#wrist = dataNeg *arm_length    #For inverted wrist data
wrist = wrist_filtered * arm_length    #For non-inverted wrist data
#wrist = raw_wrist * arm_length

time = (data['Chest IMU (linear acceleration) timestamp']-data['Chest IMU (linear acceleration) timestamp'].iloc[0]) / 208
time = time.iloc[:size_chest]
wrist_time = (data['Wrist IMU (Gyroscope) timestamp']-data['Wrist IMU (Gyroscope) timestamp'].iloc[0])/208
wrist_time = wrist_time.iloc[:size_wrist]

raw_chest_filtered = highpass_filter(raw_chest, chest_cutoff_freq, sample_rate_IMU)
chest = integrate.cumulative_trapezoid(raw_chest_filtered, x=None, dx= 1/sample_rate_IMU, initial =0)

tot_vel = np.round(wrist[31500:31900]+chest[32300:32700],2)
tot_vel_c = np.round(wrist[52050:52450]+ chest[57175:57575],2)

# Plotting
with PdfPages(pdf_name) as pdf: 
        peaks, _ = find_peaks(tot_vel, height=150)   # Peak detection
        valleys, _ = find_peaks(-tot_vel, height=75)  # Valley detection
        
        start = 31500
        end_quiet_stance = start + valleys[0]
        load_end = start + peaks[0]  
        wind_up = start + 193         # End of wind-up
        drive_end = start + valleys[1]           # End of stride
        stride_end = start + peaks[1]              
        release = start + 208               # Release (peak velocity)
        follow_through_start = start + valleys[2] # Start of follow-through
        

        plots(wrist_samples, data['Wrist IMU (Gyroscope) z'].iloc[:size_wrist]
             ,'Raw Wrist IMU Velocity', 'Sample #', 'Wrist IMU angular velocity in z [rad/s]'
             , '')
        plots(wrist_samples[31000:32500], data['Wrist IMU (Gyroscope) z'].iloc[31000:32500]
              ,'Raw Wrist IMU Velocity Single Pitch', 'Sample #', ('Wrist IMU angular velocity in z [rad/s]')
              , 'Fastball #3')
       
        plots(wrist_samples, wrist_filtered, 'Raw Filtered Wrist IMU Velocity', 'Sample #'
              , 'Wrist IMU angular velocity in z [rad/s]', '')
        plots(wrist_samples[31550:31850], wrist_filtered[31550:31850], "Raw Filtered Wrist IMU Velocity Single Pitch",
              "Samples (#)", 'Wrist IMU (Gyroscope) z [rad/s]', "Fastball #3")
        plot_comp(wrist_samples[31550:31850], wrist_filtered[31550:31850], "Raw Filtered Wrist IMU Velocity Single Pitch",
              "Samples (#)", 'Wrist IMU (Gyroscope) z [rad/s]', "Fastball #3", wrist_samples[31550:31850], wrist_filtered[52100:52400],
              'Change Up #2')
      
        plots(wrist_time, wrist, 'Filtered Wrist Tangential Velocity', "Time (s)", 'Wrist IMU (Gyroscope) z [m/s]', "")
        plots(wrist_time[31550:31850], wrist[31550:31850], 'Filtered Wrist Tangential Velocity Single pitch', "Time (s)"
              , 'Wrist IMU (Gyroscope) z [m/s]', 'Fastball #3')
        plot_comp(wrist_time[31550:31850], wrist[31550:31850], 'Filtered Wrist Tangential Velocity Single pitch', "Time (s)"
              , 'Wrist IMU (Gyroscope) z [m/s]', 'Fastball #3', wrist_time[31550:31850], wrist[52100:52400], 'Change Up #2')
       
        plots(time, raw_chest, 'Raw Chest Linear Acceleration', "Time (s)", 'Chest IMU (Gyroscope) z [m/s^2]', "")
        plots(time[32500:32600], raw_chest[32500:32600], 'Raw Chest Linear Acceleration Single Pitch', "Time (s)", 
              'Chest IMU (Gyroscope) z [m/s^2]', 'Fastball #3')
        
        plots(time, raw_chest_filtered, 'Raw Filtered Chest Linear Acceleration', "Time (s)", 'Chest IMU (Gyroscope) z [m/s^2]', "")
        plots(time[32300:32700], raw_chest_filtered[32300:32700], 'Raw Filtered Chest Linear Acceleration Single Pitch'
              , "Time (s)", 'Chest IMU (Gyroscope) z [m/s^2]', 'Fastball #3')
        plot_comp(time[32300:32700], raw_chest_filtered[32300:32700], 'Raw Filtered Chest Linear Acceleration Single Pitch'
              ,"Time (s)", 'Chest IMU (Gyroscope) z [m/s^2]', 'Fastball #3', time[32300:32700], raw_chest_filtered[57175:57575], 'Change Up #2')

        
        plots(time, chest, 'Chest Linear Velocity', "Time (s)", 'Chest IMU (Gyroscope) Velocity z [m/s]', '')
        plots(time[32300:32700], chest[32300:32700], 'Chest Linear Velocity Single Pitch', "Time (s)", 
              'Chest IMU (Gyroscope) Velocity z [m/s]', 'Fastball #3')
        plot_comp(time[32300:32700], chest[32300:32700], 'Chest Linear Velocity Single Pitch'
              ,"Time (s)", 'Chest IMU (Gyroscope) z Velocity[m/s]', 'Fastball #3', time[32300:32700],chest[57175:57575], 'Change Up #2')

        plot_comp(wrist_time[31500:31900], tot_vel, 'Total Velocity Single Pitch', 'Time (s)', "velocity [m/s]", 'Fastball #3',
                  wrist_time[31500:31900], tot_vel_c, "Change Up#2")
    
        fig6, ax6 = plt.subplots()
        peaks_v, _ = find_peaks(tot_vel, distance = 3000, height = 600)
        ax6.plot(wrist_time[start:end_quiet_stance+1], tot_vel[:valleys[0]+1], color= 'skyblue' , label= "Quiet Stance", linewidth = 4)
        ax6.plot(wrist_time[end_quiet_stance:load_end+1], tot_vel[valleys[0]:peaks[0]+1], color='#00fa3b',  label='Load', linewidth = 4)
        ax6.plot(wrist_time[load_end:wind_up], tot_vel[peaks[0]:193], color='#2327de', label='Wind-up', linewidth = 4)
        ax6.plot(wrist_time[wind_up-1:drive_end+1], tot_vel[192:valleys[1]+1] , color='green',  label='Stride', linewidth = 4)
        ax6.plot(wrist_time[drive_end:stride_end+1], tot_vel[valleys[1]:peaks[1]+1], color='purple',  label='Acceleration', linewidth = 4)
        ax6.plot(wrist_time[stride_end:release+1], tot_vel[peaks[1]:209], color='orange', label='Plant', linewidth = 4)
        ax6.plot(wrist_time[release:follow_through_start+1], tot_vel[208:valleys[2]+1], color='red', label='Follow Through', linewidth = 4)
        ax6.plot(wrist_time[follow_through_start:31900], tot_vel[valleys[2]:], color='skyblue',  label='Quiet Stance', linewidth = 4)
        ax6.plot(wrist_time[31692], tot_vel[192] ,'o', markersize = 10, color = 'teal', label = 'Push Off')
        ax6.set_title('Single Fastball Pitch Velocity Profile', fontsize = 20, fontweight= "bold")
        ax6.set_xlabel("Time (s)", fontsize= 16, fontweight= "bold")
        ax6.set_ylabel('Total Velocity [m/s]', fontsize = 16,fontweight= "bold")
        ax6.annotate(tot_vel[peaks_v][0], (wrist_time[31705], tot_vel[peaks_v][0]), ha='left', va='bottom', fontsize = 14, fontweight = 'bold', color = 'black') 
        ax6.plot(wrist_time[31705] , tot_vel[peaks_v], 'o', markersize =10, color = '#780401', label= "Peak Velocity")
        ax6.plot(wrist_time[release], tot_vel[208], 'o', color = '#fe7f0e', markersize =10, label= "Release")
        ax6.set_xlim([752,757.5])
        fig6.set_size_inches(17,10, forward=True)
        fig6.set_dpi = 300 
        ax6.legend(bbox_to_anchor = (1, 0.55), loc='best')
        pdf.savefig(fig6)
        plt.show()
        
        plots(force_time, force, "Raw Drive Force", "Time (s)", 'Force [N]', 'Fastball')
        plots(force_time[1350:1600], force[1350:1600], "Raw Drive Force Single Pitch", "Time (s)", 'Force [N]', 'Fastball')
        filtered_force = lowpass_filter(force, 50, sample_rate_FP)
        plots(force_time, filtered_force, "Filtered Drive Force", "Time (s)", 'Force [N]', 'Fastball')
        plots(force_time[1350:1600], filtered_force[1350:1600], "Raw Drive Force Single Pitch", "Time (s)", 'Force [N]', 'Fastball')

        ft = np.array(force_time[1350:1600])
        ff = filtered_force[1350:1600]
 
        f_peaks, _ = find_peaks(ff, height=.5)   # Peak detection
        f_neg_peaks, _ = find_peaks(ff[59:185], distance = 60)
        f_neg_peak = f_neg_peaks + 59
        f_valleys, _ = find_peaks(-ff, height=68, distance = 10)  # Valley detection
        
        flight = f_peaks[0]
        start_quiet = f_valleys[1]
        end_quiet = f_neg_peak[1]
        load = f_valleys[6]
        wind_up = f_peaks[2]
        drive = f_peaks[4]
        propulsion = drive + np.where(ff[drive:] < 0)[0]

        fig, ax = plt.subplots()
        ax.plot(ft[:flight], ff[:flight], color= 'skyblue', label='Flight', linewidth =4)
        ax.plot(ft[flight:start_quiet], ff[flight:start_quiet],color= 'yellow', label='Step On', linewidth =4)
        ax.plot(ft[start_quiet:end_quiet], ff[start_quiet:end_quiet],color= 'red', label='Quiet Stance', linewidth =4)
        ax.plot(ft[end_quiet:load+1], ff[end_quiet:load+1],color= 'green', label='Load', linewidth =4)
        ax.plot(ft[load:wind_up], ff[load:wind_up], color= 'orange', label= 'Wind-up', linewidth =4)
        ax.plot(ft[wind_up:drive], ff[wind_up:drive], color= 'teal', label= 'Drive', linewidth =4)
        ax.plot(ft[drive-1:propulsion[0]+1], ff[drive-1:propulsion[0]+1], color= 'blue', label= 'Push off', linewidth =4)
        ax.plot(ft[propulsion[0]:], ff[propulsion[0]:], color= 'skyblue', label= 'Flight', linewidth =4)
        ax.annotate(np.round(ff[f_peaks[4]],2), (ft[f_peaks[4]],ff[f_peaks[4]]),ha='left', va='bottom', fontsize = 14, fontweight = 'bold', color = 'black')
        ax.plot(ft[f_peaks[4]], ff[f_peaks[4]],'o', markersize =10, color = 'blue', label= "Peak Parallel Force")
        ax.set_title('Single Pitch Fastball Force', fontsize = 20, fontweight = "bold")
        ax.set_xlabel("Time (s)", fontsize = 16, fontweight = "bold")
        ax.set_ylabel('Parallel Force [N]', fontsize = 16, fontweight = "bold")
        fig.set_size_inches(12,8, forward=True)
        fig.set_dpi = 300 
        ax.legend(bbox_to_anchor = (1.16, 0.55), loc='best')
        pdf.savefig(fig)
        plt.show()
