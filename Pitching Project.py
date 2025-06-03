import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt

# Import csv data 
data = pd.read_csv("C:/Users/Jennifer/Downloads/Project times - Sophie.csv")

raw_wrist = data['Wrist IMU (Gyroscope) z']
arm_length = data['Arm Length [m]'].iloc[0]
force = data['Parallel Force (N)']
force_time = data['Force Plate (Parallel force) Time (s)']
wrist_samples = data['Wrist IMU (Gyroscope) timestamp'].iloc[:68504]
raw_chest = data['Chest IMU (linear acceleration) z']
speed = data['Fastball Speed [mph]'].iloc[:7]
change = data['Change up Speed [mph]'].iloc[2:7]
stride = data['Fastball Stride [in]'].iloc[:7]
c_stride = data['Change up Stride [in]'].iloc[2:7]
peak_force = [0]
peak_vel = []

sample_rate_FP = 500     #Enter your FP sample rate here, default is 500
sample_rate_IMU = 208    #Enter your IMU sammple rate here, defauly is 208
cutoff_freq = 2

# Define low-pass filter
def highpass_filter(data, cutoff_freq, sample_rate_IMU, order=4):
    nyquist = 0.5 * sample_rate_IMU
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)

#Filter wrist data with highpass filter
wrist_filtered = highpass_filter(raw_wrist.iloc[:68504], cutoff_freq, sample_rate_IMU)

# Negate Filtered Wrist IMU z
dataNeg = wrist_filtered*-1 

''' Calculate tangential velocity by multiplying negated Wrist IMU x 
    by arm length in m/s'''
wrist = dataNeg *arm_length  
"""Use wrist declaration below if not inverting raw data, comment out line above
   may need to adjust peak height and distance parameteres"""
#wrist = wrist_filtered * arm_length 
wrist_peaks, _ = find_peaks(wrist, height=250, distance = 170)   
#print("Wrist peaks")
#print(wrist[wrist_peaks])


''' Calculate time values in seconds from timestamp with 208 sample rate
    and zeros the time'''
time = (data['Chest IMU (linear acceleration) timestamp']-data['Chest IMU (linear acceleration) timestamp'].iloc[0]) / 208
wrist_time = (data['Wrist IMU (Gyroscope) timestamp']-data['Wrist IMU (Gyroscope) timestamp'].iloc[0])/208
wrist_time = wrist_time.iloc[:68504]
''' Calculate chest velocity from acceleration v = at in m/s
    You may need to adjust the offset on the raw_chest value'''

raw_chest_filtered = highpass_filter(raw_chest, cutoff_freq, sample_rate_IMU)
chest = integrate.cumulative_trapezoid(raw_chest_filtered, x=None, dx= 1/sample_rate_IMU, initial =0)
chest_peaks, _ = find_peaks(chest, height = 0.25, distance = 170)
#print("Chest peaks")
#print(chest[chest_peaks])

''' Calculate total velocity = wrist + chest'''
tot_vel = wrist 
for i in range(len(chest_peaks)):
    tot_vel[wrist_peaks[i]] += wrist[wrist_peaks[i]] + chest[chest_peaks[i]]
    print(tot_vel[wrist_peaks[i]]," = ", wrist[wrist_peaks[i]], " + ",chest[chest_peaks[i]] )
tot_vel = np.round(tot_vel, 2)
#filtered_vel = highpass_filter(tot_vel.iloc[:68503], cutoff_freq, sample_rate_IMU)

# Plotting
'''Check the data looks oriented correctly, if not uncomment the dataNeg wrist
   and comment out the calculation that doesn't use the negated data''' 
fig1, ax1 = plt.subplots()
ax1.plot(wrist_samples, data['Wrist IMU (Gyroscope) z'].iloc[:68504])
ax1.set_title('Raw Wrist IMU Velocity')
ax1.set_xlabel('Sample #')
ax1.set_ylabel('Wrist IMU angular velocity in z [rad/s]')
fig1.set_size_inches(15,10, forward=True)
fig1.set_dpi = 300 
plt.show()

fig01, ax01 = plt.subplots()
ax01.plot(wrist_samples, wrist_filtered)
ax01.set_title('Raw Filtered Wrist IMU Velocity')
ax01.set_xlabel('Sample #')
ax01.set_ylabel('Wrist IMU angular velocity in z [rad/s]')
fig01.set_size_inches(15,10, forward=True)
fig01.set_dpi = 300 
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(wrist_samples, dataNeg, color= 'teal' , label= "Wrist IMU")
ax2.set_title("Filtered Wrist IMU Velocity Negated")
ax2.set_xlabel("Samples (#)")
ax2.set_ylabel('Wrist IMU (Gyroscope) z [rad/s]')
fig2.set_size_inches(15,10, forward=True)
fig2.set_dpi = 300  
plt.show()

fig3, ax3 = plt.subplots()
ax3.plot(wrist_time.iloc[:68504], wrist, color= 'blue' , label= "Wrist Tangential Velocity")
ax3.set_title('Filtered Wrist Tangential Velocity')
ax3.set_xlabel("Samples (#)")
ax3.set_ylabel('Wrist IMU (Gyroscope) z [m/s]')
fig3.set_size_inches(15,10, forward=True)
fig3.set_dpi = 300  
plt.show()

fig4, ax4 = plt.subplots()
ax4.plot(time, raw_chest, color= 'orange' , label= "Chest IMU")
ax4.set_title('Raw Chest Linear Acceleration')
ax4.set_xlabel("Time (s)")
ax4.set_ylabel('Chest IMU (Gyroscope) z [m/s^2]')
fig4.set_size_inches(15,10, forward=True)
fig4.set_dpi = 300  
plt.show()

fig10, ax10 = plt.subplots()
ax10.plot(time, chest, color= 'red' , label= "Chest IMU")
ax10.set_title('Chest Linear Velocity')
ax10.set_xlabel("Time (s)")
ax10.set_ylabel('Chest IMU (Gyroscope) Velocity z [m/s]')
fig10.set_size_inches(15,10, forward=True)
fig10.set_dpi = 300  
plt.show()

fig5, ax5 = plt.subplots()
"""The line below finds the peak values and puts them in an array.
   You may need to adjust the height threshold or add a vertical
   threshold to capture the desired peaks."""
peaks, _ = find_peaks(force, height=150)
ax5.plot(force_time, force, color= 'green' , label= "Parallel Force (N)")
ax5.set_title('Raw Force', fontsize = 16, fontweight = "bold")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel('Parallel Force [N]')
'''The next 3 lines prints the peak values to the plot'''
for i, txt in enumerate(peaks):
    ax5.annotate(force[txt], (force_time[txt], force[txt]), ha='center', va='bottom', fontsize = 14, fontweight = 'bold') 
ax5.plot(force_time[peaks] , force[peaks], 'o', markersize =6, color = 'red')
ax5.set_xlim([-1,260])
fig5.set_size_inches(15,10, forward=True)
fig5.set_dpi = 300 
plt.show()
peak_force [1:]= force[peaks] 
print("Peak Force Array")
print(peak_force)


"""Stop here and analyze graphs and change plot settings as needed. 
   Once the graphs look right, uncomment the section below. The plots
   below create the correlation plots and trendlines. Some values will
   need to be adjusted from file to file"""
"""------------------------------------------------------------------------- add 3 apostropes here to uncomment below """

'''Plots total velocity'''   
fig6, ax6 = plt.subplots()
peaks_v, _ = find_peaks(tot_vel, distance = 200, height = 500)
ax6.plot(wrist_time, tot_vel, color= 'black' , label= "Total Velocity")
ax6.set_title('Total Velocity', fontsize = 20)
ax6.set_xlabel("Time (s)", fontsize= 16)
ax6.set_ylabel('Total Velocity [m/s]', fontsize = 16)
ax6.set_xlim([0,1600])
peak_values = tot_vel[peaks_v]
'''This filters out a peak we don't want. Use print(tot_vel[peaks_v[]]) to get
   the value of the peak you want ignored and plug into filtered_peaks line.
   If you don't need to ignore peaks, comment out.'''
''''-------------------------------------------------------------------------'''   
#print("Filter peak: ", tot_vel[peaks_v[1]]) #Uncomment to print filtered peak value
filtered_peaks = peaks_v[peak_values != 1039.58]
for tx in filtered_peaks:
    ax6.annotate(tot_vel[tx], (wrist_time[tx], tot_vel[tx]), ha='center', va='bottom', fontsize = 14, fontweight = 'bold') 
ax6.plot(wrist_time[filtered_peaks] , tot_vel[filtered_peaks], 'o', markersize =6, color = 'red')
peak_vel = tot_vel[filtered_peaks]
''''-------------------------------------------------------------------------'''
#peak_vel = tot_vel[peaks_v]     #Comment out if filtering a peak out
fig6.set_size_inches(15,8, forward=True)
fig6.set_dpi = 300  
plt.show()
print("Peak Velocity Array")
print(peak_vel)

'''Plots arm velocity and speed'''
fig7, ax7 = plt.subplots()
ax7.scatter(speed[:2], peak_vel[:2], marker = "*", s =90, color = 'red')
ax7.scatter(speed[2:7], peak_vel[2:7], s = 60, color = 'red', label = 'Fastball')
ax7.annotate("No Stride", (speed[0],peak_vel[0]), ha='center', va='bottom', fontsize=14)
ax7.annotate("No Arm", (speed[1],peak_vel[1]), ha='center', va='bottom', fontsize=14)
ax7.set_title("Arm Speed vs Pitch Speed", fontweight= "bold", fontsize = 16)
ax7.set_xlabel('Pitch Speed [mph]', fontweight= "bold")
ax7.set_ylabel('Arm Velocity [m/s]',fontweight= "bold")
plt.xlim([34,55])
a, b = np.polyfit(speed[2:7], peak_vel[2:7], 1)
ax7.plot(speed[2:7], a*speed[2:7]+b)
x_mid = np.mean(speed[2:7].values)
y_mid = a * x_mid + b
ax7.text(x_mid-5.5, y_mid, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)
'''Uncomment next 6 lines for change up data'''
'''---------------------------------------------------------------'''
ax7.scatter(change, peak_vel[7:], s=60, label = 'Change Up')
a, b = np.polyfit(change, peak_vel[7:], 1)
ax7.plot(change, a*change+b)
x_mid = np.mean(change)
y_mid = a * x_mid + b
ax7.text(x_mid-2, y_mid-25, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)
ax7.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
''''------------------------------------------------------------'''
fig7.set_size_inches(12,8, forward=True)
fig7.set_dpi = 300
plt.show()

'''Plots Drive force and speed'''
fig8, ax8 = plt.subplots()
ax8.scatter(speed[:2], peak_force[:2], marker = "*", s =90, color = 'red')
ax8.scatter(speed[2:7], peak_force[2:7], s = 60, color = 'red', label = 'Fastball')
ax8.annotate("No Stride", (speed[0],peak_force[0]), ha='center', va='bottom', fontsize=14)
ax8.annotate("No Arm", (speed[1],peak_force[1]), ha='center', va='bottom', fontsize=14)
ax8.set_title("Drive Force vs Pitch Speed", fontweight= "bold", fontsize = 16)
ax8.set_xlabel('Pitch Speed [mph]', fontweight= "bold")
ax8.set_ylabel('Force [N]',fontweight= "bold")
ax8.set_ylim([-3,350])
ax8.set_xlim([34,55])
a, b = np.polyfit(speed[2:7], peak_force[2:7], 1)
ax8.plot(speed[2:7], a*speed[2:7]+b)
x_mid = np.mean(speed[2:7].values)
y_mid = a * x_mid + b
ax8.text(x_mid-2, y_mid-25, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)
ax8.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
'''Uncomment next 6 lines for change up data'''
'''---------------------------------------------------------------'''
ax8.scatter(change, peak_force[7:], s=60, label = 'Change Up')
a, b = np.polyfit(change, peak_force[7:], 1)
ax8.plot(change, a*change+b)
x_mid = np.mean(change)
y_mid = a * x_mid + b
ax8.text(x_mid-2, y_mid-25, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)
ax8.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
''''------------------------------------------------------------'''
fig8.set_size_inches(12,8, forward=True)
fig8.set_dpi = 300
plt.show()

'''Plots stride and speed'''
fig9, ax9 = plt.subplots()
ax9.scatter(speed[:2], stride[:2], marker = "*", s =90, color = 'red')
ax9.scatter(speed[2:7], stride[2:7], s = 60, color = 'red', label = 'Fastball')
ax9.annotate("No Stride", (speed[0],stride[0]), ha='center', va='bottom', fontsize=14)
ax9.annotate("No Arm", (speed[1],stride[1]), ha='center', va='bottom', fontsize=14)
ax9.set_title("Stride Distance vs Pitch Speed", fontweight= "bold", fontsize = 16)
ax9.set_xlabel('Pitch Speed [mph]', fontweight= "bold")
ax9.set_ylabel('Stride [inches]',fontweight= "bold")
ax9.set_ylim([-3,80])
ax9.set_xlim([34,55])
a, b = np.polyfit(speed[2:7], stride[2:7], 1)
ax9.plot(speed[2:7], a*speed.iloc[2:7]+b)
x_mid = np.mean(speed[2:7].values)
y_mid = a * x_mid + b
ax9.text(x_mid-2, y_mid-5, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)
ax9.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
'''Uncomment next 6 lines for change up data'''
'''---------------------------------------------------------------'''
ax9.scatter(change, c_stride, s=60, label = 'Change Up')
a, b = np.polyfit(change, c_stride, 1)
ax9.plot(change, a*change+b)
x_mid = np.mean(change)
y_mid = a * x_mid + b
ax9.text(x_mid-3, y_mid-2.9, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)
''''------------------------------------------------------------'''
ax9.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
fig9.set_size_inches(12,8, forward=True)
fig9.set_dpi = 300
plt.show()
#delete these to uncomment section
