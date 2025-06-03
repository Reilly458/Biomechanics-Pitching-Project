import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from matplotlib.backends.backend_pdf import PdfPages


"""*****************************************************"""
"""*******   Put name of export file here   ************"""
""" Failure to change the file name will overwrite the  """
"""  last persons pdf. I've done it, more than once :(  """
"""*****************************************************"""
pdf_name = "Drew_plots.pdf"

# Import csv data 
data = pd.read_csv("C:/Users/Jennifer/Downloads/Project times - Drew.csv")

size_wrist = 79384        #Number of arm cells with data
size_chest = 81576        #Number of chest cells with data
raw_wrist = data['Wrist IMU (Gyroscope) z'].iloc[:size_wrist]
raw_chest = data['Chest IMU (linear acceleration) z'].iloc[:size_chest]
arm_length = data['Arm Length [m]'].iloc[0]
force = data['Parallel Force (N)']
force_time = data['Force Plate (Parallel force) Time (s)']
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
cutoff_freq = 2

# Define low-pass filter
def highpass_filter(data, cutoff_freq, sample_rate_IMU, order=4):
    nyquist = 0.5 * sample_rate_IMU
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)

#Filter wrist data with highpass filter
wrist_filtered = highpass_filter(raw_wrist, cutoff_freq, sample_rate_IMU)

# Negate Filtered Wrist IMU z
dataNeg = wrist_filtered*-1 

''' Calculate tangential velocity by multiplying negated Wrist IMU x 
    by arm length in m/s'''
wrist = dataNeg *arm_length    #For inverted wrist data
#wrist = wrist_filtered * arm_length    #For non-inverted wrist data
#wrist = raw_wrist * arm_length
"""May need to adjust the height and distance on peak detection"""
wrist_peaks, _ = find_peaks(wrist, height=100, distance = 1000)   
print("Wrist peaks")
print(wrist[wrist_peaks])
#peak_values = np.round(wrist[wrist_peaks], 2)
#unwanted_peaks = {193.64, 171.94}
#filtered_peaks = wrist_peaks[~np.isin(peak_values, list(unwanted_peaks))]
#wrist_peaks = filtered_peaks

''' Calculate time values in seconds from timestamp with 208 sample rate
    and zeros the time'''
time = (data['Chest IMU (linear acceleration) timestamp']-data['Chest IMU (linear acceleration) timestamp'].iloc[0]) / 208
time = time.iloc[:size_chest]
wrist_time = (data['Wrist IMU (Gyroscope) timestamp']-data['Wrist IMU (Gyroscope) timestamp'].iloc[0])/208
wrist_time = wrist_time.iloc[:size_wrist]
''' Calculate chest velocity from acceleration v = at in m/s
    You may need to adjust the offset on the raw_chest value'''

raw_chest_filtered = highpass_filter(raw_chest, cutoff_freq, sample_rate_IMU)
chest = integrate.cumulative_trapezoid(raw_chest_filtered, x=None, dx= 1/sample_rate_IMU, initial =0)
chest_peaks, _ = find_peaks(chest, height = 0.08, distance = 1000)
print("Chest peaks")
print(chest[chest_peaks])

''' Calculate total velocity = wrist + chest, an offest may need to 
    be placed on i if peaks are missing from data.'''
tot_vel = wrist #sets total velocity equal to wrist velocity
for i in range(len(chest_peaks)):
    print("Before: ", tot_vel[wrist_peaks[i]]," = ", wrist[wrist_peaks[i]], " + ",chest[chest_peaks[i]],"i= ", i )
    #Adds the peak chest value to the peak wrist value at the peak wrist values location in total velocity
    tot_vel[wrist_peaks[i]] += chest[chest_peaks[i]]    
    print("After: ", tot_vel[wrist_peaks[i]] )
tot_vel = np.round(tot_vel, 2)

# Plotting
with PdfPages(pdf_name) as pdf:  
        """If you want to add a comment to the pdf use pdf.attach_note('')
        It only seems to work once and in random places, must be directly
        after a pdf.savefig() line, so put everything you want in 1 comment."""
        fig1, ax1 = plt.subplots()
        ax1.plot(wrist_samples, data['Wrist IMU (Gyroscope) z'].iloc[:size_wrist]*-1)
        ax1.set_title('Raw Wrist IMU Velocity')
        ax1.set_xlabel('Sample #')
        ax1.set_ylabel('Wrist IMU angular velocity in z [rad/s]')
        fig1.set_size_inches(15,10, forward=True)
        fig1.set_dpi = 300 
        pdf.savefig(fig1)
        plt.show()

        fig01, ax01 = plt.subplots()
        ax01.plot(wrist_samples, wrist_filtered)
        ax01.set_title('Raw Filtered Wrist IMU Velocity')
        ax01.set_xlabel('Sample #')
        ax01.set_ylabel('Wrist IMU angular velocity in z [rad/s]')
        fig01.set_size_inches(15,10, forward=True)
        fig01.set_dpi = 300 
        pdf.savefig(fig01)
        plt.show()

        '''Check that data looks oriented correctly, if not, uncomment the dataNeg wrist
           and comment out the calculation that doesn't use the negated data and re-run
           the 2 plots above''' 
           
        fig2, ax2 = plt.subplots()
        ax2.plot(wrist_samples, dataNeg, color= 'teal' , label= "Wrist IMU")
        ax2.set_title("Filtered Wrist IMU Velocity Negated")
        ax2.set_xlabel("Samples (#)")
        ax2.set_ylabel('Wrist IMU (Gyroscope) z [rad/s]')
        fig2.set_size_inches(15,10, forward=True)
        fig2.set_dpi = 300  
        pdf.savefig(fig2)
        plt.show()

        fig3, ax3 = plt.subplots()
        ax3.plot(wrist_time, wrist, color= 'blue' , label= "Wrist Tangential Velocity")
        ax3.set_title('Filtered Wrist Tangential Velocity')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel('Wrist IMU (Gyroscope) z [m/s]')
        fig3.set_size_inches(15,10, forward=True)
        fig3.set_dpi = 300 
        pdf.savefig(fig3) 
        plt.show()

        fig4, ax4 = plt.subplots()
        ax4.plot(time, raw_chest_filtered, color= 'orange' , label= "Chest IMU")
        ax4.set_title('Raw Chest Linear Acceleration')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel('Chest IMU (Gyroscope) z [m/s^2]')
        fig4.set_size_inches(15,10, forward=True)
        fig4.set_dpi = 300  
        pdf.savefig(fig4)
        plt.show()

        fig10, ax10 = plt.subplots()
        ax10.plot(time, chest, color= 'red' , label= "Chest IMU")
        ax10.set_title('Chest Linear Velocity')
        ax10.set_xlabel("Time (s)")
        ax10.set_ylabel('Chest IMU (Gyroscope) Velocity z [m/s]')
        fig10.set_size_inches(15,10, forward=True)
        fig10.set_dpi = 300  
        pdf.savefig(fig10)
        plt.show()

        fig5, ax5 = plt.subplots()
        """The line below finds the peak values and puts them in an array.
           You may need to adjust the height threshold or add a vertical
           threshold to capture the desired peaks."""
        peaks, _ = find_peaks(force, height=150, distance = 10)
#        peak_values = force[peaks]
#        unwanted_peaks = {335.24, 302.94}
#        filtered_peaks = peaks[~np.isin(peak_values, list(unwanted_peaks))]
#        peaks = filtered_peaks
        ax5.plot(force_time, force, color= 'green' , label= "Parallel Force (N)")
        ax5.set_title('Raw Force', fontsize = 20, fontweight = "bold")
        ax5.set_xlabel("Time (s)", fontsize = 16, fontweight = "bold")
        ax5.set_ylabel('Parallel Force [N]', fontsize = 16, fontweight = "bold")
        '''The next 3 lines prints the peak values to the plot'''
        for i, txt in enumerate(peaks):
            ax5.annotate(force[txt], (force_time[txt], force[txt]), ha='center', va='bottom', fontsize = 14, fontweight = 'bold') 
        ax5.plot(force_time[peaks] , force[peaks], 'o', markersize =6, color = 'red')
#        ax5.set_xlim([-1,140])      #sets x-axis range, adjust as needed
#        ax5.set_ylim([,])      #sets y-axis range, adjust as needed
        fig5.set_size_inches(15,10, forward=True)
        fig5.set_dpi = 300 
        pdf.savefig(fig5)
        plt.show()
        peak_force = np.zeros(len(peaks) + 1, dtype=float)
        peak_force [1:]= force[peaks] 
        print("Peak Force Array")
        print(peak_force)

        """Stop here and analyze graphs and change plot settings as needed. 
           Once the graphs look right, uncomment the section below. The plots
           below create the correlation plots and trendlines. Some values will
           need to be adjusted from file to file"""
        """-------------------------------------------------------------------------"""#add 3 apostropes here to uncomment below 

        '''Plots total velocity'''   
        fig6, ax6 = plt.subplots()
        peaks_v, _ = find_peaks(tot_vel, distance = 1000, height = 100)
        ax6.plot(wrist_time, tot_vel, color= 'black' , label= "Total Velocity")
        ax6.set_title('Total Velocity', fontsize = 20, fontweight= "bold")
        ax6.set_xlabel("Time (s)", fontsize= 16, fontweight= "bold")
        ax6.set_ylabel('Total Velocity [m/s]', fontsize = 16,fontweight= "bold")
        ax6.set_xlim([-20,1850])      #sets x-axis range, adjust as needed
#        ax6.set_ylim([-450,410])      #sets y-axis range, adjust as needed
        peak_values = tot_vel[peaks_v]
#        unwanted_peaks = {41.07, 42.17, 35.69}
 #       filtered_peaks = peaks_v[~np.isin(peak_values, list(unwanted_peaks))]     
        '''Below filters out a peak we don't want. Use print(tot_vel[peaks_v[]]) to get
           the value of the peak you want ignored and plug into filtered_peaks line.
           If you don't need to ignore peaks, comment out.'''
        ''''-------------------------------------------------------------------------'''   
     #   print("Filter peak: ", tot_vel[peaks_v[0]]) #Uncomment to print filtered peak value
        filtered_peaks = peaks_v[peak_values != 247.56 ]      #insert filtered peak value here
        for tx in filtered_peaks:
            ax6.annotate(tot_vel[tx], (wrist_time[tx], tot_vel[tx]), ha='center', va='bottom', fontsize = 14, fontweight = 'bold') 
        ax6.plot(wrist_time[filtered_peaks] , tot_vel[filtered_peaks], 'o', markersize =6, color = 'red')
        peak_vel = tot_vel[filtered_peaks]
        ''''-------------------------------------------------------------------------'''
#        peak_vel = tot_vel[peaks_v]     #Comment out if filtering a peak out
        fig6.set_size_inches(25,15, forward=True)
        fig6.set_dpi = 300  
        pdf.savefig(fig6)
        plt.show()
        print("Peak Velocity Array")
        print(peak_vel)

        '''Plots arm velocity and speed'''
        fig7, ax7 = plt.subplots()
        ax7.scatter(peak_vel[:2],speed[:2],  marker = "*", s =90, color = 'red')
        ax7.scatter(peak_vel[2:7],speed[2:7],  s = 60, color = 'red', label = 'Fastball')
        ax7.annotate("No Stride", (peak_vel[0], speed[0]), ha='center', va='bottom', fontsize=14)
        ax7.annotate("No Arm", (peak_vel[1], speed[1]), ha='center', va='bottom', fontsize=14)
        ax7.set_title("Arm Speed vs Pitch Speed", fontweight= "bold", fontsize = 20)
        ax7.set_xlabel('Arm Velocity [m/s]', fontsize= 16, fontweight= "bold")
        ax7.set_ylabel('Pitch Speed [mph]', fontsize= 16, fontweight= "bold")
        ax7.set_xlim([98,136])      #sets x-axis range, adjust as needed
 #       ax7.set_ylim([33,55])      #sets y-axis range, adjust as needed
        a, b = np.polyfit(peak_vel[2:7], speed[2:7], 1)
        ax7.plot(peak_vel[2:7], a*peak_vel[2:7]+b)
        x_mid = np.mean(peak_vel[2:7])
        y_mid = a * x_mid + b
        #To move equation placement, +/- values to x/y_mid in line below
        ax7.text(x_mid-5, y_mid-1.3, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', size=14)
        '''Uncomment next 6 lines for change up data'''
        '''---------------------------------------------------------------
        ax7.scatter(peak_vel[7:],change , s=60, label = 'Change')
        print(len(peak_vel[7:]))
        a, b = np.polyfit(peak_vel[7:],change,  1)
        ax7.plot(peak_vel[7:], a*peak_vel[7:]+b)
        x_mid = np.mean(peak_vel[7:])
        y_mid = a * x_mid + b
        ax7.text(x_mid-90, y_mid-2, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', size=14)
        ------------------------------------------------------------'''
        ax7.legend(bbox_to_anchor = (1.13, 0.55), loc='center right')
        fig7.set_size_inches(12,8, forward=True)
        fig7.set_dpi = 300
        pdf.savefig(fig7)
        plt.show()

        '''Plots Drive force and speed'''
        fig8, ax8 = plt.subplots()
        ax8.scatter(peak_force[:2], speed[:2], marker = "*", s =90, color = 'red')
        ax8.scatter(peak_force[2:7], speed[2:7],  s = 60, color = 'red', label = 'Fastball')
        ax8.annotate("No Stride", (peak_force[0], speed[0]), ha='left', va='bottom', fontsize=14)
        ax8.annotate("No Arm", (peak_force[1], speed[1]), ha='center', va='bottom', fontsize=14)
        ax8.set_title("Drive Force vs Pitch Speed", fontweight= "bold", fontsize = 20)
        ax8.set_xlabel('Force [N]', fontsize= 16, fontweight= "bold")
        ax8.set_ylabel('Pitch Speed [mph]', fontsize= 16, fontweight= "bold")
 #       ax8.set_xlim([-20,401])
 #      ax8.set_ylim([33,55])
        a, b = np.polyfit(peak_force[2:7], speed[2:7],  1)
        ax8.plot(peak_force[2:7].astype(float), a*peak_force[2:7].astype(float)+b)
        x_mid = np.mean(peak_force[2:7].astype(float))
        y_mid = a * x_mid + b
        ax8.text(x_mid-50, y_mid-2, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', size=14)
        '''Uncomment next 6 lines for change up data'''
        '''---------------------------------------------------------------
        ax8.scatter(peak_force[7:], change, s=60, label = 'Change')
        a, b = np.polyfit(peak_force[7:],change, 1)
        ax8.plot(peak_force[7:], a*peak_force[7:]+b)
        x_mid = np.mean(peak_force[7:])
        y_mid = a * x_mid + b
        ax8.text(x_mid-50, y_mid-1.7, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', size=14)
        ------------------------------------------------------------'''
        ax8.legend(bbox_to_anchor = (1.13, 0.55), loc='center right')
        fig8.set_size_inches(12,8, forward=True)
        fig8.set_dpi = 300
        pdf.savefig(fig8)
        plt.show()

        '''Plots stride and speed'''
        fig9, ax9 = plt.subplots()
        ax9.scatter(stride[1:2], speed[1:2], marker = "*", s =90, color = 'red')
        ax9.scatter(stride[2:7], speed[2:7], s = 60, color = 'red', label = 'Fastball')
        ax9.annotate("No Stride", (stride[0], speed[0]), ha='left', va='bottom', fontsize=14)
        ax9.annotate("No Arm", (stride[1], speed[1]), ha='left', va='bottom', fontsize=14)
        ax9.set_title("Stride Distance vs Pitch Speed", fontweight= "bold", fontsize = 20)
        ax9.set_xlabel('Stride [inches]', fontsize= 16, fontweight= "bold")
        ax9.set_ylabel('Pitch Speed [mph]', fontsize= 16,fontweight= "bold")
#        ax9.set_xlim([23.75,36.1])
#        ax9.set_ylim([33,55.1])
        a, b = np.polyfit(stride[2:7], speed[2:7], 1)
        ax9.plot(stride[2:7], a*stride[2:7]+b)
        x_mid = np.mean(stride[3:7].values)
        y_mid = a * x_mid + b
        ax9.text(x_mid-3, y_mid-1, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', size=14)
        ax9.legend(bbox_to_anchor = (1.15, 0.6), loc='center right')
        '''Uncomment next 6 lines for change up data'''
        '''---------------------------------------------------------------
        ax9.scatter(c_stride, change, s=60, label = 'Change')
        a, b = np.polyfit(c_stride, change, 1)
        ax9.plot(c_stride, a*c_stride+b)
        x_mid = np.mean(c_stride)
        y_mid = a * x_mid + b
        ax9.text(x_mid-5, y_mid+1.7, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', size=14)
        ------------------------------------------------------------'''
        ax9.legend(bbox_to_anchor = (1.13, 0.55), loc='center right')
        fig9.set_size_inches(12,8, forward=True)
        fig9.set_dpi = 300
        pdf.savefig(fig9)
        plt.show()
        #delete these to uncomment section