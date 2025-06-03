import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

pdf_name = "Combined_plots.pdf"

def plot_drive_force(players, level, speed):
#    with PdfPages(pdf_name) as pdf:  
    fig, ax = plt.subplots()
    if speed == 'Change Up':
        for n in players:
            '''Plots drive force and speed for change up'''            
            x = np.array(players[n]['CH Peak Force'])
            y = np.array(players[n]['CH Speed'])
            ax.scatter(x, y,  s = 60, label = n)       
            a, b = np.polyfit(x, y,  1)
            ax.plot(x, a*x+b)
            x_mid = np.mean(x)
            y_mid = a * x_mid + b
            ax.text(x_mid, y_mid, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)
#            ax.set_xlim([180,340])
    else:
        for n in players:
            '''Plots drive force and speed for fastball'''
            if n == 'Kira':
                x = np.array(players[n]['FB Peak Force'][1:])
                y = np.array(players[n]['FB Speed'][1:])
            else:
                x = np.array(players[n]['FB Peak Force'])
                y = np.array(players[n]['FB Speed'])
            ax.scatter(x, y,  s = 60, label = n)       
            a, b = np.polyfit(x, y,  1)
            ax.plot(x, a*x+b)
            x_mid = np.mean(x)
            y_mid = a * x_mid + b
            if players[n]['Experience'] == 'Collegiate':
                ax.text(x_mid, y_mid, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)
 #               ax.set_xlim([159,260])
            else:
                ax.text(x_mid, y_mid, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)    
#                ax.set_xlim([90,350])
#        ax.set_ylim([18,34])
    ax.set_title(speed+" Drive Force vs Pitch Speed ("+level+')', fontweight= "bold", fontsize = 20)
    ax.set_xlabel('Force [N]', fontsize= 16, fontweight= "bold")
    ax.set_ylabel('Pitch Speed [mph]', fontsize= 16, fontweight= "bold")
    ax.legend(bbox_to_anchor = (1.13, 0.55), loc='center right')
    fig.set_size_inches(12,8, forward=True)
    fig.set_dpi = 300
    pdf.savefig(fig)
    plt.show()
    return

def plot_arm_velocity(players, level, speed): 
#    with PdfPages(pdf_name) as pdf: 
    fig1, ax1 = plt.subplots()
    if speed == 'Change Up':
        for n in players:
            '''Plots arm speed and speed for change up'''            
            x = np.array(players[n]['CH Peak Vel'])
            y = np.array(players[n]['CH Speed'])
            ax1.scatter(x, y,  s = 60, label = n)       
            a, b = np.polyfit(x, y,  1)
            ax1.plot(x, a*x+b)
            x_mid = np.mean(x)
            y_mid = a * x_mid + b
            if n == 'Kass':
                ax1.text(x_mid-50, y_mid-.5, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)
            else:
                ax1.text(x_mid, y_mid, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)
#            ax1.set_xlim([449,700])
    else:
        for n in players:
            '''Plots arm speed and speed for fastball'''
            if n == 'Kira':
                x = np.array(players[n]['FB Peak Vel'][1:])
                y = np.array(players[n]['FB Speed'][1:])
            else:
                x = np.array(players[n]['FB Peak Vel'])
                y = np.array(players[n]['FB Speed'])
            ax1.scatter(x, y,  s = 60, label = n)       
            a, b = np.polyfit(x, y,  1)
            ax1.plot(x, a*x+b)
            x_mid = np.mean(x)
            y_mid = a * x_mid + b
            if players[n]['Experience'] == 'Collegiate':
                ax1.text(x_mid, y_mid, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)
#                ax1.set_xlim([630,680])
            else:
                ax1.text(x_mid, y_mid, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)#    ax1.set_xlim([90,350])
#                ax1.set_xlim([500,760])
    ax1.set_title(speed+" Arm Speed vs Pitch Speed ("+level+')', fontweight= "bold", fontsize = 20)
    ax1.set_xlabel('Arm Velocity [m/s]', fontsize= 16, fontweight= "bold")
    ax1.set_ylabel('Pitch Speed [mph]', fontsize= 16, fontweight= "bold")
    ax1.legend(bbox_to_anchor = (1.13, 0.55), loc='center right')
    fig1.set_size_inches(12,8, forward=True)
    fig1.set_dpi = 300
    pdf.savefig(fig1)
    plt.show()
    return

def plot_stride(players, level, speed):
#    with PdfPages(pdf_name) as pdf:     
    fig1, ax1 = plt.subplots()
    if speed == 'Change Up':
        for n in players:
            '''Plots stride length and speed for change up'''            
            x = np.array(players[n]['CH Stride'])
            y = np.array(players[n]['CH Speed'])
            ax1.scatter(x, y,  s = 60, label = n)       
            a, b = np.polyfit(x, y,  1)
            ax1.plot(x, a*x+b)
            x_mid = np.mean(x)
            y_mid = a * x_mid + b
            ax1.text(x_mid, y_mid, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)

    else:
        for n in players:
            '''Plots stride length and speed for fastball'''
            if n == 'Kira':
                x = np.array(players[n]['FB Stride'][1:])
                y = np.array(players[n]['FB Speed'][1:])
            else:
                x = np.array(players[n]['FB Stride'])
                y = np.array(players[n]['FB Speed'])
            ax1.scatter(x, y,  s = 60, label = n)       
            a, b = np.polyfit(x, y,  1)
            ax1.plot(x, a*x+b)
            x_mid = np.mean(x)
            y_mid = a * x_mid + b
            if players[n]['Experience'] == 'Collegiate':
                ax1.text(x_mid, y_mid, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)
            else:
                if n == 'Kira':
                    ax1.text(x_mid-2, y_mid, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)
                elif n == 'Alex':
                    ax1.text(x_mid-2, y_mid+.5, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)
                else:
                    ax1.text(x_mid-2, y_mid, 'y = ' + '{:.2f}'.format(b) + ' + {:.4f}'.format(a) + 'x', weight= "bold",size=14)
#                    ax1.set_xlim([35,65])
#    ax.set_ylim([18,34])
    ax1.set_title(speed+" Stride Distance vs Pitch Speed ("+level+')', fontweight= "bold", fontsize = 20)
    ax1.set_xlabel('Stride [in]', fontsize= 16, fontweight= "bold")
    ax1.set_ylabel('Arm Velocity [m/s]', fontsize= 16, fontweight= "bold")
    ax1.legend(bbox_to_anchor = (1.13, 0.55), loc='center right')
    fig1.set_size_inches(12,8, forward=True)
    fig1.set_dpi = 300
    pdf.savefig(fig1)
    plt.show()
    return

def fill_player_data_adv(i, data):
    player_data = {
        'Experience': data["Experience"].iloc[i],
        'FB Peak Vel': data["Peak Velocity Fastball"].iloc[i:i+5],
        'FB Peak Force': data["Peak Force Fastball"].iloc[i:i+5],
        'FB Speed': data["Fastball Speed"].iloc[i:i+5],
        'FB Stride': data["Fastball Stride"].iloc[i:i+5],
        'CH Peak Vel': data["Peak Velocity Change"].iloc[i:i+5],
        'CH Peak Force': data["Peak Force Change"].iloc[i:i+5],
        'CH Speed': data["Change Speed"].iloc[i:i+5],
        'CH Stride': data["Fastball Stride"].iloc[i:i+5]
        }
    return player_data

def fill_player_data_rec(i, data):
    player_data = {
        'Experience': data["Experience"].iloc[i],
        'FB Peak Vel': data["Peak Velocity Fastball"].iloc[i:i+5],
        'FB Peak Force': data["Peak Force Fastball"].iloc[i:i+5],
        'FB Speed': data["Fastball Speed"].iloc[i:i+5],
        'FB Stride': data["Fastball Stride"].iloc[i:i+5]
        }
    return player_data

# Import csv data 
data = pd.read_csv("C:/Users/Jennifer/Downloads/Peak Values - Sheet1.csv", na_values=['None'])
name = data[data['Name'].notna()]["Name"]
coll_player = {}
comp_rec_player ={}
rec_player = {}
nov_player = {}
i = 0
for nm in name:
#    print("i = ", i)
    if isinstance(i, int) and i < len(data):
        exp = data["Experience"].iloc[i]
        if exp == 'Collegiate':            
            coll_player[nm] = fill_player_data_adv( i, data)
            i += 5
        elif exp == 'Comp Rec':
            comp_rec_player[nm] = fill_player_data_adv( i, data)
            i += 5
        elif exp == 'Rec':
            rec_player[nm] = fill_player_data_rec(i, data)
            i += 5    
        elif exp == 'Novice':
            nov_player[nm] = fill_player_data_rec(i, data)
            i += 5 
    else:
        print(f"Warning: Index {i} is out of bounds or not an integer.")
        print("i = ", i)
with PdfPages(pdf_name) as pdf: 
    plot_arm_velocity(nov_player, 'No Experience','Fastball')
    plot_drive_force(nov_player, 'No Experience', 'Fastball')
    plot_stride(nov_player, 'No Experience', 'Fastball')    
    
    plot_arm_velocity(rec_player, 'Novice','Fastball')
    plot_drive_force(rec_player, 'Novice', 'Fastball')
    plot_stride(rec_player, 'Novice', 'Fastball')

    plot_arm_velocity(comp_rec_player, 'Mid-Level','Fastball')
    plot_drive_force(comp_rec_player, 'Mid-Level', 'Fastball')
    plot_stride(comp_rec_player, 'Mid-Level', 'Fastball')
    plot_arm_velocity(comp_rec_player, 'Mid-Level','Change Up')
    plot_drive_force(comp_rec_player, 'Mid-Level', 'Change Up')
    plot_stride(comp_rec_player, 'Mid-Level', 'Change Up')
    
    plot_arm_velocity(coll_player, 'Collegiate', 'Fastball')
    plot_drive_force(coll_player, 'Collegiate', 'Fastball')
    plot_stride(coll_player, 'Collegiate', 'Fastball')
    plot_arm_velocity(coll_player, 'Collegiate', 'Change Up')
    plot_drive_force(coll_player, 'Collegiate', 'Change Up')
    plot_stride(coll_player, 'Collegiate', 'Change Up')
